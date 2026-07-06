#include "metal3-backend.h"
#include "nvrhi/nvrhi.h"
#include <Metal/Metal.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#define IR_PRIVATE_IMPLEMENTATION
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>

namespace nvrhi::metal3 
{
    CommandList::CommandList(class Device* device, const MTL3Context& context, const CommandListParameters& params)
        : m_Context(context)
            , m_Device(device)
            , m_UploadManager(context, params.uploadChunkSize, 0, false)
            , m_Desc(params) {}
        
    CommandList::~CommandList() {};

    Object CommandList::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::Nvrhi_Metal3_CommandList:
            return Object(this);
        default:
            return nullptr;
        }
    }

    // crates a new command buffer, invalidates compute and graphics state, and the encoders
    void CommandList::open()
    {
        trackedCmdBuffer = [m_Context.commonQueue commandBuffer];
        m_CurrentGraphicsStateValid = false;
        m_CurrentComputeStateValid = false;
        m_RenderEncoder = nil;
        m_ComputeEncoder = nil;
    }
    // closing it commits the command buffer to queue, and invalidates the encoders, command buffers
    void CommandList::close()
    {
        endEncoding();
        const MTL3Context* context = &m_Context;
        
        [trackedCmdBuffer commit];
    }

    void CommandList::clearState()
    {
        m_CurrentGraphicsState = GraphicsState();
        m_CurrentComputeState = ComputeState();
        m_CurrentGraphicsStateValid = false;
        m_CurrentComputeStateValid = false;
        endEncoding();
    }

    void CommandList::clearTextureFloat(ITexture* t, TextureSubresourceSet subresources, const Color& clearColor)
    {
        (void)subresources;
        auto* texture = static_cast<Texture*>(t);
        if (!texture || !texture->texture)
            return;

        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = texture->texture;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = MTLClearColorMake(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
        endEncoding();
        id<MTLRenderCommandEncoder> encoder = [trackedCmdBuffer renderCommandEncoderWithDescriptor:rp];
        [encoder endEncoding];
    }
    
    void CommandList::clearDepthStencilTexture(ITexture* t, TextureSubresourceSet subresources, bool clearDepth, float depth, bool clearStencil, uint8_t stencil)
    {
        (void)subresources; (void)clearStencil; (void)stencil;
        auto* texture = static_cast<Texture*>(t);
        if (!texture || !texture->texture || !clearDepth)
            return;

        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.depthAttachment.texture = texture->texture;
        rp.depthAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        rp.depthAttachment.clearDepth = depth;
        endEncoding();
        id<MTLRenderCommandEncoder> encoder = [trackedCmdBuffer renderCommandEncoderWithDescriptor:rp];
        [encoder endEncoding];
    }

    void CommandList::clearTextureUInt(ITexture* t, TextureSubresourceSet subresources, uint32_t clearColor)
    {
        Color color(float(clearColor), 0.f, 0.f, 0.f);
        clearTextureFloat(t, subresources, color);
    }

    void CommandList::copyTexture(ITexture* dest, const TextureSlice& destSlice, ITexture* src, const TextureSlice& srcSlice)
    {
        auto* d = static_cast<Texture*>(dest);
        auto* s = static_cast<Texture*>(src);
        if (!d || !s || !d->texture || !s->texture)
            return;

        const TextureSlice ds = destSlice.resolve(d->desc);
        const TextureSlice ss = srcSlice.resolve(s->desc);
        const uint32_t srcMipWidth = std::max(1u, s->desc.width >> ss.mipLevel);
        const uint32_t srcMipHeight = std::max(1u, s->desc.height >> ss.mipLevel);
        const uint32_t srcMipDepth = s->desc.dimension == TextureDimension::Texture3D
            ? std::max(1u, s->desc.depth >> ss.mipLevel)
            : 1u;
        const uint32_t dstMipWidth = std::max(1u, d->desc.width >> ds.mipLevel);
        const uint32_t dstMipHeight = std::max(1u, d->desc.height >> ds.mipLevel);
        const uint32_t dstMipDepth = d->desc.dimension == TextureDimension::Texture3D
            ? std::max(1u, d->desc.depth >> ds.mipLevel)
            : 1u;

        if (ss.x >= srcMipWidth || ss.y >= srcMipHeight || ss.z >= srcMipDepth ||
            ds.x >= dstMipWidth || ds.y >= dstMipHeight || ds.z >= dstMipDepth)
            return;

        const MTLSize copySize = MTLSizeMake(
            std::min({ ss.width, ds.width, srcMipWidth - ss.x, dstMipWidth - ds.x }),
            std::min({ ss.height, ds.height, srcMipHeight - ss.y, dstMipHeight - ds.y }),
            std::min({ ss.depth, ds.depth, srcMipDepth - ss.z, dstMipDepth - ds.z }));

        if (copySize.width == 0 || copySize.height == 0 || copySize.depth == 0)
            return;

        endEncoding();
        id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
        [blit copyFromTexture:s->texture
                  sourceSlice:ss.arraySlice
                  sourceLevel:ss.mipLevel
                 sourceOrigin:MTLOriginMake(ss.x, ss.y, ss.z)
                   sourceSize:copySize
                    toTexture:d->texture
             destinationSlice:ds.arraySlice
             destinationLevel:ds.mipLevel
            destinationOrigin:MTLOriginMake(ds.x, ds.y, ds.z)];
        [blit endEncoding];
    }

    void CommandList::writeTexture(ITexture* dest, uint32_t arraySlice, uint32_t mipLevel, const void* data, size_t rowPitch, size_t depthPitch)
    {
        auto* texture = static_cast<Texture*>(dest);
        if (!texture || !texture->texture || !data)
            return;

        const TextureDesc& desc = texture->desc;
        if (mipLevel >= desc.mipLevels || arraySlice >= desc.arraySize)
            return;

        const FormatInfo& formatInfo = getFormatInfo(desc.format);
        const uint32_t mipWidth = std::max(1u, desc.width >> mipLevel);
        const uint32_t mipHeight = std::max(1u, desc.height >> mipLevel);
        const uint32_t mipDepth = desc.dimension == TextureDimension::Texture3D
            ? std::max(1u, desc.depth >> mipLevel)
            : 1u;

        // Metal's buffer-to-texture copy describes rows in bytes, but NVRHI rowPitch
        // describes the caller's source layout. For block-compressed formats, one
        // logical row is a row of compression blocks, not pixels, so compute the
        // tightly packed GPU row size from block columns and bytes per block.
        const uint32_t blockCols = (mipWidth + formatInfo.blockSize - 1u) / formatInfo.blockSize;
        const uint32_t blockRows = (mipHeight + formatInfo.blockSize - 1u) / formatInfo.blockSize;
        const size_t naturalRowPitch = size_t(blockCols) * formatInfo.bytesPerBlock;

        if (rowPitch == 0)
            rowPitch = naturalRowPitch;
        if (rowPitch < naturalRowPitch)
            return;

        const size_t naturalBytesPerImage = naturalRowPitch * blockRows;
        if (depthPitch == 0)
            depthPitch = rowPitch * blockRows;
        if (depthPitch < rowPitch * blockRows)
            return;

        const size_t copySize = naturalBytesPerImage * mipDepth;
        id<MTLBuffer> upload = [m_Context.device newBufferWithLength:copySize options:MTLResourceStorageModeShared];
        if (!upload)
            return;

        // pack only the meaningful row bytes into the upload buffer. The caller's
        // rows/depth slices may include padding, while the Metal copy below uses a
        // tightly packed sourceBytesPerRow/sourceBytesPerImage layout.
        auto* destBytes = static_cast<uint8_t*>([upload contents]);
        const auto* srcBytes = static_cast<const uint8_t*>(data);
        for (uint32_t z = 0; z < mipDepth; ++z)
        {
            uint8_t* destImage = destBytes + naturalBytesPerImage * z;
            const uint8_t* srcImage = srcBytes + depthPitch * z;
            for (uint32_t row = 0; row < blockRows; ++row)
            {
                memcpy(destImage + naturalRowPitch * row, srcImage + rowPitch * row, naturalRowPitch);
            }
        }

        m_ReferencedNativeBuffers.push_back(upload);
        endEncoding();
        id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
        [blit copyFromBuffer:upload
                 sourceOffset:0
            sourceBytesPerRow:naturalRowPitch
          sourceBytesPerImage:naturalBytesPerImage
                   sourceSize:MTLSizeMake(mipWidth, mipHeight, mipDepth)
                    toTexture:texture->texture
             destinationSlice:arraySlice
             destinationLevel:mipLevel
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];
    }

    void CommandList::resolveTexture(ITexture* dest, const TextureSubresourceSet& dstSubresources, ITexture* src, const TextureSubresourceSet& srcSubresources)
    {
        auto* d = static_cast<Texture*>(dest);
        auto* s = static_cast<Texture*>(src);
        if (!d || !s || !d->texture || !s->texture)
            return;

        if (s->desc.sampleCount <= 1 || d->desc.sampleCount != 1 || s->desc.format != d->desc.format)
            return;

        const FormatInfo& formatInfo = getFormatInfo(d->desc.format);
        if (formatInfo.kind == FormatKind::DepthStencil)
            return;

        // NVRHI resolveTexture can target a range of mip levels and array slices.
        // Metal resolves one attachment subresource per render pass, so resolve
        // both ranges first and then issue one tiny render pass for each pair.
        const TextureSubresourceSet dstSR = dstSubresources.resolve(d->desc, false);
        const TextureSubresourceSet srcSR = srcSubresources.resolve(s->desc, false);
        if (dstSR.numArraySlices != srcSR.numArraySlices || dstSR.numMipLevels != srcSR.numMipLevels)
            return;

        endEncoding();

        for (ArraySlice arrayIndex = 0; arrayIndex < dstSR.numArraySlices; ++arrayIndex)
        {
            for (MipLevel mipLevel = 0; mipLevel < dstSR.numMipLevels; ++mipLevel)
            {
                // Metal does not expose MSAA resolve as a blit operation. The
                // source MSAA texture is attached for loading, and the resolved
                // single-sample texture is written by the resolve store action.
                MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
                MTLRenderPassColorAttachmentDescriptor* colorAttachment = rp.colorAttachments[0];
                colorAttachment.texture = s->texture;
                colorAttachment.level = mipLevel + srcSR.baseMipLevel;
                colorAttachment.slice = arrayIndex + srcSR.baseArraySlice;
                colorAttachment.loadAction = MTLLoadActionLoad;
                colorAttachment.storeAction = MTLStoreActionMultisampleResolve;
                colorAttachment.resolveTexture = d->texture;
                colorAttachment.resolveLevel = mipLevel + dstSR.baseMipLevel;
                colorAttachment.resolveSlice = arrayIndex + dstSR.baseArraySlice;

                id<MTLRenderCommandEncoder> encoder = [trackedCmdBuffer renderCommandEncoderWithDescriptor:rp];
                [encoder endEncoding];
            }
        }
    }
}
