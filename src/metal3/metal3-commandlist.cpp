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
    static bool traceMetalRuntime()
    {
        static bool enabled = [] {
            const char* value = std::getenv("LDV_METAL3_TRACE");
            return !value || std::string(value) != "0";
        }();
        return enabled;
    }

    static size_t alignUp(size_t value, size_t alignment)
    {
        if (alignment <= 1)
            return value;
        return (value + alignment - 1) & ~(alignment - 1);
    }

    // specify default chunk size a 4MB minimum
    // m_CompletedSerial tracks which submitted command buffers have finished on the GPU
    UploadManager::UploadManager(const MTL3Context& context, size_t uploadChunkSize, size_t scratchMaxMem, bool isScratchBuffer)
        : m_Context(context)
        , m_DefaultChunkSize(std::max<size_t>(uploadChunkSize, 4 * 1024 * 1024))
        , m_CompletedSerial(std::make_shared<std::atomic<uint64_t>>(0))
    {
        (void)scratchMaxMem;
        (void)isScratchBuffer;
    }
    // called, from cmdList.open, and every upload allocation made during this command (when cmdList open) list gets tagged with m_ActiveSerial
    void UploadManager::beginCommandBuffer()
    {
        m_ActiveSerial = ++m_SubmittedSerial;
        m_CurrentChunk = size_t(-1);
    }
    /*
     * called from CommandList::close() before commit.
     * It attaches a Metal completion handler: `[commandBuffer addCompletedHandler:...]`
     * When the GPU finishes this command buffer, it updates: `m_CompletedSerial = submittedSerial;`
    */
    void UploadManager::submitCommandBuffer(id<MTLCommandBuffer> commandBuffer)
    {
        if (!commandBuffer || m_ActiveSerial == 0)
            return;

        const uint64_t submittedSerial = m_ActiveSerial;
        std::shared_ptr<std::atomic<uint64_t>> completedSerial = m_CompletedSerial;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            uint64_t previousValue = completedSerial->load(std::memory_order_relaxed);
            while (previousValue < submittedSerial &&
                !completedSerial->compare_exchange_weak(previousValue, submittedSerial,
                    std::memory_order_release, std::memory_order_relaxed))
            {
            }
        }];
    }

    /*
     * this finds a chunk with enough free space, or creates a new one.
    */
    UploadManager::Chunk* UploadManager::findOrCreateChunk(size_t size, size_t alignment)
    {
        if (m_CurrentChunk != size_t(-1))
        {
            Chunk& chunk = m_Chunks[m_CurrentChunk];
            // try the current chunk it it has room for data, if yes return chunk*
            if (alignUp(chunk.offset, alignment) + size <= chunk.size)
                return &chunk;
        }

        const uint64_t completedSerial = m_CompletedSerial->load(std::memory_order_acquire);
        for (size_t index = 0; index < m_Chunks.size(); ++index)
        {
            Chunk& chunk = m_Chunks[index];
            // try if the old chunks are free now, checked with the GPU sync; if yes return chunk
            if (chunk.lastUsedSerial <= completedSerial && size <= chunk.size)
            {
                chunk.offset = 0;
                m_CurrentChunk = index;
                return &chunk;
            }
        }

        // if no free chunk found, create a new MTL buffer with the m_DefaultChunkSize, and return chunk from that memory
        const size_t chunkSize = std::max(m_DefaultChunkSize, alignUp(size, alignment));
        id<MTLBuffer> buffer = [m_Context.device newBufferWithLength:NSUInteger(chunkSize) options:MTLResourceStorageModeShared];
        if (!buffer)
        {
            m_Context.error("[nvrhi] Failed to allocate Metal upload chunk.");
            return nullptr;
        }

        Chunk chunk;
        chunk.buffer = buffer;
        chunk.cpuAddress = static_cast<uint8_t*>([buffer contents]);
        chunk.size = chunkSize;
        m_Chunks.push_back(chunk);
        m_CurrentChunk = m_Chunks.size() - 1;
        return &m_Chunks.back();
    }

    // public allocater, used for asking chunks
    UploadAllocation UploadManager::suballocate(size_t size, size_t alignment)
    {
        UploadAllocation allocation;
        if (size == 0)
            return allocation;

        if (m_ActiveSerial == 0)
            beginCommandBuffer();

        Chunk* chunk = findOrCreateChunk(size, alignment);
        if (!chunk)
            return allocation;

        const size_t offset = alignUp(chunk->offset, alignment);
        allocation.buffer = chunk->buffer;
        allocation.offset = NSUInteger(offset);
        allocation.cpuAddress = chunk->cpuAddress + offset;

        chunk->offset = offset + size;
        chunk->lastUsedSerial = m_ActiveSerial;
        return allocation;
    }

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
        m_UploadManager.beginCommandBuffer();
        m_VolatileBufferAllocations.clear();
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
        m_UploadManager.submitCommandBuffer(trackedCmdBuffer);
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

    /*
     * writeTexture computes packed row layout, asks for upload memory, writes rows into it, then does:
        copyFromBuffer:upload.buffer
        sourceOffset:upload.offset
        sourceBytesPerRow:naturalRowPitch
        sourceBytesPerImage:naturalBytesPerImage
        toTexture:texture->texture
     * So textures also avoid per-upload Metal buffer allocation.
    */
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
        UploadAllocation upload = m_UploadManager.suballocate(copySize, 256);
        if (!upload.buffer || !upload.cpuAddress)
            return;

        // pack only the meaningful row bytes into the upload buffer. The caller's
        // rows/depth slices may include padding, while the Metal copy below uses a
        // tightly packed sourceBytesPerRow/sourceBytesPerImage layout.
        auto* destBytes = static_cast<uint8_t*>(upload.cpuAddress);
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

        endEncoding();
        id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
        [blit copyFromBuffer:upload.buffer
                 sourceOffset:upload.offset
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
    void CommandList::writeBuffer(IBuffer* b, const void* data, size_t dataSize, uint64_t destOffsetBytes)
    {
        auto* buffer = static_cast<Buffer*>(b);
        if (!buffer || !data || dataSize == 0)
            return;

        if (destOffsetBytes > buffer->desc.byteSize || dataSize > buffer->desc.byteSize - destOffsetBytes)
            return;

        // if isVolatile, write into the CPU-GPU shared mem directly and return
        if (buffer->desc.isVolatile)
        {
            UploadAllocation allocation = m_UploadManager.suballocate(dataSize, 256);
            if (!allocation.buffer || !allocation.cpuAddress)
                return;

            memcpy(allocation.cpuAddress, data, dataSize);
            m_VolatileBufferAllocations[buffer] = allocation;
            return;
        }

        if (!buffer->buffer)
            return;

        // if not volatile, for normal buffer, CPU writes into upload shared memory, and then perform blit op
        UploadAllocation upload = m_UploadManager.suballocate(dataSize, 256);
        if (!upload.buffer || !upload.cpuAddress)
            return;

        memcpy(upload.cpuAddress, data, dataSize);

        endEncoding();
        id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
        [blit copyFromBuffer:upload.buffer
                sourceOffset:upload.offset
                    toBuffer:buffer->buffer
           destinationOffset:NSUInteger(destOffsetBytes)
                        size:NSUInteger(dataSize)];
        [blit endEncoding];
    }

    void CommandList::clearBufferUInt(IBuffer* b, uint32_t clearValue)
    {
        auto* buffer = static_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return;

        const size_t clearSize = size_t(buffer->desc.byteSize);
        if (clearSize == 0)
            return;

        if (clearValue == 0)
        {
            // This clear must be ordered with later GPU work. CPU writes into a
            // shared buffer can race already-encoded dispatches, which lets GPU
            // append counters accumulate across culling passes. A blit fill is
            // recorded in the command buffer, so Metal executes it before the
            // following compute encoder.
            endEncoding();
            id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
            [blit fillBuffer:buffer->buffer
                       range:NSMakeRange(0, NSUInteger(buffer->desc.byteSize))
                       value:0];
            [blit endEncoding];
            return;
        }
        // repeatable pattern does the same style blit
        const uint8_t bytePattern = uint8_t(clearValue & 0xffu);
        const bool isRepeatedBytePattern = clearValue == uint32_t(bytePattern) * 0x01010101u;
        if (isRepeatedBytePattern)
        {
            endEncoding();
            id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
            [blit fillBuffer:buffer->buffer
                       range:NSMakeRange(0, NSUInteger(clearSize))
                       value:bytePattern];
            [blit endEncoding];
            return;
        }

        // if no repeatable pattern, perform the copy from buffer with UploadManager
        const size_t wordCount = clearSize / sizeof(uint32_t);
        const size_t uploadSize = wordCount * sizeof(uint32_t);
        if (uploadSize == 0)
            return;

        UploadAllocation upload = m_UploadManager.suballocate(uploadSize, 256);
        if (!upload.buffer || !upload.cpuAddress)
            return;

        uint32_t* words = static_cast<uint32_t*>(upload.cpuAddress);
        for (size_t i = 0; i < wordCount; ++i)
            words[i] = clearValue;

        endEncoding();
        id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
        [blit copyFromBuffer:upload.buffer
                sourceOffset:upload.offset
                    toBuffer:buffer->buffer
           destinationOffset:0
                        size:NSUInteger(uploadSize)];
        [blit endEncoding];
    }

    void CommandList::copyBuffer(IBuffer* dest, uint64_t destOffsetBytes, IBuffer* src, uint64_t srcOffsetBytes, uint64_t dataSizeBytes)
    {
        auto* d = static_cast<Buffer*>(dest);
        auto* s = static_cast<Buffer*>(src);
        if (!d || !s || !d->buffer || !s->buffer || dataSizeBytes == 0)
            return;
        // volatile buffers should always be handled with cmdList.writeBuffer, not copyBuffer
        if (d->desc.isVolatile || s->desc.isVolatile)
            return;

        if (srcOffsetBytes > s->desc.byteSize || dataSizeBytes > s->desc.byteSize - srcOffsetBytes)
            return;

        if (destOffsetBytes > d->desc.byteSize || dataSizeBytes > d->desc.byteSize - destOffsetBytes)
            return;

        endEncoding();
        id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
        [blit copyFromBuffer:s->buffer
                sourceOffset:NSUInteger(srcOffsetBytes)
                    toBuffer:d->buffer
           destinationOffset:NSUInteger(destOffsetBytes)
                        size:NSUInteger(dataSizeBytes)];
        [blit endEncoding];
    }

    id<MTLRenderCommandEncoder> CommandList::getOrCreateRenderEncoder()
    {
        if (m_RenderEncoder)
            return m_RenderEncoder;
        endEncoding();

        auto* framebuffer = static_cast<Framebuffer*>(m_CurrentGraphicsState.framebuffer);
        if (!framebuffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] render encoder requested without framebuffer");
            return nil;
        }

        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        for (NSUInteger index = 0; index < framebuffer->desc.colorAttachments.size(); ++index)
        {
            const auto& attachment = framebuffer->desc.colorAttachments[index];
            auto* texture = static_cast<Texture*>(framebuffer->desc.colorAttachments[index].texture);
            rp.colorAttachments[index].texture = texture ? texture->texture : nil;
            rp.colorAttachments[index].level = attachment.subresources.baseMipLevel;
            rp.colorAttachments[index].slice = attachment.subresources.baseArraySlice;
            rp.colorAttachments[index].loadAction = MTLLoadActionLoad;
            rp.colorAttachments[index].storeAction = MTLStoreActionStore;
        }
        if (framebuffer->desc.depthAttachment.texture)
        {
            const auto& attachment = framebuffer->desc.depthAttachment;
            auto* depth = static_cast<Texture*>(framebuffer->desc.depthAttachment.texture);
            rp.depthAttachment.texture = depth ? depth->texture : nil;
            rp.depthAttachment.level = attachment.subresources.baseMipLevel;
            rp.depthAttachment.slice = attachment.subresources.baseArraySlice;
            rp.depthAttachment.loadAction = MTLLoadActionLoad;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
        }

        m_RenderEncoder = [trackedCmdBuffer renderCommandEncoderWithDescriptor:rp];
        if (!m_RenderEncoder)
            m_Context.error("[metal3-trace] failed to create render command encoder");
        return m_RenderEncoder;
    }
    
    void CommandList::draw(const DrawArguments& args)
    {
        if (!m_CurrentGraphicsStateValid)
            return;
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        if (!encoder || !pipeline)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] draw skipped: encoder/pipeline missing");
            return;
        }
        IRRuntimeDrawPrimitives(encoder, pipeline->primitiveType, args.startVertexLocation, args.vertexCount, args.instanceCount, args.startInstanceLocation);
    }

    void CommandList::drawIndexed(const DrawArguments& args)
    {
        if (!m_CurrentGraphicsStateValid)
            return;
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indexBuffer = static_cast<Buffer*>(m_CurrentGraphicsState.indexBuffer.buffer);
        if (!encoder || !pipeline || !indexBuffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexed skipped: encoder/pipeline/index missing");
            return;
        }
        IRRuntimeDrawIndexedPrimitives(encoder, pipeline->primitiveType, args.vertexCount, convertIndexFormat(m_CurrentGraphicsState.indexBuffer.format), indexBuffer->buffer, m_CurrentGraphicsState.indexBuffer.offset + args.startIndexLocation * (m_CurrentGraphicsState.indexBuffer.format == Format::R32_UINT ? 4 : 2), args.instanceCount, args.startVertexLocation, args.startInstanceLocation);
    }
}
