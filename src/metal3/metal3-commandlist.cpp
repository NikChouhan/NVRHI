#include "metal3-backend.h"
#include "nvrhi/nvrhi.h"
#include <Metal/Metal.h>
#include <algorithm>
#include <cstdlib>
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
}