#include "metal3-backend.h"
#include "nvrhi/common/misc.h"
#include "nvrhi/common/resource.h"
#include "nvrhi/utils.h"
#include <Metal/Metal.h>
#include <cstdio>

namespace nvrhi::metal3
{
    void MTL3Context::error(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Error, message.c_str());
    }
    void MTL3Context::warning(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Warning, message.c_str());
    }

    void MTL3Context::info(const std::string& message) const
    {
        if (messageCallback)
            messageCallback->message(MessageSeverity::Info, message.c_str());
    }

    // device creation
    DeviceHandle createDevice(const DeviceDesc& desc)
    {
        Device* device = new Device(desc);
        return DeviceHandle::Create(device);
    }
    Device::Device(const DeviceDesc& desc)
        : m_AftermathEnabled(false)
    {
        m_Context.device = desc.pDevice;
        m_Context.logBufferLifetime = desc.logBufferLifetime;
        m_Context.messageCallback = desc.errorCB;

        if([m_Context.device supportsFamily:MTLGPUFamilyMetal3] == NO)
        {
            m_Context.error("[nvrhi] Metal 3 unsupported!");
        }
        else m_Context.info("[nvrhi] Metal 3 supported");

        // queues, resoureces reserve, allocation, etc...
        m_Context.commonQueue = desc.commonQueue;
    }

    Device::~Device() = default;

    Object Device::getNativeObject(ObjectType objectType)
    {
        switch (objectType)
        {
        case ObjectTypes::MTL3_Device:
            return Object((__bridge void*)m_Context.device);
        case ObjectTypes::Nvrhi_Metal3_Device:
            return Object(this);
        default:
            return nullptr;
        }
    }

    GraphicsAPI Device::getGraphicsAPI()
    {
        return GraphicsAPI::METAL3;
    }
    // placeholders
    Object Device::getNativeQueue(ObjectType objectType, CommandQueue queue)
    {
        (void)queue;
        if (objectType == ObjectTypes::MTL3_CommandQueue)
            return Object((__bridge void*)m_Context.commonQueue);
        return nullptr;
    }

    bool Device::waitForIdle()
    {
        id<MTLCommandBuffer> commandBuffer = [m_Context.commonQueue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        return true;
    }

    void Device::runGarbageCollection()
    {
    }

    HeapHandle Device::createHeap(const HeapDesc& d)
    {
        (void)d;
        m_Context.warning("[nvrhi] Metal3 heaps are not implemented; using placed resources is unsupported.");
        return nullptr;
    }

    MemoryRequirements Device::getTextureMemoryRequirements(ITexture* texture)
    {
        MemoryRequirements result{};
        if (texture)
        {
            Texture* t = checked_cast<Texture*>(texture);
            result.size = t->memSize;
            result.alignment = t->memAlign;
        }
        return result;
    }

    bool Device::bindTextureMemory(ITexture* texture, IHeap* heap, uint64_t offset)
    {
        (void)texture; (void)heap; (void)offset;
        return false;
    }

    StagingTextureHandle Device::createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess)
    {
        // TODO: stub
    }

    void* Device::mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t* outRowPitch)
    {
        // TODO: stub
    }

    void Device::unmapStagingTexture(IStagingTexture* tex)
    {
        // TODO: stub
    }

    void Device::getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings)
    {
        (void)texture;
        (void)numTiles;
        (void)desc;
        (void)tileShape;
        (void)subresourceTilingsNum;
        (void)subresourceTilings;

        utils::NotSupported();
    }

    void Device::updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue)
    {
        (void)texture; (void)tileMappings; (void)numTileMappings; (void)executionQueue;
        utils::NotSupported();
    }

    SamplerFeedbackTextureHandle Device::createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc)
    {
        (void)pairedTexture;
        (void)desc;

        utils::NotSupported();
        return nullptr;
    }

    SamplerFeedbackTextureHandle Device::createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture)
    {
        (void)objectType;
        (void)texture;
        (void)pairedTexture;

        utils::NotSupported();
        return nullptr;
    }

    void* Device::mapBuffer(IBuffer* b, CpuAccessMode mapFlags)
    {
        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return nullptr;

        if (mapFlags == CpuAccessMode::None)
        {
            utils::InvalidEnum();
            return nullptr;
        }

        if (buffer->desc.cpuAccess == CpuAccessMode::None)
        {
            m_Context.error("[nvrhi] Cannot map Metal buffer without CPU access.");
            return nullptr;
        }

        if (buffer->desc.cpuAccess != mapFlags)
        {
            m_Context.error("[nvrhi] Metal buffer mapped with incompatible CPU access mode.");
            return nullptr;
        }

        if (buffer->buffer.storageMode == MTLStorageModePrivate)
        {
            m_Context.error("[nvrhi] Cannot map private Metal buffer.");
            return nullptr;
        }

        return [buffer->buffer contents];
    }

    void Device::unmapBuffer(IBuffer* b)
    {
        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer || !buffer->buffer)
            return;

        if (buffer->buffer.storageMode == MTLStorageModeManaged &&
            buffer->desc.cpuAccess == CpuAccessMode::Write)
        {
            [buffer->buffer didModifyRange:NSMakeRange(0, NSUInteger(buffer->desc.byteSize))];
        }
    }

    MemoryRequirements Device::getBufferMemoryRequirements(IBuffer* b)
    {
        MemoryRequirements result{};

        Buffer* buffer = static_cast<Buffer*>(b);
        if (!buffer)
            return result;

        const BufferDesc& desc = buffer->desc;
        if (desc.byteSize == 0)
            return result;

        MTLResourceOptions options = convertCpuAccess(desc.cpuAccess);
        MTLSizeAndAlign sizeAndAlign =
            [m_Context.device heapBufferSizeAndAlignWithLength:NSUInteger(desc.byteSize)
                                                    options:options];

        result.size = sizeAndAlign.size;
        result.alignment = sizeAndAlign.align;
        return result;
    }

    bool Device::bindBufferMemory(IBuffer *buffer, IHeap *heap, uint64_t offset)
    {
        (void)buffer;
        (void)heap;
        (void)offset;
        utils::NotSupported();
        return false;
    }

    BufferHandle Device::createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc &desc)
    {
        if (!buffer.pointer)
            return nullptr;

        if (objectType != ObjectTypes::MTL3_Buffer)
            return nullptr;

        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer.pointer;
        if (!mtlBuffer)
            return nullptr;

        if (desc.byteSize > 0 && desc.byteSize > [mtlBuffer length])
        {
            m_Context.error("[nvrhi] Native Metal buffer is smaller than BufferDesc::byteSize.");
            return nullptr;
        }
        Buffer *result	   = new Buffer();
        result->desc	   = desc;
        result->buffer	   = (__bridge id<MTLBuffer>)buffer.pointer;
        result->ownsBuffer = false;
        return BufferHandle::Create(result);
    }

    ShaderHandle Device::createShaderSpecialization(IShader *baseShader, const ShaderSpecialization *constants,
												uint32_t numConstants)
    {
        (void)constants;
        (void)numConstants;
        utils::NotSupported();
        return baseShader;
    }
    
}
