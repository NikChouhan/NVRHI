#pragma once

#include "nvrhi/common/aftermath.h"
#include "nvrhi/common/resource.h"
#include "nvrhi/nvrhi.h"
#include <nvrhi/metal3.h>
#include <nvrhi/utils.h>
#include <dispatch/dispatch.h>
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace nvrhi::metal3 
{
    class Texture;
    class Buffer;
    class Shader;
    class Sampler;
    class InputLayout;
    class Framebuffer;
    class GraphicsPipeline;
    class ComputePipeline;
    class BindingLayout;
    class BindingSet;
    class MeshletPipeline;
    class RayTracingPipeline;
    class EventQuery;
    class TimerQuery;
    // dummy classes for non essential or planned for future features
    // Not planned
    class DummyOpacityMicromap;
    // TODO: for future when RT backend will be added
    class DummyAccelStruct;

    // implementation in metal3-constants.cpp
    MTLTextureType convertTextureDimension(TextureDimension dimension, uint32_t sampleCount);
    MTLResourceOptions convertCpuAccess(CpuAccessMode cpuAccess);
    MTLVertexFormat convertVertexFormat(Format format);
    MTLIndexType convertIndexFormat(Format format);
    MTLPrimitiveType convertPrimitiveType(PrimitiveType primitiveType);
    MTLCullMode convertCullMode(RasterCullMode cullMode);
    MTLWinding convertWinding(bool frontCounterClockwise);
    MTLCompareFunction convertCompareFunction(ComparisonFunc func);
    MTLSamplerAddressMode convertSamplerAddressMode(SamplerAddressMode mode);
    MTLBlendFactor convertBlendFactor(BlendFactor factor);
    MTLBlendOperation convertBlendOp(BlendOp op);

    // metal 3 context
    struct MTL3Context
    {
        id<MTLDevice> device;
        id<MTLCommandQueue> commonQueue;

        bool logBufferLifetime = false;
        IMessageCallback* messageCallback = nullptr;
        void error(const std::string& message) const;
        void warning(const std::string& message) const;
        void info(const std::string& message) const;
    };

    // TODO: stub for now
    class UploadManager
    {
    public:
        UploadManager(const MTL3Context& context, size_t uploadChunkSize, size_t scratchMaxMem, bool isScratchBuffer);
    };

    class Texture : public RefCounter<ITexture>
    {
    public:
        TextureDesc desc;
        id<MTLTexture> texture = nil;
        NSUInteger memSize;
        NSUInteger memAlign;
        bool ownsTexture = true;

        const TextureDesc& getDesc() const override { return desc; }
        Object getNativeObject(ObjectType objectType) override;
        Object getNativeView(ObjectType objectType, Format format = Format::UNKNOWN, TextureSubresourceSet subresources = AllSubresources, TextureDimension dimension = TextureDimension::Unknown, bool isReadOnlyDSV = false) override;
    };

    //TODO: stub
    class StagingTexture : public RefCounter<IStagingTexture>
    {
    public:
        TextureDesc desc;
        const TextureDesc& getDesc() const override { return desc; }
    };

    class Buffer : public RefCounter<IBuffer>
    {
    public:
        BufferDesc desc;
        id<MTLBuffer> buffer = nil;
        bool ownsBuffer = true;

        const BufferDesc& getDesc() const override { return desc; }
        GpuVirtualAddress getGpuVirtualAddress() const override { return buffer ? buffer.gpuAddress : 0; }
        Object getNativeObject(ObjectType objectType) override;
    };

    // metal3 device
    /* 
     * device functions declarations of virtual nvrhi functions declared in nvrhi.h
     * 
    */

    class Device : public RefCounter<nvrhi::metal3::IDevice>
    {
    public:
        explicit Device(const DeviceDesc& desc);
        ~Device() override;
        
        // IResource impl
        Object getNativeObject(ObjectType objectType) override;

        // IDevice impl

        // Metal3: MTLHeap or resource heaps (closest to dx12/vulkan type descriptor heaps) not implemented yet, does nothing
        HeapHandle createHeap(const HeapDesc& d) override;

        TextureHandle createTexture(const TextureDesc& d) override;

        // Metal3: useful for texture allocated inside heaps, for manual/placed allocation; we create committed textures directly currently
        MemoryRequirements getTextureMemoryRequirements(ITexture* texture) override;
        // Metal 3: nothing like binding a texture memory in metal, does nothing, a stub
        bool bindTextureMemory(ITexture* texture, IHeap* heap, uint64_t offset) override;

        TextureHandle createHandleForNativeTexture(ObjectType objectType, Object texture, const TextureDesc& desc) override;

        StagingTextureHandle createStagingTexture(const TextureDesc& d, CpuAccessMode cpuAccess) override;
        void *mapStagingTexture(IStagingTexture* tex, const TextureSlice& slice, CpuAccessMode cpuAccess, size_t *outRowPitch) override;
        void unmapStagingTexture(IStagingTexture* tex) override;

        void getTextureTiling(ITexture* texture, uint32_t* numTiles, PackedMipDesc* desc, TileShape* tileShape, uint32_t* subresourceTilingsNum, SubresourceTiling* subresourceTilings) override;
        void updateTextureTileMappings(ITexture* texture, const TextureTilesMapping* tileMappings, uint32_t numTileMappings, CommandQueue executionQueue = CommandQueue::Graphics) override;

        SamplerFeedbackTextureHandle createSamplerFeedbackTexture(ITexture* pairedTexture, const SamplerFeedbackTextureDesc& desc) override;
        SamplerFeedbackTextureHandle createSamplerFeedbackForNativeTexture(ObjectType objectType, Object texture, ITexture* pairedTexture) override;

        BufferHandle createBuffer(const BufferDesc& d) override;
        void *mapBuffer(IBuffer* b, CpuAccessMode mapFlags) override;
        void unmapBuffer(IBuffer* b) override;

        // Metal3: useful for buffers allocated inside heaps, for manual/placed allocation; we create committed buffers directly currently
        MemoryRequirements getBufferMemoryRequirements(IBuffer* buffer) override;
        bool bindBufferMemory(IBuffer* buffer, IHeap* heap, uint64_t offset) override;

        BufferHandle createHandleForNativeBuffer(ObjectType objectType, Object buffer, const BufferDesc& desc) override;

        ShaderHandle createShader(const ShaderDesc& d, const void* binary, size_t binarySize) override;
        ShaderHandle createShaderSpecialization(IShader* baseShader, const ShaderSpecialization* constants, uint32_t numConstants) override;
        ShaderLibraryHandle createShaderLibrary(const void* binary, size_t binarySize) override;

        SamplerHandle createSampler(const SamplerDesc& d) override;

        InputLayoutHandle createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader) override;

        EventQueryHandle createEventQuery() override;
        void setEventQuery(IEventQuery* query, CommandQueue queue) override;
        bool pollEventQuery(IEventQuery* query) override;
        void waitEventQuery(IEventQuery* query) override;
        void resetEventQuery(IEventQuery* query) override;

        TimerQueryHandle createTimerQuery() override;
        bool pollTimerQuery(ITimerQuery* query) override;
        float getTimerQueryTime(ITimerQuery* query) override;
        void resetTimerQuery(ITimerQuery* query) override;

        GraphicsAPI getGraphicsAPI() override;

        FramebufferHandle createFramebuffer(const FramebufferDesc& desc) override;
        
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo) override;
        
        GraphicsPipelineHandle createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb) override;
        
        ComputePipelineHandle createComputePipeline(const ComputePipelineDesc& desc) override;

        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, FramebufferInfo const& fbinfo) override;

        MeshletPipelineHandle createMeshletPipeline(const MeshletPipelineDesc& desc, IFramebuffer* fb) override;

        // Unsupported: RT backend is WIP
        rt::PipelineHandle createRayTracingPipeline(const rt::PipelineDesc& desc) override;

        BindingLayoutHandle createBindingLayout(const BindingLayoutDesc& desc) override;
        BindingLayoutHandle createBindlessLayout(const BindlessLayoutDesc& desc) override;

        BindingSetHandle createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout) override;
        DescriptorTableHandle createDescriptorTable(IBindingLayout* layout) override;

        void resizeDescriptorTable(IDescriptorTable* descriptorTable, uint32_t newSize, bool keepContents = true) override;
        bool writeDescriptorTable(IDescriptorTable* descriptorTable, const BindingSetItem& item) override;

        rt::OpacityMicromapHandle createOpacityMicromap(const rt::OpacityMicromapDesc& desc) override;
        rt::AccelStructHandle createAccelStruct(const rt::AccelStructDesc& desc) override;
        MemoryRequirements getAccelStructMemoryRequirements(rt::IAccelStruct* as) override;
        rt::cluster::OperationSizeInfo getClusterOperationSizeInfo(const rt::cluster::OperationParams& params) override;

        bool bindAccelStructMemory(rt::IAccelStruct* as, IHeap* heap, uint64_t offset) override;

        nvrhi::CommandListHandle createCommandList(const CommandListParameters& params = CommandListParameters()) override;
        uint64_t executeCommandLists(nvrhi::ICommandList* const* pCommandLists, size_t numCommandLists, CommandQueue executionQueue = CommandQueue::Graphics) override;
        void queueWaitForCommandList(CommandQueue waitQueue, CommandQueue executionQueue, uint64_t instance) override;
        bool waitForIdle() override;

        CommandListLifetimeTrackerHandle createCommandListLifetimeTracker(CommandQueue executionQueue) override;

        // Metal3: does nothing
        void runGarbageCollection() override;
        bool queryFeatureSupport(Feature feature, void* pInfo = nullptr, size_t infoSize = 0) override;
        FormatSupport queryFormatSupport(Format format) override;
        coopvec::DeviceFeatures queryCoopVecFeatures() override;
        coopvec::MatMulFormatSupport queryCoopVecMatMulFormatSupport(const coopvec::MatMulFormatCombo& combination) override;
        coopvec::TrainingFormatSupport queryCoopVecTrainingFormatSupport(coopvec::DataType componentType) override;
        size_t getCoopVecMatrixSize(coopvec::DataType type, coopvec::MatrixLayout layout, int rows, int columns) override;
        Object getNativeQueue(ObjectType objectType, CommandQueue queue) override;
        IMessageCallback* getMessageCallback() override { return m_Context.messageCallback; }

        // unused funcs, no aftermath sdk here :/
        bool isAftermathEnabled() override { return m_AftermathEnabled; }
        AftermathCrashDumpHelper& getAftermathCrashDumpHelper() override { return m_AftermathCrashDumpHelper; }

        // TODO: metal.h virtual funs implementation if any
        
    private:
        bool m_AftermathEnabled;
        AftermathCrashDumpHelper m_AftermathCrashDumpHelper;

        MTL3Context m_Context;

        std::mutex m_Mutex;
    };
}
