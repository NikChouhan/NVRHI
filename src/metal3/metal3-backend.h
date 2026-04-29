#pragma once

#include "nvrhi/common/aftermath.h"
#include "nvrhi/common/resource.h"
#include "nvrhi/nvrhi.h"
#include <nvrhi/metal3.h>

// okay so the plan is to first write the nvrhi API code and then write the subsequent
// backend to do it:
// -> Triangle (test bindings, psos, vertex/index buffers, command buffers and queue)
// -> Texture (test textures)
// -> a simple compute shader (post process maybe, test compute queues, psos)
// -> simple pass for a triangle but with Draw Indirect, yeah thats it for now

namespace nvrhi::metal3 
{
    struct MTL3Context
    {
        id<MTLDevice> device;

        bool logBufferLifetime = false;
        IMessageCallback* messageCallback = nullptr;
        void error(const std::string& message) const;
        void warning(const std::string& message) const;
        void info(const std::string& message) const;
    };

    // metal3 device
    class Device : public RefCounter<nvrhi::metal3::IDevice>
    {
    public:
        explicit Device(const DeviceDesc& desc);
        ~Device() override;
        
        // IResource impl
        Object getNativeObject(ObjectType objectType) override;

        // IDevice impl

        HeapHandle createHeap(const HeapDesc& d) override;

        TextureHandle createTexture(const TextureDesc& d) override;
        MemoryRequirements getTextureMemoryRequirements(ITexture* texture) override;
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
        void runGarbageCollection() override;
        bool queryFeatureSupport(Feature feature, void* pInfo = nullptr, size_t infoSize = 0) override;
        FormatSupport queryFormatSupport(Format format) override;
        coopvec::DeviceFeatures queryCoopVecFeatures() override;
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