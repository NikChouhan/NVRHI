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
    // taken from <metal_irconverter_runtime/metal_irconverter_runtime.h>
    // predefined bind point values for shaders converted with metal shader converter
    static constexpr uint32_t c_MscVertexBufferBindPoint = 6;
    static constexpr uint32_t c_IrDrawArgumentsBindPoint = 4;

    struct IcbIndexedIndirectParams
    {
        uint32_t maxDrawCount = 0;
        uint32_t paramOffsetBytes = 0;
        uint32_t countOffsetBytes = 0;
        uint32_t indexType = 0;
        uint32_t indexBufferOffset = 0;
        uint32_t primitiveType = 0;
        uint32_t drawArgumentsBindPoint = c_IrDrawArgumentsBindPoint;
        uint32_t pad = 0;
    };

    /*
    // helper used by drawIndexedIndirectCount. Metal cannot
    // execute a counted indexed-indirect draw directly, so a compute shader
    // converts the NVRHI args/count buffers into an MTLIndirectCommandBuffer.
    // The pipeline runs that compute shader; icbEncoder encodes the ICB object
    // into the helper shader's argument buffer.
    */
    struct IndexedIndirectIcbFillState
    {
        id<MTLComputePipelineState> pipeline = nil;
        id<MTLArgumentEncoder> icbEncoder = nil;
    };

    // Compile and cache the ICB-fill compute pipeline once. The returned state
    // stays empty if compilation fails, and callers simply skip counted indirect
    // draws rather than trying to fall back to a CPU readback of the count.
    static const IndexedIndirectIcbFillState& getIndexedIndirectIcbFillState(const MTL3Context& context)
    {
        static std::mutex mutex;
        static IndexedIndirectIcbFillState state;
        static bool attempted = false;

        std::lock_guard<std::mutex> lock(mutex);
        if (attempted)
            return state;
        attempted = true;

        static constexpr const char* source = R"(
            #include <metal_stdlib>
            using namespace metal;

            struct DrawIndexedIndirectArguments
            {
                uint indexCount;
                uint instanceCount;
                uint startIndexLocation;
                int baseVertexLocation;
                uint startInstanceLocation;
            };

            struct IcbIndexedIndirectParams
            {
                uint maxDrawCount;
                uint paramOffsetBytes;
                uint countOffsetBytes;
                uint indexType;
                uint indexBufferOffset;
                uint primitiveType;
                uint drawArgumentsBindPoint;
                uint pad;
            };

            struct IcbArgumentBuffer
            {
                command_buffer icb [[id(0)]];
            };

            static primitive_type to_primitive_type(uint value)
            {
                switch (value)
                {
                case 0: return primitive_type::point;
                case 1: return primitive_type::line;
                case 2: return primitive_type::line_strip;
                case 4: return primitive_type::triangle_strip;
                default: return primitive_type::triangle;
                }
            }

            // buffers encoded into compute encoder at 0...5 pos
            // indirectParams -> indirect paramaters buffer (DrawArguments struct in nvrhi.h)
            // rawCount -> indirect draw Count
            // executionRange -> execute ICB commands from index 'n' for count 'm'
            // indexBuffer - same index buffer, that is cached from setGraphicsState fn for the drawIndexedIndirectCount cmd 
            // icbArgumentBuffer -> the MTLIndirectCommandBuffer is passed to the compute shader, to write a draw_indexed_primitives
            // --- into render_command. this is processed in the end, with :
            // --- [renderEncoder executeCommandsInBuffer:icb indirectBuffer:executionRange indirectBufferOffset:0];
            // --- this, as the name implies, executes the commands in the ICB.

            // all in all this, compute shader basically writes the drawIndexed commands into an indirect command buffer,
            // (as UAV in d3d12 terms), and then the render encoder, executes the buffer commands
            kernel void nvrhi_metal3_fill_indexed_indirect_icb(
                device const uchar* indirectParams [[buffer(0)]],
                device const uchar* rawCount [[buffer(1)]],
                device uint2* executionRange [[buffer(2)]],
                constant IcbArgumentBuffer& icbArgumentBuffer [[buffer(3)]],
                device const uchar* indexBuffer [[buffer(4)]],
                constant IcbIndexedIndirectParams& params [[buffer(5)]],
                    uint tid [[thread_position_in_grid]])
            {
                device const uint* countPtr = reinterpret_cast<device const uint*>(rawCount + params.countOffsetBytes);
                const uint gpuCount = min(*countPtr, params.maxDrawCount);

                // Metal executes an ICB over an explicit range. write that range from
                // the GPU count buffer so the render pass never needs a CPU readback.
                if (tid == 0)
                    executionRange[0] = uint2(0, gpuCount);

                if (tid >= gpuCount)
                    return;

                device const DrawIndexedIndirectArguments* args =
                    reinterpret_cast<device const DrawIndexedIndirectArguments*>(indirectParams + params.paramOffsetBytes);
                const DrawIndexedIndirectArguments a = args[tid];

                if (a.indexCount == 0 || a.instanceCount == 0)
                    return;

                // One compute thread writes one render command into the same slot as
                // its source indirect argument.
                render_command command(icbArgumentBuffer.icb, tid);

                // Metal Shader Converter reads D3D draw parameters from this runtime
                // bind point. Point every ICB command at its own source indirect args.
                device const uchar* drawArgs =
                    indirectParams + params.paramOffsetBytes + tid * uint(sizeof(DrawIndexedIndirectArguments));
                command.set_vertex_buffer(drawArgs, params.drawArgumentsBindPoint);

                if (params.indexType == 0)
                {
                    device const ushort* indices =
                        reinterpret_cast<device const ushort*>(indexBuffer + params.indexBufferOffset) + a.startIndexLocation;
                    command.draw_indexed_primitives(to_primitive_type(params.primitiveType),
                        a.indexCount,
                        indices,
                        a.instanceCount,
                        uint(a.baseVertexLocation),
                        a.startInstanceLocation);
                }
                else
                {
                    device const uint* indices =
                        reinterpret_cast<device const uint*>(indexBuffer + params.indexBufferOffset) + a.startIndexLocation;
                    command.draw_indexed_primitives(to_primitive_type(params.primitiveType),
                        a.indexCount,
                        indices,
                        a.instanceCount,
                        uint(a.baseVertexLocation),
                        a.startInstanceLocation);
                }
            }
        )";

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 13.0, *))
            options.languageVersion = MTLLanguageVersion3_0;

        id<MTLLibrary> library = [context.device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                               options:options
                                                                 error:&error];
        if (!library)
        {
            std::string message = "[metal3] failed to compile indexed-indirect ICB helper";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            context.error(message);
            return state;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"nvrhi_metal3_fill_indexed_indirect_icb"];
        if (!function)
        {
            context.error("[metal3] indexed-indirect ICB helper function missing");
            return state;
        }

        state.icbEncoder = [function newArgumentEncoderWithBufferIndex:3];
        if (!state.icbEncoder)
        {
            context.error("[metal3] failed to create indexed-indirect ICB argument encoder");
            return state;
        }

        if (@available(macOS 11.0, *))
        {
            MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
            descriptor.computeFunction = function;
            // Required because this compute pipeline writes render commands into
            // an MTLIndirectCommandBuffer.
            descriptor.supportIndirectCommandBuffers = YES;
            state.pipeline = [context.device newComputePipelineStateWithDescriptor:descriptor
                                                                           options:MTLPipelineOptionNone
                                                                        reflection:nil
                                                                             error:&error];
        }
        else
        {
            state.pipeline = [context.device newComputePipelineStateWithFunction:function error:&error];
        }

        if (!state.pipeline)
        {
            std::string message = "[metal3] failed to create indexed-indirect ICB helper pipeline";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            context.error(message); 
        }
        
        return state;
    }

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

    static const char* resourceTypeName(ResourceType type)
    {
        switch (type)
        {
        case ResourceType::Texture_SRV: return "Texture_SRV";
        case ResourceType::Texture_UAV: return "Texture_UAV";
        case ResourceType::TypedBuffer_SRV: return "TypedBuffer_SRV";
        case ResourceType::TypedBuffer_UAV: return "TypedBuffer_UAV";
        case ResourceType::StructuredBuffer_SRV: return "StructuredBuffer_SRV";
        case ResourceType::StructuredBuffer_UAV: return "StructuredBuffer_UAV";
        case ResourceType::RawBuffer_SRV: return "RawBuffer_SRV";
        case ResourceType::RawBuffer_UAV: return "RawBuffer_UAV";
        case ResourceType::ConstantBuffer: return "ConstantBuffer";
        case ResourceType::VolatileConstantBuffer: return "VolatileConstantBuffer";
        case ResourceType::Sampler: return "Sampler";
        case ResourceType::None: return "None";
        default: return "Other";
        }
    }

    static MetalArgumentTableCacheKey makeArgumentTableCacheKey(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets)
    {
        MetalArgumentTableCacheKey key;
        key.plan = &plan;
        key.bindingSets.reserve(bindingSets.size());
        key.bindingSetVersions.reserve(bindingSets.size());

        for (IBindingSet* bindingSet : bindingSets)
        {
            auto* set = static_cast<BindingSet*>(bindingSet);
            key.bindingSets.push_back(set);
            key.bindingSetVersions.push_back(set ? set->version : 0);
        }

        return key;
    }

    static bool usesDirectVolatileConstantBufferBinding(const MetalBindingPlanEntry& planEntry)
    {
        return planEntry.layoutMatched &&
            planEntry.argumentType == MscArgumentType::CBV &&
            planEntry.layoutType == ResourceType::VolatileConstantBuffer;
    }

    static bool isSrvType(ResourceType type)
    {
        return type == ResourceType::Texture_SRV ||
            type == ResourceType::TypedBuffer_SRV ||
            type == ResourceType::StructuredBuffer_SRV ||
            type == ResourceType::RawBuffer_SRV;
    }

    static bool isUavType(ResourceType type)
    {
        return type == ResourceType::Texture_UAV ||
            type == ResourceType::TypedBuffer_UAV ||
            type == ResourceType::StructuredBuffer_UAV ||
            type == ResourceType::RawBuffer_UAV;
    }

    static bool isBufferType(ResourceType type)
    {
        return type == ResourceType::ConstantBuffer ||
            type == ResourceType::VolatileConstantBuffer ||
            type == ResourceType::TypedBuffer_SRV ||
            type == ResourceType::TypedBuffer_UAV ||
            type == ResourceType::StructuredBuffer_SRV ||
            type == ResourceType::StructuredBuffer_UAV ||
            type == ResourceType::RawBuffer_SRV ||
            type == ResourceType::RawBuffer_UAV;
    }

    static bool matchesMscArgumentType(ResourceType resourceType, MscArgumentType argumentType)
    {
        switch (argumentType)
        {
        case MscArgumentType::SRV:
            return isSrvType(resourceType);
        case MscArgumentType::UAV:
            return isUavType(resourceType);
        case MscArgumentType::CBV:
            return resourceType == ResourceType::ConstantBuffer || resourceType == ResourceType::VolatileConstantBuffer;
        case MscArgumentType::Sampler:
            return resourceType == ResourceType::Sampler;
        }

        return false;
    }

    // Metal binding model:
    // - regular CB/SRV/UAV/sampler bindings are encoded into Metal Shader
    //   Converter descriptor tables using the reflected per-stage plan.
    // - volatile constant buffers are command-list dynamic; writeBuffer()
    //   suballocates upload memory, and the current allocation is bound
    //   directly to the translated b# slot. They are intentionally excluded
    //   from cached argument tables so writes do not have to invalidate them.
    // - vertex, index, indirect, helper, and argument-table buffers remain
    //   direct Metal bindings because they are not ordinary shader resources.
    static const MetalBindingResource* findArgumentTableResource(
        const BindingSetVector& bindingSets,
        const MetalBindingPlanEntry& planEntry)
    {
        if (usesDirectVolatileConstantBufferBinding(planEntry))
            return nullptr;

        if (!planEntry.layoutMatched || planEntry.layoutIndex >= bindingSets.size())
            return nullptr;

        auto* set = static_cast<BindingSet*>(bindingSets[planEntry.layoutIndex]);
        if (!set)
            return nullptr;

        for (const MetalBindingResource& entry : set->entries)
        {
            if (entry.type == ResourceType::None ||
                entry.type == ResourceType::VolatileConstantBuffer ||
                entry.type == ResourceType::SamplerFeedbackTexture_UAV ||
                !matchesMscArgumentType(entry.type, planEntry.argumentType))
                continue;

            if (entry.slot + entry.arrayElement == planEntry.slot)
                return &entry;
        }

        return nullptr;
    }

    static void encodeArgumentTableEntry(IRDescriptorTableEntry* entry, const MetalBindingResource& resource)
    {
        switch (resource.type)
        {
        case ResourceType::Texture_SRV:
        case ResourceType::Texture_UAV:
            if (resource.texture)
                IRDescriptorTableSetTexture(entry, resource.texture, 0.f, 0);
            break;
        case ResourceType::Sampler:
            if (resource.sampler)
                IRDescriptorTableSetSampler(entry, resource.sampler, resource.samplerMipBias);
            break;
        case ResourceType::ConstantBuffer:
        case ResourceType::TypedBuffer_SRV:
        case ResourceType::TypedBuffer_UAV:
        case ResourceType::StructuredBuffer_SRV:
        case ResourceType::StructuredBuffer_UAV:
        case ResourceType::RawBuffer_SRV:
        case ResourceType::RawBuffer_UAV:
        {
            if (!resource.buffer)
                break;

            IRBufferView view{};
            view.buffer = resource.buffer;
            view.bufferOffset = resource.bufferOffset;
            view.bufferSize = resource.bufferSize;
            view.textureBufferView = nil;
            view.textureViewOffsetInElements = 0;
            view.typedBuffer = false;
            IRDescriptorTableSetBufferView(entry, &view);
            break;
        }
        default:
            break;
        }
    }

    static void useArgumentTableResource(id<MTLComputeCommandEncoder> encoder, const MetalBindingResource& resource)
    {
        switch (resource.type)
        {
        case ResourceType::Texture_SRV:
        case ResourceType::Texture_UAV:
            if (resource.texture)
                [encoder useResource:resource.texture usage:resource.usage];
            break;
        default:
            if (isBufferType(resource.type) && resource.buffer)
                [encoder useResource:resource.buffer usage:resource.usage];
            break;
        }
    }

    static void useArgumentTableResource(id<MTLRenderCommandEncoder> encoder, const MetalBindingResource& resource, MTLRenderStages stages)
    {
        switch (resource.type)
        {
        case ResourceType::Texture_SRV:
        case ResourceType::Texture_UAV:
            if (resource.texture)
                [encoder useResource:resource.texture usage:resource.usage stages:stages];
            break;
        default:
            if (isBufferType(resource.type) && resource.buffer)
                [encoder useResource:resource.buffer usage:resource.usage stages:stages];
            break;
        }
    }

    // similar to useArgumentTableResources (with render command encoder), but with compute command encoder 
    static void useArgumentTableResources(id<MTLComputeCommandEncoder> encoder, const BindingSetVector& bindingSets, const MetalStageBindingPlan& plan)
    {
        for (const MetalBindingPlanEntry& planEntry : plan.entries)
        {
            const MetalBindingResource* resource = findArgumentTableResource(bindingSets, planEntry);
            if (resource)
                useArgumentTableResource(encoder, *resource);
        }
    }

    // After the descriptor table buffer is bound, tell Metal which native
    // resources the table may reference for this render stage. Encoding the
    // IRDescriptorTableEntry values makes the resources visible to the
    // translated shader; useResource gives Metal explicit usage/stage
    // information for hazard tracking, especially for resources reached
    // indirectly through the table.
    static void useArgumentTableResources(id<MTLRenderCommandEncoder> encoder, const BindingSetVector& bindingSets, const MetalStageBindingPlan& plan, MTLRenderStages stages)
    {
        for (const MetalBindingPlanEntry& planEntry : plan.entries)
        {
            const MetalBindingResource* resource = findArgumentTableResource(bindingSets, planEntry);
            if (resource)
                useArgumentTableResource(encoder, *resource, stages);
        }
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

    id<MTLCommandBuffer> CommandList::getNativeCommandBuffer()
    {
        return trackedCmdBuffer;
    }

    void CommandList::setTracyGpuScope(const char* name, const char* file, const char* function, uint32_t line, void* context)
    {
        m_TracyGpuScope.name = name;
        m_TracyGpuScope.file = file;
        m_TracyGpuScope.function = function;
        m_TracyGpuScope.line = line;
        m_TracyGpuScope.context = context;
        m_TracyGpuScope.active = name && file && function && context;
    }

    void CommandList::clearTracyGpuScope()
    {
        m_TracyGpuScope = TracyGpuScopeDesc{};
    }

#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
    tracy::SourceLocationData* CommandList::getOrCreateTracySourceLocation()
    {
        if (!m_TracyGpuScope.active)
            return nullptr;

        for (const auto& sourceLocation : m_TracySourceLocations)
        {
            if (sourceLocation->line == m_TracyGpuScope.line &&
                std::strcmp(sourceLocation->name ? sourceLocation->name : "", m_TracyGpuScope.name ? m_TracyGpuScope.name : "") == 0 &&
                std::strcmp(sourceLocation->file ? sourceLocation->file : "", m_TracyGpuScope.file ? m_TracyGpuScope.file : "") == 0 &&
                std::strcmp(sourceLocation->function ? sourceLocation->function : "", m_TracyGpuScope.function ? m_TracyGpuScope.function : "") == 0)
                return sourceLocation.get();
        }

        auto sourceLocation = std::make_unique<tracy::SourceLocationData>();
        sourceLocation->name = m_TracyGpuScope.name;
        sourceLocation->function = m_TracyGpuScope.function;
        sourceLocation->file = m_TracyGpuScope.file;
        sourceLocation->line = m_TracyGpuScope.line;
        sourceLocation->color = 0;

        tracy::SourceLocationData* ptr = sourceLocation.get();
        m_TracySourceLocations.push_back(std::move(sourceLocation));
        return ptr;
    }

    void CommandList::beginTracyRenderEncoderZone(MTLRenderPassDescriptor* desc)
    {
        if (!desc || !m_TracyGpuScope.active || !m_TracyGpuScope.context)
            return;

        tracy::SourceLocationData* sourceLocation = getOrCreateTracySourceLocation();
        if (!sourceLocation)
            return;

        m_TracyEncoderScope.emplace(static_cast<TracyMetalCtx*>(m_TracyGpuScope.context), desc, sourceLocation, true);
    }

    void CommandList::beginTracyComputeEncoderZone(MTLComputePassDescriptor* desc)
    {
        if (!desc || !m_TracyGpuScope.active || !m_TracyGpuScope.context)
            return;

        tracy::SourceLocationData* sourceLocation = getOrCreateTracySourceLocation();
        if (!sourceLocation)
            return;

        m_TracyEncoderScope.emplace(static_cast<TracyMetalCtx*>(m_TracyGpuScope.context), desc, sourceLocation, true);
    }
#endif

    void CommandList::endEncoding()
    {
        if (m_RenderEncoder)
        {
            [m_RenderEncoder endEncoding];
            m_RenderEncoder = nil;
        }
        if (m_ComputeEncoder)
        {
            [m_ComputeEncoder endEncoding];
            m_ComputeEncoder = nil;
        }
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        m_TracyEncoderScope.reset();
#endif
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

    void CommandList::setGraphicsState(const GraphicsState& state)
    {
        m_CurrentGraphicsState = state;
        m_CurrentGraphicsStateValid = true;
        endEncoding();
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(state.pipeline);
        if (!encoder || !pipeline)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] setGraphicsState skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no"));
            return;
        }

        applyGraphicsStateToEncoder(encoder, state);
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
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        beginTracyRenderEncoderZone(rp);
#endif

        m_RenderEncoder = [trackedCmdBuffer renderCommandEncoderWithDescriptor:rp];
        if (!m_RenderEncoder)
            m_Context.error("[metal3-trace] failed to create render command encoder");
        return m_RenderEncoder;
    }
    id<MTLComputeCommandEncoder> CommandList::getOrCreateComputeEncoder()
    {
        if (m_ComputeEncoder)
            return m_ComputeEncoder;
        endEncoding();
        MTLComputePassDescriptor* cp = [MTLComputePassDescriptor computePassDescriptor];
#if defined(NVRHI_METAL3_WITH_TRACY) && defined(TRACY_ENABLE)
        beginTracyComputeEncoderZone(cp);
#endif
        m_ComputeEncoder = [trackedCmdBuffer computeCommandEncoderWithDescriptor:cp];
        if (!m_ComputeEncoder)
            m_Context.error("[metal3-trace] failed to create compute command encoder");
        return m_ComputeEncoder;
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

    void CommandList::drawIndirect(uint32_t offsetBytes, uint32_t drawCount)
    {
        if (!m_CurrentGraphicsStateValid)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndirect skipped: graphics state invalid");
            return;
        }
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        if (!encoder || !pipeline || !indirectParams || !indirectParams->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndirect skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no") +
                    " indirect='" + (indirectParams ? indirectParams->desc.debugName : std::string("<null>")) + "'");
            return;
        }

        // uncomment for debugging
        // static int drawIndirectLogCount = 0;
        // if (traceMetalRuntime() && drawIndirectLogCount++ < 64)
        //     m_Context.info("[metal3-trace] drawIndirect count=" + std::to_string(drawCount) +
        //         " offset=" + std::to_string(offsetBytes) + " args='" + indirectParams->desc.debugName + "'");

        for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex)
        {
            // Metal Shader Converter vertex shaders read draw parameters from
            // the IR runtime bind points. Use the runtime wrapper for indirect
            // draws too; raw Metal draws do not populate those translated
            // D3D-style draw parameter bindings.
            IRRuntimeDrawPrimitives(encoder, pipeline->primitiveType, indirectParams->buffer, offsetBytes);
            offsetBytes += sizeof(DrawIndirectArguments);
        }
    }
    
    void CommandList::drawIndexedIndirect(uint32_t offsetBytes, uint32_t drawCount)
    {
        if (!m_CurrentGraphicsStateValid)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexedIndirect skipped: graphics state invalid");
            return;
        }
        id<MTLRenderCommandEncoder> encoder = getOrCreateRenderEncoder();
        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indexBuffer = static_cast<Buffer*>(m_CurrentGraphicsState.indexBuffer.buffer);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        if (!encoder || !pipeline || !indexBuffer || !indexBuffer->buffer || !indirectParams || !indirectParams->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexedIndirect skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no") +
                    " index='" + (indexBuffer ? indexBuffer->desc.debugName : std::string("<null>")) +
                    "' indirect='" + (indirectParams ? indirectParams->desc.debugName : std::string("<null>")) + "'");
            return;
        }

        const uint64_t requiredBytes = uint64_t(offsetBytes) + uint64_t(drawCount) * sizeof(DrawIndexedIndirectArguments);
        if (requiredBytes > indirectParams->desc.byteSize)
            m_Context.warning("[metal3-trace] drawIndexedIndirect range exceeds args buffer: required=" +
                std::to_string(requiredBytes) + " bufferSize=" + std::to_string(indirectParams->desc.byteSize) +
                " args='" + indirectParams->desc.debugName + "'");

        static int drawIndexedIndirectLogCount = 0;
        if (traceMetalRuntime() && drawIndexedIndirectLogCount++ < 96)
            m_Context.info("[metal3-trace] drawIndexedIndirect count=" + std::to_string(drawCount) +
                " offset=" + std::to_string(offsetBytes) +
                " index='" + indexBuffer->desc.debugName +
                "' args='" + indirectParams->desc.debugName + "'");

        for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex)
        {
            // Keep converted vertex shaders in the same draw-parameter path as
            // direct indexed draws. This is especially important for indirect
            // base-instance/start-instance data produced by GPU culling.
            IRRuntimeDrawIndexedPrimitives(encoder,
                pipeline->primitiveType,
                convertIndexFormat(m_CurrentGraphicsState.indexBuffer.format),
                indexBuffer->buffer,
                m_CurrentGraphicsState.indexBuffer.offset,
                indirectParams->buffer,
                offsetBytes);
            offsetBytes += sizeof(DrawIndexedIndirectArguments);
        }
    }

    void CommandList::drawIndexedIndirectCount(uint32_t paramOffsetBytes, uint32_t countOffsetBytes, uint32_t maxDrawCount)
    {
        if (!m_CurrentGraphicsStateValid || maxDrawCount == 0)
            return;

        auto* pipeline = static_cast<GraphicsPipeline*>(m_CurrentGraphicsState.pipeline);
        auto* indexBuffer = static_cast<Buffer*>(m_CurrentGraphicsState.indexBuffer.buffer);
        auto* indirectParams = static_cast<Buffer*>(m_CurrentGraphicsState.indirectParams);
        auto* indirectCount = static_cast<Buffer*>(m_CurrentGraphicsState.indirectCountBuffer);
        if (!pipeline || !indexBuffer || !indexBuffer->buffer || !indirectParams || !indirectParams->buffer ||
            !indirectCount || !indirectCount->buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] drawIndexedIndirectCount skipped: missing pipeline/index/args/count buffer");
            return;
        }

        const IndexedIndirectIcbFillState& fillState = getIndexedIndirectIcbFillState(m_Context);
        if (!fillState.pipeline || !fillState.icbEncoder)
        {
            m_Context.warning("[metal3] counted indexed indirect draw skipped because the ICB helper pipeline is unavailable");
            return;
        }

        id<MTLRenderCommandEncoder> renderEncoder = getOrCreateRenderEncoder();
        if (!renderEncoder)
            return;
        endEncoding();
        // Metal supports indexed indirect draws, but not NVRHI's counted
        // indexed indirect operation directly. Generate a render ICB on the GPU
        // from the same args/count buffers and execute it with a GPU-written
        // MTLIndirectCommandBufferExecutionRange.

        MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeDrawIndexed;
        // The helper compute pass only writes the per-draw indexed commands.
        // Pipeline state, vertex buffers, and the MSC argument table are inherited
        // from the render encoder that later executes the ICB, so the graphics
        // state must be restored before executeCommandsInBuffer below.
        icbDesc.inheritPipelineState = YES;
        icbDesc.inheritBuffers = YES;
        icbDesc.maxVertexBufferBindCount = std::max<NSUInteger>(c_IrDrawArgumentsBindPoint + 1, c_MscVertexBufferBindPoint + 1);
        icbDesc.maxFragmentBufferBindCount = 0;

        id<MTLIndirectCommandBuffer> icb =
            [m_Context.device newIndirectCommandBufferWithDescriptor:icbDesc
                                                     maxCommandCount:NSUInteger(maxDrawCount)
                                                            options:MTLResourceStorageModePrivate];
        id<MTLBuffer> executionRange =
            [m_Context.device newBufferWithLength:sizeof(MTLIndirectCommandBufferExecutionRange)
                                          options:MTLResourceStorageModePrivate];
        id<MTLBuffer> paramsBuffer =
            [m_Context.device newBufferWithLength:sizeof(IcbIndexedIndirectParams)
                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> icbArgumentBuffer =
            [m_Context.device newBufferWithLength:fillState.icbEncoder.encodedLength
                                          options:MTLResourceStorageModeShared];
        if (!icb || !executionRange || !paramsBuffer || !icbArgumentBuffer)
        {
            m_Context.error("[metal3] failed to allocate counted indexed indirect ICB resources");
            return;
        }

        // Metal compute shaders receive an indirect command buffer through an
        // argument buffer. This encodes the ICB object into that argument buffer
        // so nvrhi_metal3_fill_indexed_indirect_icb can write render commands.
        [fillState.icbEncoder setArgumentBuffer:icbArgumentBuffer offset:0];
        [fillState.icbEncoder setIndirectCommandBuffer:icb atIndex:0];

        // Small CPU-written constant block consumed by the helper shader. It
        // describes where to read draw args/counts, how to interpret the index
        // buffer, and which MSC runtime vertex-buffer slot should receive each
        // draw's original D3D-style indirect argument record.
        auto* params = reinterpret_cast<IcbIndexedIndirectParams*>([paramsBuffer contents]);
        params->maxDrawCount = maxDrawCount;
        params->paramOffsetBytes = paramOffsetBytes;
        params->countOffsetBytes = countOffsetBytes;
        params->indexType = m_CurrentGraphicsState.indexBuffer.format == Format::R16_UINT ? 0u : 1u;
        params->indexBufferOffset = uint32_t(m_CurrentGraphicsState.indexBuffer.offset);
        params->primitiveType = uint32_t(pipeline->primitiveType);
        params->drawArgumentsBindPoint = c_IrDrawArgumentsBindPoint;

        m_ReferencedNativeResources.push_back(icb);
        m_ReferencedNativeResources.push_back(executionRange);
        m_ReferencedNativeBuffers.push_back(paramsBuffer);
        m_ReferencedNativeBuffers.push_back(icbArgumentBuffer);

        // ICB memory is reused by Metal internally; reset the command range before
        // the compute pass selectively fills commands 0..min(gpuCount,maxDrawCount).
        id<MTLBlitCommandEncoder> blit = [trackedCmdBuffer blitCommandEncoder];
        [blit resetCommandsInBuffer:icb withRange:NSMakeRange(0, maxDrawCount)];
        [blit endEncoding];

        // Fill the ICB on the GPU. Thread 0 writes the execution range from the
        // GPU count buffer, and each thread below that count writes one indexed
        // draw command from indirectParams[tid].
        id<MTLComputeCommandEncoder> compute = [trackedCmdBuffer computeCommandEncoder];
        [compute setComputePipelineState:fillState.pipeline];
        [compute setBuffer:indirectParams->buffer offset:0 atIndex:0];
        [compute setBuffer:indirectCount->buffer offset:0 atIndex:1];
        [compute setBuffer:executionRange offset:0 atIndex:2];
        [compute setBuffer:icbArgumentBuffer offset:0 atIndex:3];
        [compute setBuffer:indexBuffer->buffer offset:0 atIndex:4];
        [compute setBuffer:paramsBuffer offset:0 atIndex:5];
        [compute useResource:indirectParams->buffer usage:MTLResourceUsageRead];
        [compute useResource:indirectCount->buffer usage:MTLResourceUsageRead];
        [compute useResource:indexBuffer->buffer usage:MTLResourceUsageRead];
        [compute useResource:executionRange usage:MTLResourceUsageWrite];
        [compute useResource:icb usage:MTLResourceUsageWrite];
        const NSUInteger threads = std::max<NSUInteger>(1, fillState.pipeline.threadExecutionWidth);
        const NSUInteger groups = (NSUInteger(maxDrawCount) + threads - 1) / threads;
        [compute dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [compute endEncoding];

        renderEncoder = getOrCreateRenderEncoder();
        // The blit/compute work above closes the original render encoder. Since
        // this ICB inherits pipeline and buffer state from the executing render
        // encoder, restore the full graphics state before executeCommandsInBuffer.
        applyGraphicsStateToEncoder(renderEncoder, m_CurrentGraphicsState);
        [renderEncoder useResource:icb usage:MTLResourceUsageRead];
        [renderEncoder useResource:executionRange usage:MTLResourceUsageRead];
        [renderEncoder useResource:indirectParams->buffer usage:MTLResourceUsageRead];
        [renderEncoder useResource:indexBuffer->buffer usage:MTLResourceUsageRead];
        // Execute only the GPU-written range. This is what turns NVRHI's count
        // buffer into Metal's MTLIndirectCommandBufferExecutionRange.
        [renderEncoder executeCommandsInBuffer:icb indirectBuffer:executionRange indirectBufferOffset:0];
    }

    void CommandList::setComputeState(const ComputeState& state)
    {
        m_CurrentComputeState = state;
        m_CurrentComputeStateValid = true;
        endEncoding();
        id<MTLComputeCommandEncoder> encoder = getOrCreateComputeEncoder();
        auto* pipeline = static_cast<ComputePipeline*>(state.pipeline);
        if (!encoder || !pipeline)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] setComputeState skipped: encoder=" +
                    std::string(encoder ? "yes" : "no") + " pipeline=" + (pipeline ? "yes" : "no"));
            return;
        }

        [encoder setComputePipelineState:pipeline->pipeline];
        applyComputeBindings(encoder, state);

        id<MTLBuffer> argumentBuffer = getOrCreateArgumentTable(pipeline->computeBindingPlan, state.bindings);
        if (argumentBuffer)
        {
            m_ReferencedNativeBuffers.push_back(argumentBuffer);
            [encoder setBuffer:argumentBuffer offset:0 atIndex:kIRArgumentBufferBindPoint];
            [encoder useResource:argumentBuffer usage:MTLResourceUsageRead];
            useArgumentTableResources(encoder, state.bindings, pipeline->computeBindingPlan);
        }
        else if (traceMetalRuntime() && pipeline->computeBindingPlan.valid && pipeline->computeBindingPlan.resourceCount != 0)
        {
            static int missingComputeArgBufferLogs = 0;
            if (missingComputeArgBufferLogs++ < 16)
                m_Context.warning("[metal3-trace] setComputeState created no cached argument table");
        }
    }

    void CommandList::applyGraphicsStateToEncoder(id<MTLRenderCommandEncoder> encoder, const GraphicsState& state)
    {
        auto* pipeline = static_cast<GraphicsPipeline*>(state.pipeline);
        if (!encoder || !pipeline)
            return;

        [encoder setRenderPipelineState:pipeline->pipeline];
        [encoder setDepthStencilState:pipeline->depthStencilState];
        [encoder setCullMode:pipeline->cullMode];
        [encoder setFrontFacingWinding:pipeline->frontWinding];

        for (const Viewport& vp : state.viewport.viewports)
            [encoder setViewport:MTLViewport{ vp.minX, vp.minY, vp.width(), vp.height(), vp.minZ, vp.maxZ }];
        for (const Rect& rect : state.viewport.scissorRects)
            [encoder setScissorRect:MTLScissorRect{ NSUInteger(rect.minX), NSUInteger(rect.minY), NSUInteger(rect.width()), NSUInteger(rect.height()) }];

        for (const VertexBufferBinding& vb : state.vertexBuffers)
        {
            auto* buffer = static_cast<Buffer*>(vb.buffer);
            if (traceMetalRuntime() && !buffer)
                m_Context.warning("[metal3-trace] null vertex buffer at slot " + std::to_string(vb.slot));
            [encoder setVertexBuffer:buffer ? buffer->buffer : nil offset:NSUInteger(vb.offset) atIndex:vb.slot];
            [encoder setVertexBuffer:buffer ? buffer->buffer : nil offset:NSUInteger(vb.offset) atIndex:c_MscVertexBufferBindPoint + vb.slot];
        }
        // path for finding and binding volatile command buffers
        applyGraphicsBindings(encoder, state);

        // for binding the rest of the textures, buffers, resources
        // argument table is prepared (build or used from before)
        id<MTLBuffer> vertexArgumentBuffer = getOrCreateArgumentTable(pipeline->vertexBindingPlan, state.bindings);
        if (vertexArgumentBuffer)
        {
            m_ReferencedNativeBuffers.push_back(vertexArgumentBuffer);
            [encoder setVertexBuffer:vertexArgumentBuffer offset:0 atIndex:kIRArgumentBufferBindPoint];
            [encoder useResource:vertexArgumentBuffer usage:MTLResourceUsageRead];
            useArgumentTableResources(encoder, state.bindings, pipeline->vertexBindingPlan, MTLRenderStageVertex);
        }

        if (pipeline->desc.PS)
        {
            id<MTLBuffer> fragmentArgumentBuffer = getOrCreateArgumentTable(pipeline->fragmentBindingPlan, state.bindings);
            if (fragmentArgumentBuffer)
            {
                m_ReferencedNativeBuffers.push_back(fragmentArgumentBuffer);
                [encoder setFragmentBuffer:fragmentArgumentBuffer offset:0 atIndex:kIRArgumentBufferBindPoint];
                [encoder useResource:fragmentArgumentBuffer usage:MTLResourceUsageRead];
                useArgumentTableResources(encoder, state.bindings, pipeline->fragmentBindingPlan, MTLRenderStageFragment);
            }
        }
    }

    bool CommandList::bindVolatileConstantBuffer(id<MTLRenderCommandEncoder> encoder, const BindingSetItem& item)
    {
        if (item.type != ResourceType::VolatileConstantBuffer)
            return false;

        auto* buffer = static_cast<Buffer*>(item.resourceHandle);
        auto allocationIt = buffer ? m_VolatileBufferAllocations.find(buffer) : m_VolatileBufferAllocations.end();
        if (!buffer || allocationIt == m_VolatileBufferAllocations.end() || !allocationIt->second.buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] volatile constant buffer not written before graphics bind: slot=" +
                    std::to_string(item.slot) + " name='" + (buffer ? buffer->desc.debugName : std::string("<null>")) + "'");
            return false;
        }

        const BufferRange range = item.range.resolve(buffer->desc);
        const UploadAllocation& allocation = allocationIt->second;
        const NSUInteger offset = allocation.offset + NSUInteger(range.byteOffset);
        [encoder setVertexBuffer:allocation.buffer offset:offset atIndex:item.slot];
        [encoder setFragmentBuffer:allocation.buffer offset:offset atIndex:item.slot];
        [encoder useResource:allocation.buffer usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];
        return true;
    }

    bool CommandList::bindVolatileConstantBuffer(id<MTLComputeCommandEncoder> encoder, const BindingSetItem& item)
    {
        if (item.type != ResourceType::VolatileConstantBuffer)
            return false;

        auto* buffer = static_cast<Buffer*>(item.resourceHandle);
        auto allocationIt = buffer ? m_VolatileBufferAllocations.find(buffer) : m_VolatileBufferAllocations.end();
        if (!buffer || allocationIt == m_VolatileBufferAllocations.end() || !allocationIt->second.buffer)
        {
            if (traceMetalRuntime())
                m_Context.warning("[metal3-trace] volatile constant buffer not written before compute bind: slot=" +
                    std::to_string(item.slot) + " name='" + (buffer ? buffer->desc.debugName : std::string("<null>")) + "'");
            return false;
        }

        const BufferRange range = item.range.resolve(buffer->desc);
        const UploadAllocation& allocation = allocationIt->second;
        const NSUInteger offset = allocation.offset + NSUInteger(range.byteOffset);
        [encoder setBuffer:allocation.buffer offset:offset atIndex:item.slot];
        [encoder useResource:allocation.buffer usage:MTLResourceUsageRead];
        return true;
    }
    /* 
    // Builds the Metal Shader Converter descriptor table for one shader stage.
    // The pipeline's reflected binding plan tells us which argument-table index
    // each HLSL resource occupies, and layoutIndex selects the matching NVRHI
    // binding set from the current graphics/compute state. Each matched
    // MetalBindingResource is encoded into an IRDescriptorTableEntry. The table
    // is cached for the command list when the same plan and binding sets are
    // reused by multiple draws/dispatches.
    */
    id<MTLBuffer> CommandList::getOrCreateArgumentTable(const MetalStageBindingPlan& plan, const BindingSetVector& bindingSets)
    {
        if (!plan.valid || plan.resourceCount == 0)
        {
            if (!plan.valid && traceMetalRuntime())
            {
                static int invalidPlanLogCount = 0;
                if (invalidPlanLogCount++ < 32)
                    m_Context.warning("[metal3-trace] missing reflected binding plan for " +
                        std::string(utils::ShaderStageToString(plan.stage)) + "; regular shader resources are not directly bound");
            }
            return nil; 
        }

        MetalArgumentTableCacheKey key = makeArgumentTableCacheKey(plan, bindingSets);
        for (const MetalArgumentTableCacheEntry& entry : m_ArgumentTableCache)
        {
            if (entry.key == key)
                return entry.argumentBuffer;
        }

        const NSUInteger bufferSize = sizeof(IRDescriptorTableEntry) * plan.resourceCount;
        id<MTLBuffer> argumentBuffer = [m_Context.device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        if (!argumentBuffer)
            return nil;

        auto* entries = static_cast<IRDescriptorTableEntry*>([argumentBuffer contents]);
        std::memset(entries, 0, bufferSize);

        for (const MetalBindingPlanEntry& planEntry : plan.entries)
        {
            if (planEntry.argumentIndex >= plan.resourceCount)
                continue;

            if (usesDirectVolatileConstantBufferBinding(planEntry))
                continue;

            if (!planEntry.layoutMatched)
            {
                if (traceMetalRuntime())
                {
                    static int unmatchedPlanLogCount = 0;
                    if (unmatchedPlanLogCount++ < 64)
                        m_Context.warning("[metal3-trace] reflected " +
                            std::string(utils::ShaderStageToString(plan.stage)) +
                            " binding has no NVRHI layout match: arg=" + std::to_string(planEntry.argumentIndex) +
                            " slot=" + std::to_string(planEntry.slot) +
                            " space=" + std::to_string(planEntry.space));
                }
                continue;
            }

            const MetalBindingResource* resource = findArgumentTableResource(bindingSets, planEntry);
            if (resource)
                encodeArgumentTableEntry(&entries[planEntry.argumentIndex], *resource);
            else if (traceMetalRuntime())
            {
                static int missingResourceLogCount = 0;
                if (missingResourceLogCount++ < 64)
                    m_Context.warning("[metal3-trace] no binding-set resource for reflected " +
                        std::string(utils::ShaderStageToString(plan.stage)) +
                        " binding: arg=" + std::to_string(planEntry.argumentIndex) +
                        " slot=" + std::to_string(planEntry.slot) +
                        " space=" + std::to_string(planEntry.space));
            }
        }

        MetalArgumentTableCacheEntry entry;
        entry.key = std::move(key);
        entry.argumentBuffer = argumentBuffer;
        m_ArgumentTableCache.push_back(std::move(entry));
        return argumentBuffer;
    }

    // this path basically exclusively exists for binding volatile contant buffers
    // regular resources go through the argument table path
    void CommandList::applyGraphicsBindings(id<MTLRenderCommandEncoder> encoder, const GraphicsState& state)
    {
        for (IBindingSet* bindingSet : state.bindings)
        {
            auto* set = static_cast<BindingSet*>(bindingSet);
            if (!set) continue;
            referenceBindingSet(set);
            for (const BindingSetItem& item : set->desc.bindings)
            {
                if (traceMetalRuntime() && !item.resourceHandle)
                    m_Context.warning("[metal3-trace] graphics binding has null resource: type=" +
                        std::string(resourceTypeName(item.type)) + " slot=" + std::to_string(item.slot));
                bindVolatileConstantBuffer(encoder, item);
            }
        }
    }

    void CommandList::referenceBindingSet(BindingSet* bindingSet)
    {
        if (bindingSet)
            m_ReferencedBindingSets.push_back(bindingSet);
    }

}
