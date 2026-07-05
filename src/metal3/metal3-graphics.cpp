#include "metal3-backend.h"
#include <cctype>

namespace nvrhi::metal3
{
    static std::string normalizeMscVertexInputName(const std::string& value)
    {
        std::string result;
        result.reserve(value.size());
        for (unsigned char ch : value)
        {
            if (std::isalnum(ch))
                result.push_back(static_cast<char>(std::tolower(ch)));
        }

        while (!result.empty() && std::isdigit(static_cast<unsigned char>(result.back())))
            result.pop_back();

        return result;
    }

    // c_MscVertexBufferBindPoint -> MSC's vertex-buffer binding slot where we start binding actual app vertex buffers
    // so vertex buffer slot 0 gets bound at index 6
    static constexpr uint32_t c_MscVertexBufferBindPoint = 6;
    // c_MscVertexAttributeBase -> the Metal vertex attribute index where MSC expects user vertex attributes to begin
    static constexpr uint32_t c_MscVertexAttributeBase = 11;
    // c_MetalMaxVertexAttributes -> just a guard for Metal’s vertex attribute descriptor array
    static constexpr uint32_t c_MetalMaxVertexAttributes = 31;

    static void setVertexAttribute(MTLVertexDescriptor* vertexDescriptor, uint32_t attributeIndex, const VertexAttributeDesc& attr)
    {
        vertexDescriptor.attributes[attributeIndex].format = convertVertexFormat(attr.format);
        vertexDescriptor.attributes[attributeIndex].offset = attr.offset;
        vertexDescriptor.attributes[attributeIndex].bufferIndex = c_MscVertexBufferBindPoint + attr.bufferIndex;
    }

    InputLayoutHandle Device::createInputLayout(const VertexAttributeDesc* d, uint32_t attributeCount, IShader* vertexShader)
    {
        auto* shader = static_cast<Shader*>(vertexShader);
        InputLayout* layout = new InputLayout();
        layout->attributes.assign(d, d + attributeCount);
        layout->vertexDescriptor = [[MTLVertexDescriptor alloc] init];

        for (uint32_t index = 0; index < attributeCount; ++index)
        {
            const VertexAttributeDesc& attr = d[index];
            uint32_t attributeIndex = index;
            if (shader && !shader->mscReflection.vertexInputAttributes.empty())
            {
                const std::string semantic = normalizeMscVertexInputName(attr.name);
                auto it = shader->mscReflection.vertexInputAttributes.find(semantic);
                if (it != shader->mscReflection.vertexInputAttributes.end())
                {
                    /* 
                    // MSC reflection reports the HLSL input-signature index.
                    // The generated Metal entry point places user attributes
                    // after its reserved ABI inputs, so Metal validation names
                    // POSITION0 as attribute 11, RECT0 as 12, etc.
                    */
                    attributeIndex = c_MscVertexAttributeBase + it->second;
                }
                else
                {
                    static int missingSemanticLogCount = 0;
                    if (missingSemanticLogCount++ < 32)
                    {
                        m_Context.warning("[metal3] vertex attribute '" + attr.name +
                            "' was not found in MSC reflection for shader '" +
                            (shader ? shader->desc.debugName : std::string("<null>")) +
                            "'; using input-layout index " + std::to_string(index));
                    }
                }
            }

            if (attributeIndex < c_MetalMaxVertexAttributes)
                setVertexAttribute(layout->vertexDescriptor, attributeIndex, attr);

            uint32_t bufferIndex = c_MscVertexBufferBindPoint + attr.bufferIndex;
            layout->vertexDescriptor.layouts[bufferIndex].stride = attr.elementStride;
            layout->vertexDescriptor.layouts[bufferIndex].stepFunction = attr.isInstanced ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
            layout->vertexDescriptor.layouts[bufferIndex].stepRate = 1;
        }

        return InputLayoutHandle::Create(layout);
    }

    const VertexAttributeDesc* InputLayout::getAttributeDesc(uint32_t index) const
    {
        return index < attributes.size() ? &attributes[index] : nullptr;
    }

    FramebufferHandle Device::createFramebuffer(const FramebufferDesc& desc)
    {
        Framebuffer* framebuffer = new Framebuffer();
        framebuffer->desc = desc;
        framebuffer->framebufferInfo = FramebufferInfoEx(desc);
        return FramebufferHandle::Create(framebuffer);
    }

    static void applyColorBlend(MTLRenderPipelineColorAttachmentDescriptor* attachment, const BlendState::RenderTarget& blend)
    {
        attachment.blendingEnabled = blend.blendEnable;
        attachment.sourceRGBBlendFactor = convertBlendFactor(blend.srcBlend);
        attachment.destinationRGBBlendFactor = convertBlendFactor(blend.destBlend);
        attachment.rgbBlendOperation = convertBlendOp(blend.blendOp);
        attachment.sourceAlphaBlendFactor = convertBlendFactor(blend.srcBlendAlpha);
        attachment.destinationAlphaBlendFactor = convertBlendFactor(blend.destBlendAlpha);
        attachment.alphaBlendOperation = convertBlendOp(blend.blendOpAlpha);
        attachment.writeMask = MTLColorWriteMaskNone;
        if (!!(blend.colorWriteMask & ColorMask::Red)) attachment.writeMask |= MTLColorWriteMaskRed;
        if (!!(blend.colorWriteMask & ColorMask::Green)) attachment.writeMask |= MTLColorWriteMaskGreen;
        if (!!(blend.colorWriteMask & ColorMask::Blue)) attachment.writeMask |= MTLColorWriteMaskBlue;
        if (!!(blend.colorWriteMask & ColorMask::Alpha)) attachment.writeMask |= MTLColorWriteMaskAlpha;
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, FramebufferInfo const& fbinfo)
    {
        auto* vs = static_cast<Shader*>(desc.VS.Get());
        auto* ps = static_cast<Shader*>(desc.PS.Get());
        if (!vs || !vs->function)
            return nullptr;

        MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
        pd.vertexFunction = vs->function;
        pd.fragmentFunction = ps ? ps->function : nil;  // can be nil for depth pass pipelines
        pd.supportIndirectCommandBuffers = YES;
        if (desc.inputLayout)
        {
            auto* inputLayout = static_cast<InputLayout*>(desc.inputLayout.Get());
            pd.vertexDescriptor = inputLayout ? inputLayout->vertexDescriptor : nil;
        }

        for (NSUInteger i = 0; i < fbinfo.colorFormats.size(); ++i)
        {
            pd.colorAttachments[i].pixelFormat = convertFormat(fbinfo.colorFormats[i]);
            applyColorBlend(pd.colorAttachments[i], desc.renderState.blendState.targets[i]);
        }
        if (fbinfo.depthFormat != Format::UNKNOWN)
            pd.depthAttachmentPixelFormat = convertFormat(fbinfo.depthFormat);
        pd.rasterSampleCount = fbinfo.sampleCount;

        NSError* error = nil;
        id<MTLRenderPipelineState> nativePipeline = [m_Context.device newRenderPipelineStateWithDescriptor:pd error:&error];
        if (!nativePipeline)
        {
            std::string message = "[nvrhi] Failed to create Metal render pipeline";
            if (error)
                message += std::string(": ") + [[error localizedDescription] UTF8String];
            m_Context.error(message);
            return nullptr;
        }

        MTLDepthStencilDescriptor* dd = [[MTLDepthStencilDescriptor alloc] init];
        dd.depthCompareFunction = desc.renderState.depthStencilState.depthTestEnable
            ? convertCompareFunction(desc.renderState.depthStencilState.depthFunc)
            : MTLCompareFunctionAlways;
        dd.depthWriteEnabled = desc.renderState.depthStencilState.depthWriteEnable;

        GraphicsPipeline* pipeline = new GraphicsPipeline();
        pipeline->desc = desc;
        pipeline->framebufferInfo = fbinfo;
        pipeline->pipeline = nativePipeline;
        pipeline->depthStencilState = [m_Context.device newDepthStencilStateWithDescriptor:dd];
        pipeline->primitiveType = convertPrimitiveType(desc.primType);
        pipeline->cullMode = convertCullMode(desc.renderState.rasterState.cullMode);
        pipeline->frontWinding = convertWinding(desc.renderState.rasterState.frontCounterClockwise);
        return GraphicsPipelineHandle::Create(pipeline);
    }

    GraphicsPipelineHandle Device::createGraphicsPipeline(const GraphicsPipelineDesc& desc, IFramebuffer* fb)
    {
        return createGraphicsPipeline(desc, fb ? FramebufferInfo(fb->getDesc()) : FramebufferInfo());
    }

    Object GraphicsPipeline::getNativeObject(ObjectType objectType)
    {
        if (objectType == ObjectTypes::MTL3_RenderPipeline)
            return Object((__bridge void*)pipeline);
        return nullptr;
    }
}
