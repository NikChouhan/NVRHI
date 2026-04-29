#pragma once

#include "metal3-backend.h"
#include "nvrhi/common/misc.h"
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
        return GraphicsAPI::METAL;
    }
    // placeholders
    Object Device::getNativeQueue(ObjectType objectType, CommandQueue queue)
    {
        (void)objectType;
        (void)queue;
        return nullptr;
    }

    bool Device::waitForIdle()
    {
        return true;
    }

    void Device::runGarbageCollection()
    {
    }

    MTLPixelFormat convertFormat(nvrhi::Format format)
    {
        switch (format)
        {
        case Format::RGBA8_UNORM:
            return MTLPixelFormatRGBA8Unorm;
        case Format::BGRA8_UNORM:
            return MTLPixelFormatBGRA8Unorm;
        case Format::SRGBA8_UNORM:
            return MTLPixelFormatRGBA8Unorm_sRGB;
        case Format::SBGRA8_UNORM:
            return MTLPixelFormatBGRA8Unorm_sRGB;
        case Format::R8_UNORM:
            return MTLPixelFormatR8Unorm;
        case Format::R16_FLOAT:
            return MTLPixelFormatR16Float;
        case Format::RG16_FLOAT:
            return MTLPixelFormatRG16Float;
        case Format::RGBA16_FLOAT:
            return MTLPixelFormatRGBA16Float;
        case Format::R32_FLOAT:
            return MTLPixelFormatR32Float;
        case Format::RG32_FLOAT:
            return MTLPixelFormatRG32Float;
        case Format::RGBA32_FLOAT:
            return MTLPixelFormatRGBA32Float;
        case Format::D32:
            return MTLPixelFormatDepth32Float;
        case Format::D24S8:
            return MTLPixelFormatDepth24Unorm_Stencil8;
        default:
            return MTLPixelFormatInvalid;
        }
    }
}
