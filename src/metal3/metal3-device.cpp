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
}
