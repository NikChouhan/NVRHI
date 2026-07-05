#include "metal3-backend.h"

namespace nvrhi::metal3
{
    SamplerHandle Device::createSampler(const SamplerDesc& d)
    {
        MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
        sd.minFilter = d.minFilter ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sd.magFilter = d.magFilter ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        sd.mipFilter = d.mipFilter ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
        sd.sAddressMode = convertSamplerAddressMode(d.addressU);
        sd.tAddressMode = convertSamplerAddressMode(d.addressV);
        sd.rAddressMode = convertSamplerAddressMode(d.addressW);
        sd.lodBias = d.mipBias;
        sd.maxAnisotropy = NSUInteger(std::max(1.f, d.maxAnisotropy));
        sd.supportArgumentBuffers = YES;

        Sampler* sampler = new Sampler();
        sampler->desc = d;
        sampler->sampler = [m_Context.device newSamplerStateWithDescriptor:sd];
        if (!sampler->sampler)
        {
            delete sampler;
            return nullptr;
        }
        return SamplerHandle::Create(sampler);
    }

    BindingLayoutHandle Device::createBindingLayout(const BindingLayoutDesc& desc)
    {
        BindingLayout* layout = new BindingLayout();
        layout->desc = desc;
        return BindingLayoutHandle::Create(layout);
    }

    BindingSetHandle Device::createBindingSet(const BindingSetDesc& desc, IBindingLayout* layout)
    {
        BindingSet* set = new BindingSet();
        set->desc = desc;
        set->layout = layout;
        return BindingSetHandle::Create(set);
    }
}
