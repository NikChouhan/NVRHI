/*
* Example code to test the metal backend code
*/

// main.cpp - NVRHI Metal 3 triangle, Cocoa window
//
// Shader blobs (triangle_vs.dxil / triangle_ps.dxil) can be compiled offline
// with dxc:
//   dxc -T vs_6_0 -E VSMain -Fo triangle_vs.dxil shaders.hlsl
//   dxc -T ps_6_0 -E PSMain -Fo triangle_ps.dxil shaders.hlsl

// the idea is to transpile* .hlsl to *.dxil files with DXC and then to *.metallib files with MSC (metal shader converter)
// it will all be done offline for now, then automated later

#include <nvrhi/nvrhi.h>
#include <nvrhi/metal3.h>
#include <nvrhi/utils.h>
#include <nvrhi/validation.h>

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int           WINDOW_W        = 1280;
static constexpr int           WINDOW_H        = 720;
static constexpr uint32_t      FRAME_COUNT     = 1;
static constexpr nvrhi::Format NVRHI_SWAP_FMT  = nvrhi::Format::RGBA8_UNORM;

static bool g_ShouldClose = false;

static std::vector<uint8_t> ReadFile(const char* path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("Cannot open file: ") + path);

    auto sz = static_cast<size_t>(f.tellg());
    std::vector<uint8_t> data(sz);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(sz));
    return data;
}

struct MessageCallback : public nvrhi::IMessageCallback
{
    void message(nvrhi::MessageSeverity severity, const char* msg) override
    {
        const char* sev = "INFO";
        if (severity == nvrhi::MessageSeverity::Warning) sev = "WARN";
        if (severity == nvrhi::MessageSeverity::Error) sev = "ERROR";
        if (severity == nvrhi::MessageSeverity::Fatal) sev = "FATAL";

        fprintf(stderr, "[NVRHI %s] %s\n", sev, msg);
    }
};

@interface NvrhiWindowDelegate : NSObject <NSWindowDelegate>
@end

@implementation NvrhiWindowDelegate
- (BOOL)windowShouldClose:(id)sender
{
    (void)sender;
    g_ShouldClose = true;
    return YES;
}
@end

struct MacPlatform
{
    NSWindow* window = nil;
    NvrhiWindowDelegate* windowDelegate = nil;
    id<MTLDevice> metalDevice = nil;
    id<MTLCommandQueue> commandQueue = nil;
    CAMetalLayer* layer = nil;
    id<CAMetalDrawable> currentDrawable = nil;

    nvrhi::DeviceHandle createDevice(MessageCallback& msgCb)
    {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect frame = NSMakeRect(0, 0, WINDOW_W, WINDOW_H);
        window = [[NSWindow alloc]
            initWithContentRect:frame
            styleMask:(NSWindowStyleMaskTitled |
                       NSWindowStyleMaskClosable |
                       NSWindowStyleMaskMiniaturizable |
                       NSWindowStyleMaskResizable)
            backing:NSBackingStoreBuffered
            defer:NO];

        [window setTitle:@"NVRHI Triangle"];
        [window center];

        windowDelegate = [[NvrhiWindowDelegate alloc] init];
        [window setDelegate:windowDelegate];

        metalDevice = MTLCreateSystemDefaultDevice();
        if (metalDevice == nil)
            throw std::runtime_error("MTLCreateSystemDefaultDevice failed");

        commandQueue = [metalDevice newCommandQueue];
        if (commandQueue == nil)
            throw std::runtime_error("newCommandQueue failed");

        layer = [CAMetalLayer layer];
        layer.device = metalDevice;
        layer.pixelFormat = nvrhi::metal3::convertFormat(NVRHI_SWAP_FMT);
        layer.drawableSize = CGSizeMake(WINDOW_W, WINDOW_H);
        layer.framebufferOnly = YES;

        NSView* contentView = [window contentView];
        [contentView setWantsLayer:YES];
        [contentView setLayer:layer];

        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        nvrhi::metal3::DeviceDesc nvDesc;
        nvDesc.errorCB = &msgCb;
        nvDesc.pDevice = metalDevice;
        nvDesc.commonQueue = commandQueue;

        return nvrhi::metal3::createDevice(nvDesc);
    }

    void pumpEvents()
    {
        for (;;) {
            NSEvent* event = [NSApp
                nextEventMatchingMask:NSEventMaskAny
                untilDate:nil
                inMode:NSDefaultRunLoopMode
                dequeue:YES];

            if (event == nil) break;

            if ([event type] == NSEventTypeKeyDown &&
                [[event charactersIgnoringModifiers] isEqualToString:@"\e"]) {
                g_ShouldClose = true;
            }

            [NSApp sendEvent:event];
        }
    }

    uint32_t acquireBackBuffer(
        nvrhi::IDevice* nvDevice,
        std::array<nvrhi::TextureHandle, FRAME_COUNT>& backBufferTextures,
        std::array<nvrhi::FramebufferHandle, FRAME_COUNT>& framebuffers)
    {
        if (currentDrawable != nil)
            return 0;

        currentDrawable = [layer nextDrawable];
        if (currentDrawable == nil)
            throw std::runtime_error("CAMetalLayer nextDrawable failed");

        nvrhi::TextureDesc td;
        td.width = WINDOW_W;
        td.height = WINDOW_H;
        td.format = NVRHI_SWAP_FMT;
        td.initialState = nvrhi::ResourceStates::Present;
        td.keepInitialState = true;
        td.isRenderTarget = true;
        td.debugName = "BackBuffer";

        backBufferTextures[0] = nvDevice->createHandleForNativeTexture(
            nvrhi::ObjectTypes::MTL3_Texture,
            nvrhi::Object((__bridge void*)currentDrawable.texture),
            td);

        nvrhi::FramebufferDesc fbDesc;
        fbDesc.addColorAttachment(backBufferTextures[0]);
        framebuffers[0] = nvDevice->createFramebuffer(fbDesc);

        return 0;
    }

    void present()
    {
        [currentDrawable present];
        currentDrawable = nil;
    }

    void destroy()
    {
        [window setDelegate:nil];
        [window close];
        window = nil;
        windowDelegate = nil;
        layer = nil;
        commandQueue = nil;
        metalDevice = nil;
    }
};

static int Run()
{
    MessageCallback msgCb;
    MacPlatform platform;

    nvrhi::DeviceHandle nvDevice = platform.createDevice(msgCb);

#ifdef _DEBUG
    nvDevice = nvrhi::validation::createValidationLayer(nvDevice);
#endif

    std::array<nvrhi::TextureHandle, FRAME_COUNT> backBufferTextures;
    std::array<nvrhi::FramebufferHandle, FRAME_COUNT> framebuffers;
    platform.acquireBackBuffer(nvDevice, backBufferTextures, framebuffers);

    auto vsBytes = ReadFile("triangle_vs.dxil");
    auto psBytes = ReadFile("triangle_ps.dxil");

    nvrhi::ShaderHandle vs = nvDevice->createShader(
        nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Vertex),
        vsBytes.data(), vsBytes.size());

    nvrhi::ShaderHandle ps = nvDevice->createShader(
        nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Pixel),
        psBytes.data(), psBytes.size());

    nvrhi::BindingLayoutHandle bindingLayout = nvDevice->createBindingLayout(
        nvrhi::BindingLayoutDesc()
            .setVisibility(nvrhi::ShaderType::All));

    nvrhi::BindingSetHandle bindingSet = nvDevice->createBindingSet(
        nvrhi::BindingSetDesc(),
        bindingLayout);

    nvrhi::GraphicsPipelineDesc psoDesc;
    psoDesc.VS = vs;
    psoDesc.PS = ps;
    psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
    psoDesc.addBindingLayout(bindingLayout);

    psoDesc.renderState.depthStencilState.depthTestEnable = false;
    psoDesc.renderState.depthStencilState.depthWriteEnable = false;
    psoDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;

    nvrhi::GraphicsPipelineHandle pso =
        nvDevice->createGraphicsPipeline(psoDesc, framebuffers[0]);

    nvrhi::CommandListHandle cmdList = nvDevice->createCommandList();

    while (!g_ShouldClose)
    {
        platform.pumpEvents();
        if (g_ShouldClose) break;

        uint32_t frameIdx = platform.acquireBackBuffer(
            nvDevice, backBufferTextures, framebuffers);

        cmdList->open();

        nvrhi::utils::ClearColorAttachment(
            cmdList, framebuffers[frameIdx], 0,
            nvrhi::Color(0.1f, 0.1f, 0.15f, 1.0f));

        nvrhi::GraphicsState state;
        state.pipeline = pso;
        state.framebuffer = framebuffers[frameIdx];
        state.viewport.addViewportAndScissorRect(
            nvrhi::Viewport(float(WINDOW_W), float(WINDOW_H)));
        state.bindings = { bindingSet };

        cmdList->setGraphicsState(state);

        nvrhi::DrawArguments drawArgs;
        drawArgs.vertexCount = 3;
        cmdList->draw(drawArgs);

        cmdList->close();

        nvDevice->executeCommandList(cmdList);
        platform.present();
        nvDevice->runGarbageCollection();
    }

    nvDevice->waitForIdle();
    nvDevice->runGarbageCollection();

    cmdList = nullptr;
    pso = nullptr;
    bindingSet = nullptr;
    bindingLayout = nullptr;
    vs = ps = nullptr;
    for (auto& fb : framebuffers) fb = nullptr;
    for (auto& bb : backBufferTextures) bb = nullptr;
    nvDevice = nullptr;

    platform.destroy();
    return 0;
}

int main()
{
    @autoreleasepool {
        try {
            return Run();
        } catch (const std::exception& e) {
            fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}
