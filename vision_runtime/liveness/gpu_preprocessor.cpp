#include "liveness/gpu_preprocessor.hpp"
#include "liveness/liveness_coverage_reduce.h"
#include "liveness/liveness_tensorize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include <d3d11_4.h>
#include <d3d11on12.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.ai.machinelearning.native.h>
#include <winrt/Windows.Foundation.Collections.h>

namespace vision_runtime::liveness {
namespace {

using namespace winrt::Windows::AI::MachineLearning;

// InsightFace addons/liveness.py contract: RGB uint8 aligned crop, replicate
// border, RGB/255 NCHW, and a separately warped zero-border coverage mask.
// NV12 conversion, crop and mask reduction all run here on the GPU.
struct alignas(16) ShaderParams {
    std::uint32_t src_width;
    std::uint32_t src_height;
    std::uint32_t dst_width;
    std::uint32_t dst_height;
    float inv00;
    float inv01;
    float inv02;
    float pad0;
    float inv10;
    float inv11;
    float inv12;
    float pad1;
};
static_assert(sizeof(ShaderParams) == 48);

void close_handle(HANDLE& handle) noexcept {
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
        ::CloseHandle(handle);
        handle = nullptr;
    }
}

void check_hr_stage(HRESULT hr, char const* stage) {
    if (SUCCEEDED(hr)) return;

    std::ostringstream message;
    message << "Liveness GPU stage '" << stage << "' gagal: HRESULT 0x" << std::hex
            << std::uppercase << static_cast<std::uint32_t>(hr);
    throw std::runtime_error(message.str());
}

} // namespace

struct LivenessGpuPreprocessor::Impl {
    explicit Impl(GpuFrame const& first_frame, LivenessPreprocessConfig cfg) : config(cfg) {
        initialize(first_frame);
    }

    ~Impl() {
        if (interop_context) {
            interop_context->ClearState();
            interop_context->Flush();
        }
        shared_nv12_handle = nullptr;
        close_handle(camera_fence_handle);
    }

    LivenessPreprocessConfig config{};

    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;

    winrt::com_ptr<ID3D11Device> camera_device;
    winrt::com_ptr<ID3D11Device5> camera_device5;
    winrt::com_ptr<ID3D11DeviceContext> camera_context;
    winrt::com_ptr<ID3D11DeviceContext4> camera_context4;

    winrt::com_ptr<ID3D12Device> d3d12_device;
    winrt::com_ptr<ID3D12CommandQueue> d3d12_queue;

    winrt::com_ptr<ID3D11Device> interop_device;
    winrt::com_ptr<ID3D11Device3> interop_device3;
    winrt::com_ptr<ID3D11DeviceContext> interop_context;
    winrt::com_ptr<ID3D11On12Device> interop_11on12;

    winrt::com_ptr<ID3D11Texture2D> shared_nv12_camera;
    HANDLE shared_nv12_handle = nullptr;
    winrt::com_ptr<ID3D11Texture2D> shared_nv12_interop;
    winrt::com_ptr<ID3D11ShaderResourceView> y_srv;
    winrt::com_ptr<ID3D11ShaderResourceView> uv_srv;

    winrt::com_ptr<ID3D11ComputeShader> tensorize_shader;
    winrt::com_ptr<ID3D11ComputeShader> coverage_reduce_shader;
    winrt::com_ptr<ID3D11Buffer> coverage_buffer;
    winrt::com_ptr<ID3D11Buffer> coverage_readback;
    winrt::com_ptr<ID3D11UnorderedAccessView> coverage_uav;
    winrt::com_ptr<ID3D11SamplerState> linear_sampler;
    winrt::com_ptr<ID3D11Buffer> params_buffer;

    winrt::com_ptr<ID3D12Resource> tensor_buffer_d3d12;
    winrt::com_ptr<ID3D11Buffer> tensor_buffer_wrapped11;
    winrt::com_ptr<ID3D11UnorderedAccessView> tensor_uav;

    winrt::com_ptr<ID3D11Fence> camera_fence11;
    HANDLE camera_fence_handle = nullptr;
    winrt::com_ptr<ID3D12Fence> camera_fence12;
    std::uint64_t camera_fence_value = 0;

    LearningModelDevice learning_device{nullptr};
    TensorFloat tensor{nullptr};

    bool frame_prepared = false;

    void initialize(GpuFrame const& frame) {
        if (!frame.valid()) {
            throw std::invalid_argument("GpuFrame Liveness tidak valid");
        }
        if (frame.dxgi_format != DXGI_FORMAT_NV12) {
            throw std::invalid_argument("Liveness GPU preprocessor membutuhkan DXGI_FORMAT_NV12");
        }
        if (config.input_width != 80 || config.input_height != 80) {
            throw std::invalid_argument("InsightFace liveness input dikunci ke 80x80");
        }

        source_width = frame.width;
        source_height = frame.height;

        ID3D11Device* raw_device = nullptr;
        frame.texture->GetDevice(&raw_device);
        if (raw_device == nullptr) {
            throw std::runtime_error("Gagal memperoleh D3D11 device dari camera texture");
        }
        camera_device.attach(raw_device);

        camera_device5 = camera_device.try_as<ID3D11Device5>();
        if (!camera_device5) {
            throw std::runtime_error("ID3D11Device5 tidak tersedia untuk Liveness cross-API fence");
        }

        camera_device->GetImmediateContext(camera_context.put());
        camera_context4 = camera_context.try_as<ID3D11DeviceContext4>();
        if (!camera_context4) {
            throw std::runtime_error("ID3D11DeviceContext4 tidak tersedia untuk Liveness");
        }

        create_d3d12_runtime();
        create_learning_device();
        create_interop_device();
        create_shared_nv12(frame);
        create_shaders();
        create_coverage_buffers();
        create_sampler();
        create_params_buffer();
        create_tensor_d3d12_first();
        create_camera_fence();
    }

    void create_d3d12_runtime() {
        auto dxgi_device = camera_device.as<IDXGIDevice>();
        winrt::com_ptr<IDXGIAdapter> adapter;
        check_hr_stage(dxgi_device->GetAdapter(adapter.put()), "IDXGIDevice::GetAdapter");

        check_hr_stage(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0,
                                         __uuidof(ID3D12Device), d3d12_device.put_void()),
                       "D3D12CreateDevice");

        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        q.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        q.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

        check_hr_stage(d3d12_device->CreateCommandQueue(&q, __uuidof(ID3D12CommandQueue),
                                                        d3d12_queue.put_void()),
                       "ID3D12Device::CreateCommandQueue");
    }

    void create_learning_device() {
        auto factory =
            winrt::get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();

        winrt::com_ptr<::IUnknown> unknown;
        check_hr_stage(factory->CreateFromD3D12CommandQueue(d3d12_queue.get(), unknown.put()),
                       "ILearningModelDeviceFactoryNative::CreateFromD3D12CommandQueue");

        unknown.try_as(learning_device);
        if (!learning_device) {
            throw std::runtime_error("Gagal membuat LearningModelDevice untuk Liveness");
        }
    }

    void create_interop_device() {
        IUnknown* queues[] = {d3d12_queue.get()};

        check_hr_stage(D3D11On12CreateDevice(d3d12_device.get(), 0, nullptr, 0, queues, 1, 0,
                                             interop_device.put(), interop_context.put(), nullptr),
                       "D3D11On12CreateDevice");

        interop_11on12 = interop_device.try_as<ID3D11On12Device>();
        interop_device3 = interop_device.try_as<ID3D11Device3>();

        if (!interop_11on12 || !interop_device3) {
            throw std::runtime_error(
                "D3D11On12 Liveness tidak menyediakan interface yang dibutuhkan");
        }
    }

    void create_shared_nv12(GpuFrame const& frame) {
        D3D11_TEXTURE2D_DESC src{};
        frame.texture->GetDesc(&src);

        if (src.Format != DXGI_FORMAT_NV12) {
            throw std::runtime_error("Camera texture Liveness bukan NV12");
        }

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = src.Width;
        desc.Height = src.Height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_NV12;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

        check_hr_stage(camera_device->CreateTexture2D(&desc, nullptr, shared_nv12_camera.put()),
                       "ID3D11Device::CreateTexture2D(shared NV12)");

        auto dxgi_resource = shared_nv12_camera.as<IDXGIResource>();
        check_hr_stage(dxgi_resource->GetSharedHandle(&shared_nv12_handle),
                       "IDXGIResource::GetSharedHandle(NV12)");

        if (shared_nv12_handle == nullptr || shared_nv12_handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Liveness shared NV12 handle tidak valid");
        }

        check_hr_stage(interop_device->OpenSharedResource(shared_nv12_handle,
                                                          __uuidof(ID3D11Texture2D),
                                                          shared_nv12_interop.put_void()),
                       "ID3D11Device::OpenSharedResource(NV12)");

        D3D11_SHADER_RESOURCE_VIEW_DESC1 y_desc{};
        y_desc.Format = DXGI_FORMAT_R8_UNORM;
        y_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        y_desc.Texture2D.MostDetailedMip = 0;
        y_desc.Texture2D.MipLevels = 1;
        y_desc.Texture2D.PlaneSlice = 0;

        winrt::com_ptr<ID3D11ShaderResourceView1> y1;
        check_hr_stage(interop_device3->CreateShaderResourceView1(shared_nv12_interop.get(),
                                                                  &y_desc, y1.put()),
                       "CreateShaderResourceView1(Y)");
        y_srv = y1.as<ID3D11ShaderResourceView>();

        D3D11_SHADER_RESOURCE_VIEW_DESC1 uv_desc{};
        uv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
        uv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        uv_desc.Texture2D.MostDetailedMip = 0;
        uv_desc.Texture2D.MipLevels = 1;
        uv_desc.Texture2D.PlaneSlice = 1;

        winrt::com_ptr<ID3D11ShaderResourceView1> uv1;
        check_hr_stage(interop_device3->CreateShaderResourceView1(shared_nv12_interop.get(),
                                                                  &uv_desc, uv1.put()),
                       "CreateShaderResourceView1(UV)");
        uv_srv = uv1.as<ID3D11ShaderResourceView>();
    }

    void create_shaders() {
        check_hr_stage(interop_device->CreateComputeShader(g_liveness_tensorize,
                                                           sizeof(g_liveness_tensorize), nullptr,
                                                           tensorize_shader.put()),
                       "D3D11On12::CreateComputeShader(LIVENESS tensorize)");

        check_hr_stage(interop_device->CreateComputeShader(g_liveness_coverage_reduce,
                                                           sizeof(g_liveness_coverage_reduce),
                                                           nullptr, coverage_reduce_shader.put()),
                       "D3D11On12::CreateComputeShader(LIVENESS coverage reduce)");
    }

    void create_coverage_buffers() {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = 104;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        check_hr_stage(interop_device->CreateBuffer(&desc, nullptr, coverage_buffer.put()),
                       "coverage buffer");
        D3D11_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        view.Buffer.NumElements = 26;
        view.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        check_hr_stage(interop_device->CreateUnorderedAccessView(coverage_buffer.get(), &view,
                                                                 coverage_uav.put()),
                       "coverage UAV");
        desc = {};
        desc.ByteWidth = sizeof(float);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        check_hr_stage(interop_device->CreateBuffer(&desc, nullptr, coverage_readback.put()),
                       "coverage readback (4 bytes)");
    }

    void create_sampler() {
        D3D11_SAMPLER_DESC desc{};
        desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        desc.MinLOD = 0.0F;
        desc.MaxLOD = D3D11_FLOAT32_MAX;

        check_hr_stage(interop_device->CreateSamplerState(&desc, linear_sampler.put()),
                       "D3D11On12::CreateSamplerState");
    }

    void create_params_buffer() {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = sizeof(ShaderParams);
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        check_hr_stage(interop_device->CreateBuffer(&desc, nullptr, params_buffer.put()),
                       "D3D11On12::CreateBuffer(dynamic params)");
    }

    void create_tensor_d3d12_first() {
        const std::uint64_t elements = static_cast<std::uint64_t>(config.input_width) *
                                       static_cast<std::uint64_t>(config.input_height) * 3ULL;
        const std::uint64_t bytes = elements * sizeof(float);

        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        check_hr_stage(d3d12_device->CreateCommittedResource(
                           &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                           __uuidof(ID3D12Resource), tensor_buffer_d3d12.put_void()),
                       "CreateCommittedResource(Liveness tensor)");

        auto tensor_factory = winrt::get_activation_factory<TensorFloat, ITensorStaticsNative>();

        std::array<std::int64_t, 4> shape{
            1,
            3,
            static_cast<std::int64_t>(config.input_height),
            static_cast<std::int64_t>(config.input_width),
        };

        winrt::com_ptr<::IUnknown> unknown;
        check_hr_stage(
            tensor_factory->CreateFromD3D12Resource(tensor_buffer_d3d12.get(), shape.data(),
                                                    static_cast<int>(shape.size()), unknown.put()),
            "ITensorStaticsNative::CreateFromD3D12Resource(Liveness)");

        unknown.try_as(tensor);
        if (!tensor) {
            throw std::runtime_error("Gagal membuat Liveness TensorFloat");
        }

        D3D11_RESOURCE_FLAGS flags11{};
        flags11.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        flags11.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;

        check_hr_stage(interop_11on12->CreateWrappedResource(
                           tensor_buffer_d3d12.get(), &flags11, D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_COMMON, __uuidof(ID3D11Buffer),
                           tensor_buffer_wrapped11.put_void()),
                       "CreateWrappedResource(Liveness tensor)");

        D3D11_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = static_cast<UINT>(elements);
        uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;

        check_hr_stage(interop_device->CreateUnorderedAccessView(tensor_buffer_wrapped11.get(),
                                                                 &uav, tensor_uav.put()),
                       "CreateUnorderedAccessView(Liveness tensor)");
    }

    void create_camera_fence() {
        check_hr_stage(camera_device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                                   __uuidof(ID3D11Fence),
                                                   camera_fence11.put_void()),
                       "ID3D11Device5::CreateFence(camera)");

        check_hr_stage(
            camera_fence11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &camera_fence_handle),
            "ID3D11Fence::CreateSharedHandle(camera)");

        check_hr_stage(d3d12_device->OpenSharedHandle(camera_fence_handle, __uuidof(ID3D12Fence),
                                                      camera_fence12.put_void()),
                       "ID3D12Device::OpenSharedHandle(camera fence)");
    }

    void validate_frame(GpuFrame const& frame) {
        if (!frame.valid()) {
            throw std::invalid_argument("GpuFrame Liveness tidak valid");
        }
        if (frame.width != source_width || frame.height != source_height ||
            frame.dxgi_format != DXGI_FORMAT_NV12) {
            throw std::invalid_argument(
                "Format/dimensi camera frame berubah setelah Liveness init");
        }

        ID3D11Device* raw = nullptr;
        frame.texture->GetDevice(&raw);
        if (raw == nullptr) {
            throw std::runtime_error("Camera frame Liveness tidak memiliki device");
        }
        winrt::com_ptr<ID3D11Device> frame_device;
        frame_device.attach(raw);

        if (frame_device.get() != camera_device.get()) {
            throw std::runtime_error(
                "Camera D3D11 device berubah; Liveness preprocessor harus dibuat ulang");
        }
    }

    void prepare_frame(GpuFrame const& frame) {
        validate_frame(frame);

        camera_context->CopyResource(shared_nv12_camera.get(), frame.texture.get());

        const std::uint64_t value = ++camera_fence_value;
        check_hr_stage(camera_context4->Signal(camera_fence11.get(), value),
                       "ID3D11DeviceContext4::Signal(camera copy)");
        camera_context->Flush();

        check_hr_stage(d3d12_queue->Wait(camera_fence12.get(), value),
                       "ID3D12CommandQueue::Wait(camera fence)");

        frame_prepared = true;
    }

    void update_params(arcface::SimilarityTransform2D const& transform) {
        // Bound fixed-point conversion before sending parameters to the GPU.
        for (float v : transform.dst_to_src)
            if (!std::isfinite(v) || std::abs(v) > 1000000.0F)
                throw std::invalid_argument("Invalid liveness transform");
        for (float y : {0.0F, 79.0F})
            for (float x : {0.0F, 79.0F}) {
                auto const& m = transform.dst_to_src;
                if (std::abs(m[0] * x) + std::abs(m[1] * y) + std::abs(m[2]) > 1000000.0F ||
                    std::abs(m[3] * x) + std::abs(m[4] * y) + std::abs(m[5]) > 1000000.0F)
                    throw std::invalid_argument(
                        "Liveness transform exceeds sampling coordinate range");
            }
        ShaderParams params{};
        params.src_width = source_width;
        params.src_height = source_height;
        params.dst_width = config.input_width;
        params.dst_height = config.input_height;
        params.inv00 = transform.dst_to_src[0];
        params.inv01 = transform.dst_to_src[1];
        params.inv02 = transform.dst_to_src[2];
        params.inv10 = transform.dst_to_src[3];
        params.inv11 = transform.dst_to_src[4];
        params.inv12 = transform.dst_to_src[5];

        D3D11_MAPPED_SUBRESOURCE mapped{};
        check_hr_stage(
            interop_context->Map(params_buffer.get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped),
            "ID3D11DeviceContext::Map(Liveness params)");

        std::memcpy(mapped.pData, &params, sizeof(params));
        interop_context->Unmap(params_buffer.get(), 0);
    }

    LivenessPreprocessResult preprocess_aligned(arcface::SimilarityTransform2D const& transform) {
        if (!frame_prepared) {
            throw std::logic_error("LivenessGpuPreprocessor::prepare_frame harus dipanggil dahulu");
        }

        update_params(transform);

        ID3D11Resource* wrapped[] = {tensor_buffer_wrapped11.get()};
        interop_11on12->AcquireWrappedResources(wrapped, 1);

        ID3D11ShaderResourceView* srvs[2] = {y_srv.get(), uv_srv.get()};
        interop_context->CSSetShaderResources(0, 2, srvs);

        ID3D11UnorderedAccessView* uavs[2] = {tensor_uav.get(), coverage_uav.get()};
        UINT initial_counts[2] = {0, 0};
        interop_context->CSSetUnorderedAccessViews(0, 2, uavs, initial_counts);

        ID3D11SamplerState* samplers[1] = {linear_sampler.get()};
        interop_context->CSSetSamplers(0, 1, samplers);

        ID3D11Buffer* constants[1] = {params_buffer.get()};
        interop_context->CSSetConstantBuffers(0, 1, constants);
        interop_context->CSSetShader(tensorize_shader.get(), nullptr, 0);

        const UINT gx = (config.input_width + 15U) / 16U;
        const UINT gy = (config.input_height + 15U) / 16U;
        interop_context->Dispatch(gx, gy, 1);
        interop_context->CSSetShader(coverage_reduce_shader.get(), nullptr, 0);
        interop_context->Dispatch(1, 1, 1);

        ID3D11ShaderResourceView* null_srvs[2] = {nullptr, nullptr};
        ID3D11UnorderedAccessView* null_uavs[2] = {nullptr, nullptr};
        ID3D11SamplerState* null_samplers[1] = {nullptr};
        ID3D11Buffer* null_buffers[1] = {nullptr};

        interop_context->CSSetShaderResources(0, 2, null_srvs);
        interop_context->CSSetUnorderedAccessViews(0, 2, null_uavs, initial_counts);
        interop_context->CSSetSamplers(0, 1, null_samplers);
        interop_context->CSSetConstantBuffers(0, 1, null_buffers);
        interop_context->CSSetShader(nullptr, nullptr, 0);

        interop_11on12->ReleaseWrappedResources(wrapped, 1);
        D3D11_BOX scalar_box{100, 0, 0, 104, 1, 1};
        interop_context->CopySubresourceRegion(coverage_readback.get(), 0, 0, 0, 0,
                                               coverage_buffer.get(), 0, &scalar_box);
        interop_context->Flush();
        D3D11_MAPPED_SUBRESOURCE mapped{};
        check_hr_stage(interop_context->Map(coverage_readback.get(), 0, D3D11_MAP_READ, 0, &mapped),
                       "read coverage scalar");
        float ratio = 0;
        std::memcpy(&ratio, mapped.pData, sizeof(ratio));
        interop_context->Unmap(coverage_readback.get(), 0);
        if (!std::isfinite(ratio) || ratio < 0 || ratio > 1)
            throw std::runtime_error("Invalid GPU liveness coverage");
        return {tensor, ratio};
    }
};

LivenessGpuPreprocessor::LivenessGpuPreprocessor(GpuFrame const& first_frame,
                                                 LivenessPreprocessConfig config)
    : impl_(std::make_unique<Impl>(first_frame, config)) {}

LivenessGpuPreprocessor::~LivenessGpuPreprocessor() = default;

void LivenessGpuPreprocessor::prepare_frame(GpuFrame const& frame) {
    impl_->prepare_frame(frame);
}

LivenessPreprocessResult
LivenessGpuPreprocessor::preprocess_aligned(arcface::SimilarityTransform2D const& transform) {
    return impl_->preprocess_aligned(transform);
}

LearningModelDevice LivenessGpuPreprocessor::learning_model_device() const {
    return impl_->learning_device;
}

LivenessPreprocessConfig LivenessGpuPreprocessor::config() const noexcept {
    return impl_->config;
}

} // namespace vision_runtime::liveness
