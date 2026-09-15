#include "arcface/gpu_preprocessor.hpp"

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
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <windows.ai.machinelearning.native.h>
#include <winrt/Windows.Foundation.Collections.h>

namespace vision_runtime::arcface {
namespace {

using namespace winrt::Windows::AI::MachineLearning;

constexpr char kArcFaceTensorizeShader[] = R"HLSL(
Texture2D<float>  InputY  : register(t0);
Texture2D<float2> InputUV : register(t1);
SamplerState LinearClamp  : register(s0);
RWByteAddressBuffer OutputTensor : register(u0);

cbuffer Params : register(b0)
{
    uint SrcWidth;
    uint SrcHeight;
    uint DstWidth;
    uint DstHeight;

    float Inv00;
    float Inv01;
    float Inv02;
    float _Pad0;

    float Inv10;
    float Inv11;
    float Inv12;
    float _Pad1;
};

// Same validated camera color path used by the native SCRFD/PAD runtime.
// The current 1280x720 camera surface is treated as limited-range BT.709 NV12.
float3 Nv12Bt709LimitedToRgb(float ySample, float2 uvSample)
{
    float y  = (ySample * 255.0f - 16.0f) / 219.0f;
    float cb = (uvSample.x * 255.0f - 128.0f) / 224.0f;
    float cr = (uvSample.y * 255.0f - 128.0f) / 224.0f;

    float3 rgb;
    rgb.r = y + 1.5748f * cr;
    rgb.g = y - 0.187324f * cb - 0.468124f * cr;
    rgb.b = y + 1.8556f * cb;
    return saturate(rgb);
}

float3 SampleRgbPointBorder(int sx, int sy)
{
    if (sx < 0 || sy < 0 || sx >= (int)SrcWidth || sy >= (int)SrcHeight)
        return float3(0.0f, 0.0f, 0.0f);

    const float y = InputY.Load(int3(sx, sy, 0));
    const float2 uvCoord = float2(
        ((float)sx + 0.5f) / (float)SrcWidth,
        ((float)sy + 0.5f) / (float)SrcHeight
    );
    const float2 uv = InputUV.SampleLevel(LinearClamp, uvCoord, 0.0f);
    return Nv12Bt709LimitedToRgb(y, uv);
}

// cv::warpAffine uses INTER_LINEAR by default in InsightFace face_align.norm_crop.
// Perform bilinear interpolation after applying the inverse similarity map.
float3 SampleRgbLinear(float sx, float sy)
{
    const int x0 = (int)floor(sx);
    const int y0 = (int)floor(sy);
    const float fx = sx - (float)x0;
    const float fy = sy - (float)y0;

    const float3 p00 = SampleRgbPointBorder(x0,     y0);
    const float3 p10 = SampleRgbPointBorder(x0 + 1, y0);
    const float3 p01 = SampleRgbPointBorder(x0,     y0 + 1);
    const float3 p11 = SampleRgbPointBorder(x0 + 1, y0 + 1);

    const float3 top = lerp(p00, p10, fx);
    const float3 bot = lerp(p01, p11, fx);
    return lerp(top, bot, fy);
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= DstWidth || tid.y >= DstHeight)
        return;

    // OpenCV warpAffine evaluates the inverse mapping at integer destination
    // pixel coordinates. Inv maps aligned ArcFace output -> camera source.
    const float u = (float)tid.x;
    const float v = (float)tid.y;
    const float sx = Inv00 * u + Inv01 * v + Inv02;
    const float sy = Inv10 * u + Inv11 * v + Inv12;

    const float3 rgb01 = SampleRgbLinear(sx, sy);
    const float3 pixel255 = rgb01 * 255.0f;

    // InsightFace w600k_r50 preprocessing for the buffalo_l recognition model.
    const float3 normalized = (pixel255 - 127.5f) / 127.5f;

    const uint plane = DstWidth * DstHeight;
    const uint pixel = tid.y * DstWidth + tid.x;

    // NCHW RGB. Official Python uses cv2.dnn.blobFromImages(... swapRB=True).
    OutputTensor.Store((pixel) * 4, asuint(normalized.r));
    OutputTensor.Store((plane + pixel) * 4, asuint(normalized.g));
    OutputTensor.Store((2 * plane + pixel) * 4, asuint(normalized.b));
}
)HLSL";

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

std::string blob_error(ID3DBlob* blob) {
    if (blob == nullptr || blob->GetBufferPointer() == nullptr) {
        return {};
    }
    return std::string(
        static_cast<char const*>(blob->GetBufferPointer()),
        blob->GetBufferSize()
    );
}

void check_hr_stage(HRESULT hr, char const* stage) {
    if (SUCCEEDED(hr)) return;

    std::ostringstream message;
    message
        << "ArcFace GPU stage '" << stage << "' gagal: HRESULT 0x"
        << std::hex << std::uppercase
        << static_cast<std::uint32_t>(hr);
    throw std::runtime_error(message.str());
}

} // namespace

struct ArcFaceGpuPreprocessor::Impl {
    explicit Impl(GpuFrame const& first_frame, ArcFacePreprocessConfig cfg)
        : config(cfg) {
        initialize(first_frame);
    }

    ~Impl() {
        if (interop_context) {
            interop_context->ClearState();
            interop_context->Flush();
        }

        // Legacy IDXGIResource::GetSharedHandle result is not an NT handle.
        shared_nv12_handle = nullptr;
        close_handle(camera_fence_handle);
    }

    ArcFacePreprocessConfig config{};

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
            throw std::invalid_argument("GpuFrame ArcFace tidak valid");
        }
        if (frame.dxgi_format != DXGI_FORMAT_NV12) {
            throw std::invalid_argument(
                "ArcFace GPU preprocessor membutuhkan DXGI_FORMAT_NV12"
            );
        }
        if (config.input_width != 112 || config.input_height != 112) {
            throw std::invalid_argument(
                "InsightFace w600k_r50 input dikunci ke 112x112"
            );
        }

        source_width = frame.width;
        source_height = frame.height;

        ID3D11Device* raw_device = nullptr;
        frame.texture->GetDevice(&raw_device);
        if (raw_device == nullptr) {
            throw std::runtime_error(
                "Gagal memperoleh D3D11 device dari camera texture"
            );
        }
        camera_device.attach(raw_device);

        camera_device5 = camera_device.try_as<ID3D11Device5>();
        if (!camera_device5) {
            throw std::runtime_error(
                "ID3D11Device5 tidak tersedia untuk ArcFace cross-API fence"
            );
        }

        camera_device->GetImmediateContext(camera_context.put());
        camera_context4 = camera_context.try_as<ID3D11DeviceContext4>();
        if (!camera_context4) {
            throw std::runtime_error(
                "ID3D11DeviceContext4 tidak tersedia untuk ArcFace"
            );
        }

        create_d3d12_runtime();
        create_learning_device();
        create_interop_device();
        create_shared_nv12(frame);
        compile_shader();
        create_sampler();
        create_params_buffer();
        create_tensor_d3d12_first();
        create_camera_fence();
    }

    void create_d3d12_runtime() {
        auto dxgi_device = camera_device.as<IDXGIDevice>();
        winrt::com_ptr<IDXGIAdapter> adapter;
        check_hr_stage(
            dxgi_device->GetAdapter(adapter.put()),
            "IDXGIDevice::GetAdapter"
        );

        check_hr_stage(
            D3D12CreateDevice(
                adapter.get(),
                D3D_FEATURE_LEVEL_11_0,
                __uuidof(ID3D12Device),
                d3d12_device.put_void()
            ),
            "D3D12CreateDevice"
        );

        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        q.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        q.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

        check_hr_stage(
            d3d12_device->CreateCommandQueue(
                &q,
                __uuidof(ID3D12CommandQueue),
                d3d12_queue.put_void()
            ),
            "ID3D12Device::CreateCommandQueue"
        );
    }

    void create_learning_device() {
        auto factory = winrt::get_activation_factory<
            LearningModelDevice,
            ILearningModelDeviceFactoryNative
        >();

        winrt::com_ptr<::IUnknown> unknown;
        check_hr_stage(
            factory->CreateFromD3D12CommandQueue(
                d3d12_queue.get(),
                unknown.put()
            ),
            "ILearningModelDeviceFactoryNative::CreateFromD3D12CommandQueue"
        );

        unknown.try_as(learning_device);
        if (!learning_device) {
            throw std::runtime_error(
                "Gagal membuat LearningModelDevice untuk ArcFace"
            );
        }
    }

    void create_interop_device() {
        IUnknown* queues[] = {d3d12_queue.get()};

        check_hr_stage(
            D3D11On12CreateDevice(
                d3d12_device.get(),
                0,
                nullptr,
                0,
                queues,
                1,
                0,
                interop_device.put(),
                interop_context.put(),
                nullptr
            ),
            "D3D11On12CreateDevice"
        );

        interop_11on12 = interop_device.try_as<ID3D11On12Device>();
        interop_device3 = interop_device.try_as<ID3D11Device3>();

        if (!interop_11on12 || !interop_device3) {
            throw std::runtime_error(
                "D3D11On12 ArcFace tidak menyediakan interface yang dibutuhkan"
            );
        }
    }

    void create_shared_nv12(GpuFrame const& frame) {
        D3D11_TEXTURE2D_DESC src{};
        frame.texture->GetDesc(&src);

        if (src.Format != DXGI_FORMAT_NV12) {
            throw std::runtime_error("Camera texture ArcFace bukan NV12");
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

        check_hr_stage(
            camera_device->CreateTexture2D(
                &desc,
                nullptr,
                shared_nv12_camera.put()
            ),
            "ID3D11Device::CreateTexture2D(shared NV12)"
        );

        auto dxgi_resource = shared_nv12_camera.as<IDXGIResource>();
        check_hr_stage(
            dxgi_resource->GetSharedHandle(&shared_nv12_handle),
            "IDXGIResource::GetSharedHandle(NV12)"
        );

        if (
            shared_nv12_handle == nullptr ||
            shared_nv12_handle == INVALID_HANDLE_VALUE
        ) {
            throw std::runtime_error(
                "ArcFace shared NV12 handle tidak valid"
            );
        }

        check_hr_stage(
            interop_device->OpenSharedResource(
                shared_nv12_handle,
                __uuidof(ID3D11Texture2D),
                shared_nv12_interop.put_void()
            ),
            "ID3D11Device::OpenSharedResource(NV12)"
        );

        D3D11_SHADER_RESOURCE_VIEW_DESC1 y_desc{};
        y_desc.Format = DXGI_FORMAT_R8_UNORM;
        y_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        y_desc.Texture2D.MostDetailedMip = 0;
        y_desc.Texture2D.MipLevels = 1;
        y_desc.Texture2D.PlaneSlice = 0;

        winrt::com_ptr<ID3D11ShaderResourceView1> y1;
        check_hr_stage(
            interop_device3->CreateShaderResourceView1(
                shared_nv12_interop.get(),
                &y_desc,
                y1.put()
            ),
            "CreateShaderResourceView1(Y)"
        );
        y_srv = y1.as<ID3D11ShaderResourceView>();

        D3D11_SHADER_RESOURCE_VIEW_DESC1 uv_desc{};
        uv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
        uv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        uv_desc.Texture2D.MostDetailedMip = 0;
        uv_desc.Texture2D.MipLevels = 1;
        uv_desc.Texture2D.PlaneSlice = 1;

        winrt::com_ptr<ID3D11ShaderResourceView1> uv1;
        check_hr_stage(
            interop_device3->CreateShaderResourceView1(
                shared_nv12_interop.get(),
                &uv_desc,
                uv1.put()
            ),
            "CreateShaderResourceView1(UV)"
        );
        uv_srv = uv1.as<ID3D11ShaderResourceView>();
    }

    void compile_shader() {
        winrt::com_ptr<ID3DBlob> shader_blob;
        winrt::com_ptr<ID3DBlob> error_blob;

        UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
        flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
        flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

        const HRESULT hr = D3DCompile(
            kArcFaceTensorizeShader,
            std::strlen(kArcFaceTensorizeShader),
            "arcface_align_tensorize.hlsl",
            nullptr,
            nullptr,
            "main",
            "cs_5_0",
            flags,
            0,
            shader_blob.put(),
            error_blob.put()
        );

        if (FAILED(hr)) {
            throw std::runtime_error(
                "Gagal compile ArcFace GPU shader: " +
                blob_error(error_blob.get())
            );
        }

        check_hr_stage(
            interop_device->CreateComputeShader(
                shader_blob->GetBufferPointer(),
                shader_blob->GetBufferSize(),
                nullptr,
                tensorize_shader.put()
            ),
            "D3D11On12::CreateComputeShader"
        );
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

        check_hr_stage(
            interop_device->CreateSamplerState(
                &desc,
                linear_sampler.put()
            ),
            "D3D11On12::CreateSamplerState"
        );
    }

    void create_params_buffer() {
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = sizeof(ShaderParams);
        desc.Usage = D3D11_USAGE_DYNAMIC;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        check_hr_stage(
            interop_device->CreateBuffer(
                &desc,
                nullptr,
                params_buffer.put()
            ),
            "D3D11On12::CreateBuffer(dynamic params)"
        );
    }

    void create_tensor_d3d12_first() {
        const std::uint64_t elements =
            static_cast<std::uint64_t>(config.input_width) *
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

        check_hr_stage(
            d3d12_device->CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                __uuidof(ID3D12Resource),
                tensor_buffer_d3d12.put_void()
            ),
            "CreateCommittedResource(ArcFace tensor)"
        );

        auto tensor_factory = winrt::get_activation_factory<
            TensorFloat,
            ITensorStaticsNative
        >();

        std::array<std::int64_t, 4> shape{
            1,
            3,
            static_cast<std::int64_t>(config.input_height),
            static_cast<std::int64_t>(config.input_width),
        };

        winrt::com_ptr<::IUnknown> unknown;
        check_hr_stage(
            tensor_factory->CreateFromD3D12Resource(
                tensor_buffer_d3d12.get(),
                shape.data(),
                static_cast<int>(shape.size()),
                unknown.put()
            ),
            "ITensorStaticsNative::CreateFromD3D12Resource(ArcFace)"
        );

        unknown.try_as(tensor);
        if (!tensor) {
            throw std::runtime_error("Gagal membuat ArcFace TensorFloat");
        }

        D3D11_RESOURCE_FLAGS flags11{};
        flags11.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        flags11.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;

        check_hr_stage(
            interop_11on12->CreateWrappedResource(
                tensor_buffer_d3d12.get(),
                &flags11,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COMMON,
                __uuidof(ID3D11Buffer),
                tensor_buffer_wrapped11.put_void()
            ),
            "CreateWrappedResource(ArcFace tensor)"
        );

        D3D11_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = static_cast<UINT>(elements);
        uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;

        check_hr_stage(
            interop_device->CreateUnorderedAccessView(
                tensor_buffer_wrapped11.get(),
                &uav,
                tensor_uav.put()
            ),
            "CreateUnorderedAccessView(ArcFace tensor)"
        );
    }

    void create_camera_fence() {
        check_hr_stage(
            camera_device5->CreateFence(
                0,
                D3D11_FENCE_FLAG_SHARED,
                __uuidof(ID3D11Fence),
                camera_fence11.put_void()
            ),
            "ID3D11Device5::CreateFence(camera)"
        );

        check_hr_stage(
            camera_fence11->CreateSharedHandle(
                nullptr,
                GENERIC_ALL,
                nullptr,
                &camera_fence_handle
            ),
            "ID3D11Fence::CreateSharedHandle(camera)"
        );

        check_hr_stage(
            d3d12_device->OpenSharedHandle(
                camera_fence_handle,
                __uuidof(ID3D12Fence),
                camera_fence12.put_void()
            ),
            "ID3D12Device::OpenSharedHandle(camera fence)"
        );
    }

    void validate_frame(GpuFrame const& frame) {
        if (!frame.valid()) {
            throw std::invalid_argument("GpuFrame ArcFace tidak valid");
        }
        if (
            frame.width != source_width ||
            frame.height != source_height ||
            frame.dxgi_format != DXGI_FORMAT_NV12
        ) {
            throw std::invalid_argument(
                "Format/dimensi camera frame berubah setelah ArcFace init"
            );
        }

        ID3D11Device* raw = nullptr;
        frame.texture->GetDevice(&raw);
        if (raw == nullptr) {
            throw std::runtime_error("Camera frame ArcFace tidak memiliki device");
        }
        winrt::com_ptr<ID3D11Device> frame_device;
        frame_device.attach(raw);

        if (frame_device.get() != camera_device.get()) {
            throw std::runtime_error(
                "Camera D3D11 device berubah; ArcFace preprocessor harus dibuat ulang"
            );
        }
    }

    void prepare_frame(GpuFrame const& frame) {
        validate_frame(frame);

        camera_context->CopyResource(
            shared_nv12_camera.get(),
            frame.texture.get()
        );

        const std::uint64_t value = ++camera_fence_value;
        check_hr_stage(
            camera_context4->Signal(camera_fence11.get(), value),
            "ID3D11DeviceContext4::Signal(camera copy)"
        );
        camera_context->Flush();

        check_hr_stage(
            d3d12_queue->Wait(camera_fence12.get(), value),
            "ID3D12CommandQueue::Wait(camera fence)"
        );

        frame_prepared = true;
    }

    void update_params(SimilarityTransform2D const& transform) {
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
            interop_context->Map(
                params_buffer.get(),
                0,
                D3D11_MAP_WRITE_DISCARD,
                0,
                &mapped
            ),
            "ID3D11DeviceContext::Map(ArcFace params)"
        );

        std::memcpy(mapped.pData, &params, sizeof(params));
        interop_context->Unmap(params_buffer.get(), 0);
    }

    TensorFloat preprocess_aligned(SimilarityTransform2D const& transform) {
        if (!frame_prepared) {
            throw std::logic_error(
                "ArcFaceGpuPreprocessor::prepare_frame harus dipanggil dahulu"
            );
        }

        update_params(transform);

        ID3D11Resource* wrapped[] = {
            tensor_buffer_wrapped11.get()
        };
        interop_11on12->AcquireWrappedResources(wrapped, 1);

        ID3D11ShaderResourceView* srvs[2] = {
            y_srv.get(),
            uv_srv.get()
        };
        interop_context->CSSetShaderResources(0, 2, srvs);

        ID3D11UnorderedAccessView* uavs[1] = {
            tensor_uav.get()
        };
        UINT initial_counts[1] = {0};
        interop_context->CSSetUnorderedAccessViews(
            0, 1, uavs, initial_counts
        );

        ID3D11SamplerState* samplers[1] = {
            linear_sampler.get()
        };
        interop_context->CSSetSamplers(0, 1, samplers);

        ID3D11Buffer* constants[1] = {
            params_buffer.get()
        };
        interop_context->CSSetConstantBuffers(0, 1, constants);
        interop_context->CSSetShader(tensorize_shader.get(), nullptr, 0);

        const UINT gx = (config.input_width + 15U) / 16U;
        const UINT gy = (config.input_height + 15U) / 16U;
        interop_context->Dispatch(gx, gy, 1);

        ID3D11ShaderResourceView* null_srvs[2] = {nullptr, nullptr};
        ID3D11UnorderedAccessView* null_uavs[1] = {nullptr};
        ID3D11SamplerState* null_samplers[1] = {nullptr};
        ID3D11Buffer* null_buffers[1] = {nullptr};

        interop_context->CSSetShaderResources(0, 2, null_srvs);
        interop_context->CSSetUnorderedAccessViews(
            0, 1, null_uavs, initial_counts
        );
        interop_context->CSSetSamplers(0, 1, null_samplers);
        interop_context->CSSetConstantBuffers(0, 1, null_buffers);
        interop_context->CSSetShader(nullptr, nullptr, 0);

        interop_11on12->ReleaseWrappedResources(wrapped, 1);
        interop_context->Flush();

        return tensor;
    }
};

ArcFaceGpuPreprocessor::ArcFaceGpuPreprocessor(
    GpuFrame const& first_frame,
    ArcFacePreprocessConfig config
) : impl_(std::make_unique<Impl>(first_frame, config)) {}

ArcFaceGpuPreprocessor::~ArcFaceGpuPreprocessor() = default;

void ArcFaceGpuPreprocessor::prepare_frame(GpuFrame const& frame) {
    impl_->prepare_frame(frame);
}

TensorFloat ArcFaceGpuPreprocessor::preprocess_aligned(
    SimilarityTransform2D const& transform
) {
    return impl_->preprocess_aligned(transform);
}

LearningModelDevice ArcFaceGpuPreprocessor::learning_model_device() const {
    return impl_->learning_device;
}

ArcFacePreprocessConfig ArcFaceGpuPreprocessor::config() const noexcept {
    return impl_->config;
}

} // namespace vision_runtime::arcface
