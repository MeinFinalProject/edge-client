#include "scrfd/gpu_preprocessor.hpp"

#include "scrfd/detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
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

namespace vision_runtime {
namespace {

using namespace winrt::Windows::AI::MachineLearning;

constexpr char kTensorizeShader[] = R"HLSL(
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

    float Scale;
    float PadX;
    float PadY;
    float Reserved;
};

float3 Nv12Bt709LimitedToRgb(float ySample, float2 uvSample)
{
    // UVC 720p NV12 umumnya memakai studio/limited range dan matriks BT.709.
    // Nilai dikembalikan ke domain nominal 8-bit sebelum matrix YUV->RGB.
    float y  = (ySample  * 255.0f - 16.0f)  / 219.0f;
    float cb = (uvSample.x * 255.0f - 128.0f) / 224.0f;
    float cr = (uvSample.y * 255.0f - 128.0f) / 224.0f;

    float3 rgb;
    rgb.r = y + 1.5748f * cr;
    rgb.g = y - 0.187324f * cb - 0.468124f * cr;
    rgb.b = y + 1.8556f * cb;

    return saturate(rgb);
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= DstWidth || tid.y >= DstHeight)
        return;

    const uint plane = DstWidth * DstHeight;
    const uint pixelIndex = tid.y * DstWidth + tid.x;

    // InsightFace/SCRFD membuat letterbox hitam (pixel 0), lalu:
    // blob = (pixel - 127.5) / 128.
    const float normalizedBlack = -127.5f / 128.0f;

    float3 normalized = float3(
        normalizedBlack,
        normalizedBlack,
        normalizedBlack
    );

    const float x = (float)tid.x + 0.5f;
    const float y = (float)tid.y + 0.5f;

    const float contentWidth  = (float)SrcWidth  * Scale;
    const float contentHeight = (float)SrcHeight * Scale;

    if (
        x >= PadX && x < (PadX + contentWidth) &&
        y >= PadY && y < (PadY + contentHeight)
    ) {
        const float srcX = (x - PadX) / Scale;
        const float srcY = (y - PadY) / Scale;

        const float2 uv = float2(
            srcX / (float)SrcWidth,
            srcY / (float)SrcHeight
        );

        const float ySample = InputY.SampleLevel(LinearClamp, uv, 0.0f);
        const float2 uvSample = InputUV.SampleLevel(LinearClamp, uv, 0.0f);

        const float3 rgb = Nv12Bt709LimitedToRgb(ySample, uvSample);
        normalized = (rgb * 255.0f - 127.5f) / 128.0f;
    }

    // Original SCRFD input contract: NCHW RGB float32.
    // RWByteAddressBuffer dipakai agar D3D11On12 dapat menulis langsung ke
    // backing buffer D3D12 yang juga dipakai TensorFloat WinML.
    OutputTensor.Store((pixelIndex) * 4, asuint(normalized.r));
    OutputTensor.Store((plane + pixelIndex) * 4, asuint(normalized.g));
    OutputTensor.Store((2 * plane + pixelIndex) * 4, asuint(normalized.b));
}
)HLSL";

struct alignas(16) ShaderParams {
    std::uint32_t src_width;
    std::uint32_t src_height;
    std::uint32_t dst_width;
    std::uint32_t dst_height;
    float scale;
    float pad_x;
    float pad_y;
    float reserved;
};

static_assert(sizeof(ShaderParams) == 32);

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
    if (SUCCEEDED(hr)) {
        return;
    }

    std::ostringstream message;
    message
        << "SCRFD GPU stage '" << stage << "' gagal: HRESULT 0x"
        << std::hex << std::uppercase
        << static_cast<std::uint32_t>(hr);
    throw std::runtime_error(message.str());
}

} // namespace

struct ScrfdGpuPreprocessor::Impl {
    explicit Impl(
        GpuFrame const& first_frame,
        ScrfdPreprocessConfig cfg
    ) : config(cfg) {
        initialize(first_frame);
    }

    ~Impl() {
        if (interop_context) {
            interop_context->ClearState();
            interop_context->Flush();
        }
        // shared_nv12_handle berasal dari IDXGIResource::GetSharedHandle().
        // Ini BUKAN NT HANDLE dan tidak boleh di-CloseHandle(). Validitasnya
        // mengikuti lifetime shared_nv12_camera/shared_nv12_interop.
        shared_nv12_handle = nullptr;
        close_handle(camera_fence_handle);
    }

    ScrfdPreprocessConfig config{};
    ScrfdPreprocessTransform transform{};

    // Camera-owned D3D11 device. Dipakai hanya untuk copy frame terbaru ke
    // texture NV12 shareable dan signal fence setelah copy selesai.
    winrt::com_ptr<ID3D11Device> camera_device;
    winrt::com_ptr<ID3D11Device5> camera_device5;
    winrt::com_ptr<ID3D11DeviceContext> camera_context;
    winrt::com_ptr<ID3D11DeviceContext4> camera_context4;

    // D3D12 device/queue yang dipakai bersama oleh D3D11On12 preprocessor
    // dan Windows ML. Tensor backing resource dibuat native D3D12 dari awal.
    winrt::com_ptr<ID3D12Device> d3d12_device;
    winrt::com_ptr<ID3D12CommandQueue> d3d12_queue;

    // D3D11On12 layer memungkinkan compute shader D3D11 menulis langsung
    // ke D3D12 tensor buffer tanpa D3D11 shared-buffer (yang tidak valid).
    winrt::com_ptr<ID3D11Device> interop_device;
    winrt::com_ptr<ID3D11Device1> interop_device1;
    winrt::com_ptr<ID3D11Device3> interop_device3;
    winrt::com_ptr<ID3D11DeviceContext> interop_context;
    winrt::com_ptr<ID3D11On12Device> interop_11on12;

    // Shared NV12 texture. v3.4 sengaja memakai legacy DXGI shared handle
    // (D3D11_RESOURCE_MISC_SHARED + IDXGIResource::GetSharedHandle) karena
    // driver kamera/iGPU ini menolak NV12 + SHARED_NTHANDLE dengan E_INVALIDARG.
    // Handle legacy bukan NT HANDLE: jangan panggil CloseHandle().
    winrt::com_ptr<ID3D11Texture2D> shared_nv12_camera;
    HANDLE shared_nv12_handle = nullptr;
    winrt::com_ptr<ID3D11Texture2D> shared_nv12_interop;
    winrt::com_ptr<ID3D11ShaderResourceView> y_srv;
    winrt::com_ptr<ID3D11ShaderResourceView> uv_srv;

    winrt::com_ptr<ID3D11ComputeShader> tensorize_shader;
    winrt::com_ptr<ID3D11SamplerState> linear_sampler;
    winrt::com_ptr<ID3D11Buffer> params_buffer;

    // D3D12-first tensor resource + D3D11On12 wrapped view.
    winrt::com_ptr<ID3D12Resource> tensor_buffer_d3d12;
    winrt::com_ptr<ID3D11Buffer> tensor_buffer_wrapped11;
    winrt::com_ptr<ID3D11UnorderedAccessView> tensor_uav;

    // Sinkronisasi camera D3D11 -> D3D12 queue. Tidak ada CPU wait.
    winrt::com_ptr<ID3D11Fence> camera_fence11;
    HANDLE camera_fence_handle = nullptr;
    winrt::com_ptr<ID3D12Fence> camera_fence12;
    std::uint64_t camera_fence_value = 0;

    LearningModelDevice learning_device{nullptr};
    TensorFloat tensor{nullptr};

    void initialize(GpuFrame const& frame) {
        validate_gpu_frame(frame);

        if (frame.dxgi_format != DXGI_FORMAT_NV12) {
            throw std::invalid_argument(
                "SCRFD GPU preprocessor saat ini membutuhkan DXGI_FORMAT_NV12"
            );
        }

        if (config.input_width == 0 || config.input_height == 0) {
            throw std::invalid_argument(
                "Ukuran input SCRFD tidak boleh nol"
            );
        }

        if (config.input_width != 640 || config.input_height != 640) {
            throw std::invalid_argument(
                "Baseline SCRFD dikunci ke input original 640x640"
            );
        }

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
                "ID3D11Device5 tidak tersedia; cross-API fence diperlukan"
            );
        }

        camera_device->GetImmediateContext(camera_context.put());
        camera_context4 = camera_context.try_as<ID3D11DeviceContext4>();
        if (!camera_context4) {
            throw std::runtime_error(
                "ID3D11DeviceContext4 tidak tersedia; cross-API fence diperlukan"
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

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queue_desc.NodeMask = 0;

        check_hr_stage(
            d3d12_device->CreateCommandQueue(
                &queue_desc,
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
                "Gagal membuat LearningModelDevice dari D3D12 command queue"
            );
        }
    }

    void create_interop_device() {
        IUnknown* queues[] = {
            d3d12_queue.get()
        };

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
        interop_device1 = interop_device.try_as<ID3D11Device1>();
        interop_device3 = interop_device.try_as<ID3D11Device3>();

        if (!interop_11on12 || !interop_device1 || !interop_device3) {
            throw std::runtime_error(
                "D3D11On12 device tidak menyediakan interface interop yang dibutuhkan"
            );
        }
    }

    void create_shared_nv12(GpuFrame const& frame) {
        D3D11_TEXTURE2D_DESC source_desc{};
        frame.texture->GetDesc(&source_desc);

        if (source_desc.Format != DXGI_FORMAT_NV12) {
            throw std::runtime_error(
                "Camera texture bukan NV12"
            );
        }

        // Diagnostic capability only. ExtendedNV12SharedTextureSupported
        // memberi tahu apakah driver secara eksplisit melaporkan dukungan
        // extended NV12 sharing. Kita tetap mencoba legacy SHARED karena
        // itulah jalur DXGI yang didukung OpenSharedResource().
        D3D11_FEATURE_DATA_D3D11_OPTIONS4 options4{};
        const HRESULT options4_hr = camera_device->CheckFeatureSupport(
            D3D11_FEATURE_D3D11_OPTIONS4,
            &options4,
            sizeof(options4)
        );
        const bool extended_nv12_share =
            SUCCEEDED(options4_hr) &&
            options4.ExtendedNV12SharedTextureSupported != FALSE;

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = source_desc.Width;
        desc.Height = source_desc.Height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_NV12;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;

        // v3.3 memakai SHARED_NTHANDLE dan ditolak AMD driver saat
        // CreateTexture2D(). Untuk NV12 kita gunakan jalur legacy SHARED:
        // CreateTexture2D -> IDXGIResource::GetSharedHandle -> OpenSharedResource.
        // Tidak ada CPU pixel copy; frame tetap GPU resource.
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

        const HRESULT create_hr = camera_device->CreateTexture2D(
            &desc,
            nullptr,
            shared_nv12_camera.put()
        );
        if (FAILED(create_hr)) {
            std::ostringstream message;
            message
                << "SCRFD GPU stage 'ID3D11Device::CreateTexture2D(legacy-shared NV12)' gagal: HRESULT 0x"
                << std::hex << std::uppercase
                << static_cast<std::uint32_t>(create_hr)
                << "; ExtendedNV12SharedTextureSupported="
                << (extended_nv12_share ? "TRUE" : "FALSE/UNAVAILABLE");
            throw std::runtime_error(message.str());
        }

        auto dxgi_resource = shared_nv12_camera.as<IDXGIResource>();
        check_hr_stage(
            dxgi_resource->GetSharedHandle(
                &shared_nv12_handle
            ),
            "IDXGIResource::GetSharedHandle(NV12 texture)"
        );

        if (
            shared_nv12_handle == nullptr ||
            shared_nv12_handle == INVALID_HANDLE_VALUE
        ) {
            throw std::runtime_error(
                "IDXGIResource::GetSharedHandle menghasilkan handle NV12 yang tidak valid"
            );
        }

        check_hr_stage(
            interop_device->OpenSharedResource(
                shared_nv12_handle,
                __uuidof(ID3D11Texture2D),
                shared_nv12_interop.put_void()
            ),
            "ID3D11Device::OpenSharedResource(NV12 texture)"
        );

        D3D11_SHADER_RESOURCE_VIEW_DESC1 y_desc{};
        y_desc.Format = DXGI_FORMAT_R8_UNORM;
        y_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        y_desc.Texture2D.MostDetailedMip = 0;
        y_desc.Texture2D.MipLevels = 1;
        y_desc.Texture2D.PlaneSlice = 0;

        winrt::com_ptr<ID3D11ShaderResourceView1> y_srv1;
        check_hr_stage(
            interop_device3->CreateShaderResourceView1(
                shared_nv12_interop.get(),
                &y_desc,
                y_srv1.put()
            ),
            "ID3D11Device3::CreateShaderResourceView1(Y plane)"
        );
        y_srv = y_srv1.as<ID3D11ShaderResourceView>();

        D3D11_SHADER_RESOURCE_VIEW_DESC1 uv_desc{};
        uv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
        uv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        uv_desc.Texture2D.MostDetailedMip = 0;
        uv_desc.Texture2D.MipLevels = 1;
        uv_desc.Texture2D.PlaneSlice = 1;

        winrt::com_ptr<ID3D11ShaderResourceView1> uv_srv1;
        check_hr_stage(
            interop_device3->CreateShaderResourceView1(
                shared_nv12_interop.get(),
                &uv_desc,
                uv_srv1.put()
            ),
            "ID3D11Device3::CreateShaderResourceView1(UV plane)"
        );
        uv_srv = uv_srv1.as<ID3D11ShaderResourceView>();

        transform.source_width = source_desc.Width;
        transform.source_height = source_desc.Height;
        transform.input_width = config.input_width;
        transform.input_height = config.input_height;

        const float scale_x =
            static_cast<float>(config.input_width) /
            static_cast<float>(source_desc.Width);
        const float scale_y =
            static_cast<float>(config.input_height) /
            static_cast<float>(source_desc.Height);

        transform.scale = std::min(scale_x, scale_y);

        const float resized_width =
            static_cast<float>(source_desc.Width) * transform.scale;
        const float resized_height =
            static_cast<float>(source_desc.Height) * transform.scale;

        transform.pad_x =
            (static_cast<float>(config.input_width) - resized_width) * 0.5F;
        transform.pad_y =
            (static_cast<float>(config.input_height) - resized_height) * 0.5F;
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
            kTensorizeShader,
            std::strlen(kTensorizeShader),
            "scrfd_tensorize.hlsl",
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
                "Gagal compile SCRFD GPU preprocessing shader: " +
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
        desc.MipLODBias = 0.0F;
        desc.MaxAnisotropy = 1;
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
        ShaderParams params{};
        params.src_width = transform.source_width;
        params.src_height = transform.source_height;
        params.dst_width = config.input_width;
        params.dst_height = config.input_height;
        params.scale = transform.scale;
        params.pad_x = transform.pad_x;
        params.pad_y = transform.pad_y;

        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = sizeof(ShaderParams);
        desc.Usage = D3D11_USAGE_IMMUTABLE;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;
        desc.StructureByteStride = 0;

        D3D11_SUBRESOURCE_DATA data{};
        data.pSysMem = &params;

        check_hr_stage(
            interop_device->CreateBuffer(
                &desc,
                &data,
                params_buffer.put()
            ),
            "D3D11On12::CreateBuffer(shader params)"
        );
    }

    void create_tensor_d3d12_first() {
        const std::uint64_t element_count =
            static_cast<std::uint64_t>(config.input_width) *
            static_cast<std::uint64_t>(config.input_height) * 3ULL;
        const std::uint64_t byte_count = element_count * sizeof(float);

        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Alignment = 0;
        desc.Width = byte_count;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
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
            "ID3D12Device::CreateCommittedResource(tensor buffer)"
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

        winrt::com_ptr<::IUnknown> unknown_tensor;
        check_hr_stage(
            tensor_factory->CreateFromD3D12Resource(
                tensor_buffer_d3d12.get(),
                shape.data(),
                static_cast<int>(shape.size()),
                unknown_tensor.put()
            ),
            "ITensorStaticsNative::CreateFromD3D12Resource"
        );

        unknown_tensor.try_as(tensor);
        if (!tensor) {
            throw std::runtime_error(
                "Gagal membuat TensorFloat dari D3D12 tensor resource"
            );
        }

        // Promote resource D3D12 yang sama menjadi D3D11 raw UAV melalui
        // D3D11On12. Tidak ada shared D3D11 buffer.
        D3D11_RESOURCE_FLAGS flags11{};
        flags11.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        flags11.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        flags11.CPUAccessFlags = 0;
        flags11.StructureByteStride = 0;

        check_hr_stage(
            interop_11on12->CreateWrappedResource(
                tensor_buffer_d3d12.get(),
                &flags11,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COMMON,
                __uuidof(ID3D11Buffer),
                tensor_buffer_wrapped11.put_void()
            ),
            "ID3D11On12Device::CreateWrappedResource(tensor)"
        );

        D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
        uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
        uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.FirstElement = 0;
        uav_desc.Buffer.NumElements = static_cast<UINT>(element_count);
        uav_desc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;

        check_hr_stage(
            interop_device->CreateUnorderedAccessView(
                tensor_buffer_wrapped11.get(),
                &uav_desc,
                tensor_uav.put()
            ),
            "D3D11On12::CreateUnorderedAccessView(tensor)"
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

    TensorFloat preprocess(GpuFrame const& frame) {
        validate_gpu_frame(frame);

        if (
            frame.width != transform.source_width ||
            frame.height != transform.source_height ||
            frame.dxgi_format != DXGI_FORMAT_NV12
        ) {
            throw std::invalid_argument(
                "Format/dimensi camera frame berubah setelah SCRFD preprocessor diinisialisasi"
            );
        }

        ID3D11Device* frame_device_raw = nullptr;
        frame.texture->GetDevice(&frame_device_raw);
        if (frame_device_raw == nullptr) {
            throw std::runtime_error(
                "Camera frame tidak memiliki D3D11 device"
            );
        }
        winrt::com_ptr<ID3D11Device> frame_device;
        frame_device.attach(frame_device_raw);

        if (frame_device.get() != camera_device.get()) {
            throw std::runtime_error(
                "Camera D3D11 device berubah; SCRFD preprocessor harus dibuat ulang"
            );
        }

        // 1) Camera D3D11 -> shared NV12 texture. GPU-to-GPU copy saja.
        camera_context->CopyResource(
            shared_nv12_camera.get(),
            frame.texture.get()
        );

        const std::uint64_t fence_value = ++camera_fence_value;
        check_hr_stage(
            camera_context4->Signal(
                camera_fence11.get(),
                fence_value
            ),
            "ID3D11DeviceContext4::Signal(camera copy)"
        );
        camera_context->Flush();

        // 2) D3D11On12 menggunakan D3D12 queue yang sama dengan WinML.
        // Queue baru membaca shared NV12 setelah copy dari camera device selesai.
        check_hr_stage(
            d3d12_queue->Wait(
                camera_fence12.get(),
                fence_value
            ),
            "ID3D12CommandQueue::Wait(camera fence)"
        );

        ID3D11Resource* wrapped_resources[] = {
            tensor_buffer_wrapped11.get()
        };
        interop_11on12->AcquireWrappedResources(
            wrapped_resources,
            1
        );

        ID3D11ShaderResourceView* srvs[2] = {
            y_srv.get(),
            uv_srv.get(),
        };
        interop_context->CSSetShaderResources(0, 2, srvs);

        ID3D11UnorderedAccessView* uavs[1] = {
            tensor_uav.get(),
        };
        UINT initial_counts[1] = {0};
        interop_context->CSSetUnorderedAccessViews(
            0,
            1,
            uavs,
            initial_counts
        );

        ID3D11SamplerState* samplers[1] = {
            linear_sampler.get(),
        };
        interop_context->CSSetSamplers(0, 1, samplers);

        ID3D11Buffer* constants[1] = {
            params_buffer.get(),
        };
        interop_context->CSSetConstantBuffers(0, 1, constants);
        interop_context->CSSetShader(
            tensorize_shader.get(),
            nullptr,
            0
        );

        const UINT groups_x = (config.input_width + 15U) / 16U;
        const UINT groups_y = (config.input_height + 15U) / 16U;
        interop_context->Dispatch(groups_x, groups_y, 1);

        ID3D11ShaderResourceView* null_srvs[2] = {nullptr, nullptr};
        ID3D11UnorderedAccessView* null_uavs[1] = {nullptr};
        ID3D11SamplerState* null_samplers[1] = {nullptr};
        ID3D11Buffer* null_buffers[1] = {nullptr};

        interop_context->CSSetShaderResources(0, 2, null_srvs);
        interop_context->CSSetUnorderedAccessViews(
            0,
            1,
            null_uavs,
            initial_counts
        );
        interop_context->CSSetSamplers(0, 1, null_samplers);
        interop_context->CSSetConstantBuffers(0, 1, null_buffers);
        interop_context->CSSetShader(nullptr, nullptr, 0);

        // ReleaseWrappedResources() mengembalikan backing D3D12 resource ke
        // COMMON. Flush mengirim D3D11On12 commands ke queue D3D12 yang sama.
        // Evaluate() sesudah fungsi ini akan masuk ke queue yang sama sehingga
        // urutannya tetap GPU-side tanpa CPU wait/readback.
        interop_11on12->ReleaseWrappedResources(
            wrapped_resources,
            1
        );
        interop_context->Flush();

        return tensor;
    }
};

ScrfdGpuPreprocessor::ScrfdGpuPreprocessor(
    GpuFrame const& first_frame,
    ScrfdPreprocessConfig config
) : impl_(std::make_unique<Impl>(first_frame, config)) {}

ScrfdGpuPreprocessor::~ScrfdGpuPreprocessor() = default;

TensorFloat ScrfdGpuPreprocessor::preprocess(
    GpuFrame const& frame
) {
    return impl_->preprocess(frame);
}

LearningModelDevice ScrfdGpuPreprocessor::learning_model_device() const {
    return impl_->learning_device;
}

ScrfdPreprocessTransform ScrfdGpuPreprocessor::transform() const noexcept {
    return impl_->transform;
}

ScrfdPreprocessConfig ScrfdGpuPreprocessor::config() const noexcept {
    return impl_->config;
}

} // namespace vision_runtime
