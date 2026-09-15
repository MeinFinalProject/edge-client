// Hardware integration test. Synthetic NV12 input only; never opens a camera.
// Full tensor readback exists ONLY in this test, to inspect the production GPU output.
#include "liveness/gpu_preprocessor.hpp"
#include "liveness/liveness.hpp"
#include <d3d12.h>
#include <windows.ai.machinelearning.native.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace vision_runtime;
using winrt::check_hresult;
template<class T> using Ptr = winrt::com_ptr<T>;
void require(bool ok, char const* message) { if (!ok) throw std::runtime_error(message); }

std::vector<float> read_tensor(winrt::Windows::AI::MachineLearning::TensorFloat const& tensor) {
    Ptr<ID3D12Resource> source;
    check_hresult(tensor.as<ITensorNative>()->GetD3D12Resource(source.put()));
    Ptr<ID3D12Device> device;
    check_hresult(source->GetDevice(__uuidof(ID3D12Device), device.put_void()));
    Ptr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC q{};
    check_hresult(device->CreateCommandQueue(&q, __uuidof(ID3D12CommandQueue), queue.put_void()));
    Ptr<ID3D12CommandAllocator> allocator;
    check_hresult(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        __uuidof(ID3D12CommandAllocator), allocator.put_void()));
    Ptr<ID3D12GraphicsCommandList> list;
    check_hresult(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.get(), nullptr,
        __uuidof(ID3D12GraphicsCommandList), list.put_void()));
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    auto desc = source->GetDesc();
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    Ptr<ID3D12Resource> readback;
    check_hresult(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource), readback.put_void()));
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = source.get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    list->CopyResource(readback.get(), source.get());
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    list->ResourceBarrier(1, &barrier);
    check_hresult(list->Close());
    ID3D12CommandList* lists[] = {list.get()};
    queue->ExecuteCommandLists(1, lists);
    Ptr<ID3D12Fence> fence;
    check_hresult(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), fence.put_void()));
    winrt::handle event{CreateEventW(nullptr, FALSE, FALSE, nullptr)};
    require(static_cast<bool>(event), "Cannot create test fence event");
    check_hresult(queue->Signal(fence.get(), 1));
    check_hresult(fence->SetEventOnCompletion(1, event.get()));
    require(WaitForSingleObject(event.get(), 15000) == WAIT_OBJECT_0, "GPU test timed out");
    std::vector<float> values(3*80*80);
    void* mapped = nullptr;
    D3D12_RANGE range{0, values.size()*sizeof(float)};
    check_hresult(readback->Map(0, &range, &mapped));
    std::memcpy(values.data(), mapped, range.End);
    D3D12_RANGE no_write{0, 0};
    readback->Unmap(0, &no_write);
    return values;
}

constexpr int width = 160, height = 120;
double channel(int x, int y, int c) {
    x = std::clamp(x, 0, width-1); y = std::clamp(y, 0, height-1);
    const double luma = ((x*13+y*7)%200)/219.0;
    const double cb = (150.0-128)/224, cr = (90.0-128)/224;
    const double value = c==0 ? luma+1.5748*cr : c==1 ? luma-.187324*cb-.468124*cr : luma+1.8556*cb;
    return std::floor(std::clamp(value, 0.0, 1.0)*255+.5);
}

int main(int argc, char** argv) {
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        Ptr<ID3D11Device> device;
        Ptr<ID3D11DeviceContext> context;
        check_hresult(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
            device.put(), nullptr, context.put()));
        std::vector<unsigned char> bytes(width*height*3/2);
        for (int y=0; y<height; ++y) for (int x=0; x<width; ++x)
            bytes[y*width+x] = static_cast<unsigned char>(16+(x*13+y*7)%200);
        for (int i=width*height; i<static_cast<int>(bytes.size()); i+=2) {
            bytes[i]=150; bytes[i+1]=90;
        }
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width=width; desc.Height=height; desc.MipLevels=1; desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_NV12; desc.SampleDesc.Count=1;
        desc.Usage=D3D11_USAGE_DEFAULT; desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA data{bytes.data(),width,0};
        GpuFrame frame;
        check_hresult(device->CreateTexture2D(&desc, &data, frame.texture.put()));
        frame.width=width; frame.height=height; frame.dxgi_format=DXGI_FORMAT_NV12; frame.frame_id=1;
        liveness::LivenessGpuPreprocessor prep{frame};
        struct Case { char const* name; std::array<float,6> inverse; double missing; bool reject; };
        const std::vector<Case> cases{
            {"interior", {1,0,20,0,1,20}, 0, false},
            {"left_fractional", {1,0,-.25F,0,1,0}, .25/80, false},
            {"left_25_percent", {1,0,-20,0,1,0}, .25, false},
            {"left_30_percent", {1,0,-24,0,1,0}, .30, false},
            {"left_over_30", {1,0,-24.03125F,0,1,0}, 24.03125/80, true},
            {"left_under_30", {1,0,-23.96875F,0,1,0}, 23.96875/80, false},
            {"right_25_percent", {1,0,100,0,1,0}, .25, false},
            {"top_25_percent", {1,0,0,0,1,-20}, .25, false},
            {"bottom_25_percent", {1,0,0,0,1,60}, .25, false},
            {"corner", {1,0,-20,0,1,-20}, 1-.75*.75, true},
            {"fully_outside", {1,0,-300,0,1,-300}, 1, true},
            {"rotation_scale", {.75F,-.25F,40,.25F,.75F,10}, 0, false},
            {"recover_after_reject", {1,0,20,0,1,20}, 0, false}
        };
        for (auto const& test : cases) {
            arcface::SimilarityTransform2D transform{};
            transform.dst_to_src=test.inverse;
            prep.prepare_frame(frame);
            const auto output=prep.preprocess_aligned(transform);
            require(std::abs(output.out_of_bounds_ratio-test.missing)<1e-6, "GPU coverage differs from analytical area");
            require(output.input_rejected()==test.reject, "GPU gate boundary incorrect");
            const auto tensor=read_tensor(output.tensor);
            double max_error=0;
            for (int y=0; y<80; ++y) for (int x=0; x<80; ++x) {
                auto const& m=test.inverse;
                // Cases use exact multiples of 1/32, so no affine-table ambiguity.
                const double sx=m[0]*x+m[1]*y+m[2], sy=m[3]*x+m[4]*y+m[5];
                const int ix=static_cast<int>(std::floor(sx)), iy=static_cast<int>(std::floor(sy));
                const double fx=sx-ix, fy=sy-iy;
                for (int c=0; c<3; ++c) {
                    const double interpolated=(1-fy)*((1-fx)*channel(ix,iy,c)+fx*channel(ix+1,iy,c)) +
                        fy*((1-fx)*channel(ix,iy+1,c)+fx*channel(ix+1,iy+1,c));
                    const double expected=std::floor(interpolated+.5)/255;
                    const auto actual=tensor[c*6400+y*80+x];
                    require(std::isfinite(actual) && actual>=0 && actual<=1, "Invalid tensor element");
                    max_error=std::max(max_error,std::abs(actual-expected));
                }
            }
            require(max_error<=1.0/255+1e-6, "GPU tensor: border/RGB/layout/normalization mismatch");
            std::cout << test.name << " oob=" << output.out_of_bounds_ratio << " max_tensor_error=" << max_error << '\n';
        }
        arcface::SimilarityTransform2D invalid{};
        invalid.dst_to_src[0]=std::numeric_limits<float>::quiet_NaN();
        bool rejected=false;
        try { (void)prep.preprocess_aligned(invalid); } catch (std::invalid_argument const&) { rejected=true; }
        require(rejected, "Invalid transform reached GPU");
        if (argc == 2) {
            liveness::LivenessConfig config;
            config.model_path=argv[1];
            liveness::LivenessDetector detector{config};
            constexpr Point2f targets[5]={{32.03F,38.06F},{47.89F,37.98F},{40.01F,47.08F},{33.50F,56.36F},{46.63F,56.29F}};
            for (bool outside : {true,false,true,false}) {
                FaceDetection observation;
                for (int i=0;i<5;++i) observation.landmarks[i]={targets[i].x+(outside?-300.F:20.F),targets[i].y+20};
                const auto result=detector.evaluate(frame,observation);
                if (outside) require(!result.live_score && result.decision==liveness::LivenessDecision::InputRejected &&
                    result.timing.evaluate_ms==0, "Rejected input ran inference or acquired a score");
                else require(result.live_score && std::isfinite(*result.live_score), "Accepted input failed model inference");
            }
            std::cout << "Actual model GPU inference + rejected/accepted transitions PASS\n";
        }
        std::cout << "13 GPU preprocessing cases PASS; synthetic data, no camera\n";
        return 0;
    } catch(winrt::hresult_error const& e) { std::cerr << winrt::to_string(e.message()) << '\n'; }
      catch(std::exception const& e) { std::cerr << e.what() << '\n'; }
    return 1;
}
