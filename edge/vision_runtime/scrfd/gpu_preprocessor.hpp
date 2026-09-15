#pragma once

#include "common/gpu_frame.hpp"

#include <cstdint>
#include <memory>

#include <winrt/Windows.AI.MachineLearning.h>

namespace vision_runtime {

struct ScrfdPreprocessConfig {
    std::uint32_t input_width = 640;
    std::uint32_t input_height = 640;
};

struct ScrfdPreprocessTransform {
    float scale = 1.0F;
    float pad_x = 0.0F;
    float pad_y = 0.0F;
    std::uint32_t source_width = 0;
    std::uint32_t source_height = 0;
    std::uint32_t input_width = 0;
    std::uint32_t input_height = 0;
};

// GPU preprocessing khusus input SCRFD.
//
// Jalur data v3.4:
//   camera ID3D11Texture2D NV12
//      -> GPU CopyResource ke legacy-shared NV12 Texture2D
//      -> D3D11On12 compute shader
//         * bilinear resize + letterbox
//         * NV12 -> RGB (BT.709 limited range)
//         * (x - 127.5) / 128
//         * NCHW float32
//      -> D3D12-first tensor buffer
//      -> TensorFloat WinML
//
// Kenapa D3D12-first:
// D3D11 cross-device shared resources secara resmi ditujukan untuk Texture2D;
// shared structured buffer v3.2 ditolak driver dengan E_INVALIDARG.
// v3.3 kemudian menemukan NV12 + SHARED_NTHANDLE juga ditolak driver; v3.4
// memakai legacy DXGI SHARED handle untuk texture NV12. Tensor
// sekarang dibuat sebagai native D3D12 buffer, lalu hanya diberi wrapped D3D11
// UAV melalui D3D11On12 agar compute shader dapat mengisinya.
//
// Tidak ada Map()/readback pixel ke CPU.
class ScrfdGpuPreprocessor {
public:
    explicit ScrfdGpuPreprocessor(
        GpuFrame const& first_frame,
        ScrfdPreprocessConfig config = {}
    );

    ~ScrfdGpuPreprocessor();

    ScrfdGpuPreprocessor(ScrfdGpuPreprocessor const&) = delete;
    ScrfdGpuPreprocessor& operator=(ScrfdGpuPreprocessor const&) = delete;
    ScrfdGpuPreprocessor(ScrfdGpuPreprocessor&&) = delete;
    ScrfdGpuPreprocessor& operator=(ScrfdGpuPreprocessor&&) = delete;

    [[nodiscard]] winrt::Windows::AI::MachineLearning::TensorFloat preprocess(
        GpuFrame const& frame
    );

    [[nodiscard]] winrt::Windows::AI::MachineLearning::LearningModelDevice
    learning_model_device() const;

    [[nodiscard]] ScrfdPreprocessTransform transform() const noexcept;
    [[nodiscard]] ScrfdPreprocessConfig config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime
