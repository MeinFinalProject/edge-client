#pragma once

#include <cstdint>

#include <d3d11.h>
#include <winrt/base.h>

namespace vision_runtime {

// Shared GPU-image boundary for camera, detection, PAD, and recognition.
// The texture is reference-counted and remains valid while this object lives.
// Pixel data stays on the Direct3D path; small metadata is CPU-visible.
struct GpuFrame {
    winrt::com_ptr<ID3D11Texture2D> texture;

    // Monotonic within one Camera::start() session; resets on restart.
    std::uint64_t frame_id = 0;

    // steady_clock-based application timestamps, expressed as nanoseconds.
    std::uint64_t callback_entry_ns = 0;
    std::uint64_t timestamp_ns = 0;

    // Media Foundation system-relative timestamp when supplied, otherwise 0.
    std::uint64_t source_timestamp_ns = 0;

    std::uint32_t width = 0;
    std::uint32_t height = 0;
    DXGI_FORMAT dxgi_format = DXGI_FORMAT_UNKNOWN;

    [[nodiscard]] bool valid() const noexcept {
        return texture != nullptr;
    }
};

} // namespace vision_runtime
