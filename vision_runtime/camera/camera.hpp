#pragma once

#include "common/gpu_frame.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace vision_runtime {

enum class CameraAcquisitionMode {
    Realtime,
    Buffered,
};

enum class CameraStreamPreference {
    Auto,
    VideoPreview,
    VideoRecord,
};

struct CameraMode {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    double fps = 0.0;

    // Format selected on MediaFrameSource.
    std::wstring subtype;

    // Format requested from MediaFrameReader. The GPU pipeline defaults to NV12.
    std::wstring output_subtype;

    CameraStreamPreference stream = CameraStreamPreference::Auto;
};

struct CameraConfig {
    std::uint32_t index = 0;
    std::uint32_t width = 1280;
    std::uint32_t height = 720;
    std::uint32_t fps = 30;

    CameraAcquisitionMode acquisition_mode = CameraAcquisitionMode::Realtime;
    CameraStreamPreference stream_preference = CameraStreamPreference::Auto;

    // Empty selects a source automatically. Probes may pin an exact source ID.
    std::wstring source_id;

    // Empty selects a source subtype using production ranking.
    std::wstring source_subtype;

    // Empty requests the source subtype; production defaults to NV12 output.
    std::wstring output_subtype = L"NV12";

    // Force legacy UVC/KS low-light auto-exposure priority (#19) to zero before
    // opening Media Foundation capture, preserving a constant frame cadence.
    bool force_constant_framerate = true;

    // Restore the user's legacy property #19 value on Camera::stop().
    bool restore_legacy_control_on_stop = true;
};

struct CameraControlStatus {
    bool legacy_low_light_supported = false;
    bool legacy_low_light_forced_off = false;
    bool legacy_low_light_restore_pending = false;
    long legacy_low_light_original_value = -1;
    long legacy_low_light_current_value = -1;

    bool exposure_priority_supported = false;
    bool exposure_priority_enabled = false;
    bool exposure_priority_forced_off = false;

    bool exposure_control_supported = false;
    bool exposure_auto = false;
    std::int64_t exposure_value_100ns = 0;
    std::int64_t exposure_min_100ns = 0;
    std::int64_t exposure_max_100ns = 0;
    std::int64_t exposure_step_100ns = 0;
};

class Camera {
public:
    explicit Camera(CameraConfig config = {});

    // Backward-compatible convenience constructor.
    Camera(
        std::uint32_t index,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t fps,
        CameraAcquisitionMode acquisition_mode = CameraAcquisitionMode::Realtime
    );

    ~Camera();

    Camera(Camera const&) = delete;
    Camera& operator=(Camera const&) = delete;
    Camera(Camera&&) = delete;
    Camera& operator=(Camera&&) = delete;

    Camera& start();
    void stop() noexcept;

    [[nodiscard]] std::optional<GpuFrame> wait_for_frame(
        std::uint64_t last_frame_id = 0,
        std::chrono::milliseconds timeout = std::chrono::milliseconds{1000}
    );

    [[nodiscard]] std::uint64_t failed_reads() const noexcept;
    [[nodiscard]] std::uint64_t frame_id() const noexcept;
    [[nodiscard]] bool running() const noexcept;

    [[nodiscard]] CameraMode mode() const;
    [[nodiscard]] CameraControlStatus control_status() const;
    [[nodiscard]] CameraAcquisitionMode acquisition_mode() const noexcept;
    [[nodiscard]] CameraStreamPreference stream_preference() const noexcept;
    [[nodiscard]] CameraConfig config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime
