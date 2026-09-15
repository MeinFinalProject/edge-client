#pragma once

#include "bytetrack/bytetrack.hpp"
#include "camera/camera.hpp"
#include "scrfd/scrfd.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace vision_runtime::pipeline {

struct FaceTrackingConfig {
    CameraConfig camera{};
    ScrfdConfig scrfd{};
    bytetrack::ByteTrackConfig bytetrack{};
};

struct FaceTrackingTiming {
    double detector_total_ms = 0.0;
    double detector_evaluate_ms = 0.0;
    double detector_postprocess_ms = 0.0;
    double tracker_total_ms = 0.0;
    double tracker_association_ms = 0.0;
    double pipeline_total_ms = 0.0;
};

// One coherent frame boundary for downstream PAD / recognition.
// - gpu_frame: exact camera frame used by SCRFD.
// - detections: SCRFD candidates down to ByteTrack's low threshold.
// - tracks: active/confirmed temporal tracks for this same frame.
struct FaceTrackingFrame {
    GpuFrame gpu_frame{};
    std::vector<FaceDetection> detections;
    std::vector<bytetrack::TrackResult> tracks;
    FaceTrackingTiming timing{};
};

// Production orchestration for the validated native path:
// Camera (GPU-backed NV12) -> SCRFD (GPU) -> ByteTrack (CPU).
//
// This class intentionally does not perform PAD or recognition. The returned
// GpuFrame + current matched FaceDetection metadata form the boundary for the
// next pipeline stage without bringing the image through OpenCV/NumPy.
class FaceTrackingRuntime {
public:
    explicit FaceTrackingRuntime(FaceTrackingConfig config);
    ~FaceTrackingRuntime();

    FaceTrackingRuntime(FaceTrackingRuntime const&) = delete;
    FaceTrackingRuntime& operator=(FaceTrackingRuntime const&) = delete;
    FaceTrackingRuntime(FaceTrackingRuntime&&) = delete;
    FaceTrackingRuntime& operator=(FaceTrackingRuntime&&) = delete;

    FaceTrackingRuntime& start();
    void stop() noexcept;

    [[nodiscard]] std::optional<FaceTrackingFrame> wait_for_frame(
        std::chrono::milliseconds timeout = std::chrono::milliseconds{1000}
    );

    void reset_tracker();

    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] FaceTrackingConfig config() const;
    [[nodiscard]] std::uint64_t failed_camera_reads() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime::pipeline
