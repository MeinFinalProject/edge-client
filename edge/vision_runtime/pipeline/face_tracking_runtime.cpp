#include "pipeline/face_tracking_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

namespace vision_runtime::pipeline {
namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

void validate_config(FaceTrackingConfig const& cfg) {
    if (cfg.scrfd.model_path.empty()) {
        throw std::invalid_argument("FaceTrackingConfig.scrfd.model_path belum diisi");
    }

    // ByteTrack's distinctive second association only works if low-score
    // detector candidates survive SCRFD's candidate filter.
    if (cfg.scrfd.score_threshold > cfg.bytetrack.low_threshold) {
        throw std::invalid_argument(
            "SCRFD score_threshold terlalu tinggi untuk ByteTrack: detector=" +
            std::to_string(cfg.scrfd.score_threshold) +
            ", ByteTrack low_threshold=" +
            std::to_string(cfg.bytetrack.low_threshold) +
            ". Gunakan SCRFD candidate threshold <= ByteTrack low_threshold."
        );
    }

    if (cfg.bytetrack.low_threshold >= cfg.bytetrack.track_threshold) {
        throw std::invalid_argument("ByteTrack low_threshold harus < track_threshold");
    }

    if (cfg.bytetrack.track_threshold >= cfg.bytetrack.new_track_threshold) {
        throw std::invalid_argument("ByteTrack track_threshold harus < new_track_threshold");
    }
}

FaceTrackingConfig validated_config(FaceTrackingConfig cfg) {
    validate_config(cfg);
    return cfg;
}

} // namespace

struct FaceTrackingRuntime::Impl {
    FaceTrackingConfig cfg;
    Camera camera;
    ScrfdDetector detector;
    bytetrack::ByteTracker tracker;
    std::uint64_t last_camera_frame_id = 0;
    bool started = false;

    explicit Impl(FaceTrackingConfig config)
        : cfg(validated_config(std::move(config))),
          camera(cfg.camera),
          detector(cfg.scrfd),
          tracker(cfg.bytetrack) {}

    FaceTrackingFrame process(GpuFrame frame) {
        const auto begin = Clock::now();

        auto detections = detector.detect(frame);
        const auto detector_timing = detector.last_timing();

        auto tracks = tracker.update(detections);
        const auto tracker_timing = tracker.last_timing();

        const auto end = Clock::now();

        FaceTrackingFrame out{};
        out.gpu_frame = std::move(frame);
        out.detections = std::move(detections);
        out.tracks = std::move(tracks);
        out.timing.detector_total_ms = detector_timing.total_ms;
        out.timing.detector_evaluate_ms = detector_timing.evaluate_ms;
        out.timing.detector_postprocess_ms = detector_timing.postprocess_ms;
        out.timing.tracker_total_ms = tracker_timing.update_total_ms;
        out.timing.tracker_association_ms = tracker_timing.association_ms;
        out.timing.pipeline_total_ms = ms(end - begin);
        return out;
    }
};

FaceTrackingRuntime::FaceTrackingRuntime(FaceTrackingConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

FaceTrackingRuntime::~FaceTrackingRuntime() {
    stop();
}

FaceTrackingRuntime& FaceTrackingRuntime::start() {
    if (!impl_->started) {
        impl_->camera.start();
        impl_->last_camera_frame_id = 0;
        impl_->tracker.reset();
        impl_->started = true;
    }
    return *this;
}

void FaceTrackingRuntime::stop() noexcept {
    if (!impl_ || !impl_->started) return;
    impl_->camera.stop();
    impl_->started = false;
    impl_->last_camera_frame_id = 0;
}

std::optional<FaceTrackingFrame> FaceTrackingRuntime::wait_for_frame(
    std::chrono::milliseconds timeout
) {
    if (!impl_->started) {
        throw std::logic_error("FaceTrackingRuntime::start() harus dipanggil dahulu");
    }

    auto frame = impl_->camera.wait_for_frame(impl_->last_camera_frame_id, timeout);
    if (!frame) return std::nullopt;

    impl_->last_camera_frame_id = frame->frame_id;
    return impl_->process(std::move(*frame));
}

void FaceTrackingRuntime::reset_tracker() {
    impl_->tracker.reset();
}

bool FaceTrackingRuntime::running() const noexcept {
    return impl_->started;
}

FaceTrackingConfig FaceTrackingRuntime::config() const {
    return impl_->cfg;
}

std::uint64_t FaceTrackingRuntime::failed_camera_reads() const noexcept {
    return impl_->camera.failed_reads();
}


} // namespace vision_runtime::pipeline
