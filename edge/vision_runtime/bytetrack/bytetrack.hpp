#pragma once

#include "common/vision_types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace vision_runtime::bytetrack {

enum class TrackState {
    New = 0,
    Tracked,
    Lost,
    Removed,
};

struct ByteTrackConfig {
    float track_threshold = 0.50F;
    float low_threshold = 0.10F;
    float new_track_threshold = 0.60F; // official logic: track_threshold + 0.1
    float match_threshold = 0.80F;
    float second_match_threshold = 0.50F;
    float unconfirmed_match_threshold = 0.70F;
    int track_buffer = 30;
    int frame_rate = 30;
    bool mot20 = false; // false => fuse detection score in first/unconfirmed association
};

struct TrackResult {
    std::uint64_t track_id = 0;

    // Kalman-filtered track box in original camera coordinates.
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;

    // Score of the detection matched to this track on the current frame.
    float score = 0.0F;
    TrackState state = TrackState::New;
    int start_frame = 0;
    int frame_id = 0;
    int tracklet_len = 0;

    // Current SCRFD observation that updated this track, including the five
    // landmarks needed by downstream PAD/ArcFace. This is deliberately kept
    // separate from the smoothed Kalman box above.
    bool has_observation = false;
    FaceDetection observation{};
};

struct ByteTrackTiming {
    double update_total_ms = 0.0;
    // Sum of native CPU IoU/fuse-score cost-matrix calls in this update.
    double association_ms = 0.0;
    std::size_t association_calls = 0;
};

class ByteTracker {
public:
    explicit ByteTracker(ByteTrackConfig config);
    ~ByteTracker();

    ByteTracker(ByteTracker const&) = delete;
    ByteTracker& operator=(ByteTracker const&) = delete;
    ByteTracker(ByteTracker&&) noexcept;
    ByteTracker& operator=(ByteTracker&&) noexcept;

    // Input detections must already be in original camera coordinates. To keep
    // ByteTrack's second association alive, detector candidates down to
    // low_threshold (normally 0.1) must reach this function.
    std::vector<TrackResult> update(std::vector<FaceDetection> const& detections);

    void reset();

    [[nodiscard]] ByteTrackTiming last_timing() const noexcept;
    [[nodiscard]] int frame_id() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime::bytetrack
