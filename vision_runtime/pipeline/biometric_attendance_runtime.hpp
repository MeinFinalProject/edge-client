#pragma once

#include "arcface/arcface.hpp"
#include "liveness/liveness.hpp"
#include "pipeline/face_tracking_runtime.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace vision_runtime::pipeline {

enum class BiometricTrackPhase {
    WaitingQuality = 0,
    CheckingPad,
    RejectedAttack,
    Recognizing,
    Recognized,
    UnknownIdentity,
    RecognitionDeferred,
    DuplicateIdentity,
};

struct GalleryTemplate {
    // Stable application-level identity, e.g. student/NIM UUID.
    std::string identity_id;

    // ArcFace 512-D template. It is normalized when loaded into the runtime.
    std::array<float, arcface::kEmbeddingSize> embedding{};
};

struct BiometricAttendanceConfig {
    FaceTrackingConfig tracking{};
    liveness::LivenessConfig pad{};
    arcface::ArcFaceConfig arcface{};

    // Track-level temporal vote. Default is majority 2-of-3.
    // This reduces frame noise; it does NOT fix a consistently misclassified
    // presentation attack.
    std::size_t pad_bona_fide_votes_required = 2;
    std::size_t pad_attack_votes_required = 2;
    std::size_t pad_max_attempts = 3;
    std::size_t pad_every_frames = 3;

    // Only expensive biometric inference is allowed on a current SCRFD
    // observation that passes these quality gates.
    float min_detection_score = 0.50F;
    float min_face_width = 64.0F;
    float min_face_height = 64.0F;

    // ArcFace is retried only after PAD passes. Embeddings from successful
    // attempts are averaged at track level before gallery matching.
    std::size_t recognition_max_attempts = 3;
    std::size_t recognition_every_frames = 3;

    // MUST be calibrated for the deployed gallery. No universal ArcFace cosine
    // threshold is assumed by this runtime.
    float recognition_similarity_threshold = std::numeric_limits<float>::quiet_NaN();
    std::string calibration_id;
    bool allow_uncalibrated_thresholds = false;

    // Development quality/retry policy; validate on target walk-through data.
    float max_alignment_rmse = 8.0F; // ArcFace output pixels, not source pixels
    float retry_quality_gain = 1.25F;
    std::size_t recognition_extra_attempts = 1;
    std::chrono::milliseconds event_cooldown{30000};
    std::chrono::milliseconds evidence_ttl{3000};

    // Optional ambiguity guard between the best and second-best DIFFERENT
    // identities in the gallery. Set 0 to disable.
    float recognition_min_margin = 0.0F;

    // Runtime-state garbage collection after a Track ID disappears.
    std::size_t max_missing_runtime_frames = 45;
};

struct BiometricTrackSnapshot {
    std::uint64_t track_id = 0;
    BiometricTrackPhase phase = BiometricTrackPhase::WaitingQuality;

    float current_detection_score = 0.0F;
    float current_face_width = 0.0F;
    float current_face_height = 0.0F;

    std::size_t pad_attempts = 0;
    std::size_t pad_input_rejections = 0; // crop rejection is not an attack vote
    std::size_t pad_bona_fide_votes = 0;
    std::size_t pad_attack_votes = 0;
    float pad_median_p_real = 0.0F;

    std::size_t recognition_attempts = 0;
    std::string identity_id;
    float identity_similarity = 0.0F;
    float identity_margin = 0.0F;
};

struct AttendanceEvent {
    std::int64_t occurred_at_utc_ms = 0;
    std::uint64_t camera_frame_id = 0;
    std::uint64_t track_id = 0;

    std::string identity_id;
    float similarity = 0.0F;
    float similarity_margin = 0.0F;

    float pad_median_p_real = 0.0F;
};

struct BiometricAttendanceTiming {
    double tracking_ms = 0.0;
    double pad_batch_wall_ms = 0.0;
    double arcface_batch_wall_ms = 0.0;
};

struct BiometricAttendanceFrame {
    GpuFrame gpu_frame{};
    std::vector<BiometricTrackSnapshot> tracks;
    std::vector<AttendanceEvent> attendance_events;
    BiometricAttendanceTiming timing{};
};

// TA application orchestration (validation is still in development):
// Camera -> SCRFD -> ByteTrack -> per-Track state
//   -> quality gate
//   -> PAD temporal vote
//   -> ArcFace temporal embedding + gallery match
//   -> biometric events with a short monotonic identity cooldown.
//
// The class intentionally does not write HTTP/SQLite itself. AttendanceEvent is
// the boundary consumed by edge_app for local persistence and synchronization.
class BiometricAttendanceRuntime {
public:
    explicit BiometricAttendanceRuntime(BiometricAttendanceConfig config);
    ~BiometricAttendanceRuntime();

    BiometricAttendanceRuntime(BiometricAttendanceRuntime const&) = delete;
    BiometricAttendanceRuntime& operator=(BiometricAttendanceRuntime const&) = delete;
    BiometricAttendanceRuntime(BiometricAttendanceRuntime&&) = delete;
    BiometricAttendanceRuntime& operator=(BiometricAttendanceRuntime&&) = delete;

    // Replace the in-memory recognition gallery. Multiple templates may use the
    // same identity_id; the matcher keeps the best score per identity.
    // Owner-thread only, between frames. Clears evidence, retains cooldowns.
    void set_gallery(std::vector<GalleryTemplate> gallery);
    [[nodiscard]] std::size_t gallery_size() const noexcept;

    BiometricAttendanceRuntime& start();
    void stop() noexcept;

    [[nodiscard]] std::optional<BiometricAttendanceFrame> wait_for_frame(
        std::chrono::milliseconds timeout = std::chrono::milliseconds{1000}
    );

    // Clears Track evidence; keeps the gallery and unexpired identity cooldowns.
    void reset_session();

    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] BiometricAttendanceConfig config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] char const* to_string(BiometricTrackPhase phase) noexcept;

} // namespace vision_runtime::pipeline
