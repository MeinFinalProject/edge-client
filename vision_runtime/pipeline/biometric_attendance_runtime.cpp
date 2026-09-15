#include "pipeline/biometric_attendance_runtime.hpp"
#include "liveness/vote.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vision_runtime::pipeline {
namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

float median(std::vector<float> values) {
    if (values.empty()) return 0.0F;
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2];
    return 0.5F * (values[n / 2 - 1] + values[n / 2]);
}

bool finite_probability(float v) noexcept {
    return std::isfinite(v) && v >= 0.0F && v <= 1.0F;
}

void validate_config(BiometricAttendanceConfig const& cfg) {
    if (!finite_probability(cfg.pad.live_threshold)) {
        throw std::invalid_argument("pad.live_threshold harus 0..1");
    }
    if (
        cfg.pad_bona_fide_votes_required == 0 ||
        cfg.pad_attack_votes_required == 0 ||
        cfg.pad_max_attempts == 0 ||
        cfg.pad_every_frames == 0
    ) {
        throw std::invalid_argument("Konfigurasi voting/cadence PAD harus >= 1");
    }
    if (
        cfg.pad_bona_fide_votes_required > cfg.pad_max_attempts ||
        cfg.pad_attack_votes_required > cfg.pad_max_attempts
    ) {
        throw std::invalid_argument(
            "Vote PAD yang dibutuhkan tidak boleh melebihi pad_max_attempts"
        );
    }
    if (
        !std::isfinite(cfg.min_detection_score) ||
        cfg.min_detection_score < 0.0F ||
        cfg.min_detection_score > 1.0F
    ) {
        throw std::invalid_argument("min_detection_score harus 0..1");
    }
    if (
        !std::isfinite(cfg.min_face_width) ||
        !std::isfinite(cfg.min_face_height) ||
        cfg.min_face_width <= 0.0F ||
        cfg.min_face_height <= 0.0F
    ) {
        throw std::invalid_argument("minimum face size harus > 0");
    }
    if (
        cfg.recognition_max_attempts == 0 ||
        cfg.recognition_every_frames == 0
    ) {
        throw std::invalid_argument("Konfigurasi recognition harus >= 1");
    }
    if (
        !std::isfinite(cfg.recognition_similarity_threshold) ||
        cfg.recognition_similarity_threshold <= -1.0F ||
        cfg.recognition_similarity_threshold > 1.0F
    ) {
        throw std::invalid_argument(
            "recognition_similarity_threshold harus berada pada (-1,1]"
        );
    }
    if (
        !std::isfinite(cfg.recognition_min_margin) ||
        cfg.recognition_min_margin < 0.0F ||
        cfg.recognition_min_margin > 2.0F
    ) {
        throw std::invalid_argument("recognition_min_margin harus 0..2");
    }
    if (cfg.max_missing_runtime_frames == 0) {
        throw std::invalid_argument("max_missing_runtime_frames harus >= 1");
    }
}

bool quality_ok(
    bytetrack::TrackResult const& track,
    BiometricAttendanceConfig const& cfg
) noexcept {
    if (!track.has_observation) return false;

    const auto& obs = track.observation;
    const float width = obs.x2 - obs.x1;
    const float height = obs.y2 - obs.y1;

    return
        obs.score >= cfg.min_detection_score &&
        width >= cfg.min_face_width &&
        height >= cfg.min_face_height;
}

struct MatchResult {
    bool valid = false;
    std::string identity_id;
    float best_similarity = -1.0F;
    float second_similarity = -1.0F;
    float margin = 0.0F;
};

} // namespace

char const* to_string(BiometricTrackPhase phase) noexcept {
    switch (phase) {
    case BiometricTrackPhase::WaitingQuality:   return "WAITING_QUALITY";
    case BiometricTrackPhase::CheckingPad:      return "CHECKING_PAD";
    case BiometricTrackPhase::RejectedAttack:   return "REJECTED_ATTACK";
    case BiometricTrackPhase::Recognizing:      return "RECOGNIZING";
    case BiometricTrackPhase::Recognized:       return "RECOGNIZED";
    case BiometricTrackPhase::UnknownIdentity:  return "UNKNOWN_IDENTITY";
    case BiometricTrackPhase::DuplicateIdentity:return "DUPLICATE_IDENTITY";
    default:                                    return "UNKNOWN_PHASE";
    }
}

struct BiometricAttendanceRuntime::Impl {
    struct TrackState {
        BiometricTrackPhase phase = BiometricTrackPhase::WaitingQuality;

        std::size_t first_seen_runtime_frame = 0;
        std::size_t last_seen_runtime_frame = 0;

        std::size_t last_pad_runtime_frame = 0;
        liveness::VoteCounts pad_votes;
        std::vector<float> pad_p_real;

        std::size_t last_recognition_runtime_frame = 0;
        std::size_t recognition_attempts = 0;
        std::array<double, arcface::kEmbeddingSize> embedding_sum{};
        std::size_t embedding_count = 0;

        std::string identity_id;
        float identity_similarity = 0.0F;
        float identity_margin = 0.0F;
    };

    struct PadWork {
        std::uint64_t track_id = 0;
        FaceDetection observation{};
    };

    struct ArcWork {
        std::uint64_t track_id = 0;
        FaceDetection observation{};
    };

    BiometricAttendanceConfig cfg;
    FaceTrackingRuntime tracking;
    liveness::LivenessDetector pad_detector;
    arcface::ArcFaceRecognizer recognizer;

    std::vector<GalleryTemplate> gallery;
    std::unordered_map<std::uint64_t, TrackState> states;
    std::unordered_set<std::string> emitted_identities;

    std::size_t runtime_frame = 0;
    bool started = false;

    explicit Impl(BiometricAttendanceConfig config)
        : cfg(std::move(config)),
          tracking(cfg.tracking),
          pad_detector(cfg.pad),
          recognizer(cfg.arcface) {
        validate_config(cfg);
    }

    void set_gallery(std::vector<GalleryTemplate> value) {
        for (auto& item : value) {
            if (item.identity_id.empty()) {
                throw std::invalid_argument(
                    "GalleryTemplate.identity_id tidak boleh kosong"
                );
            }
            item.embedding = arcface::normalize_embedding(
                std::move(item.embedding)
            );
        }
        gallery = std::move(value);
    }

    MatchResult match_gallery(
        std::array<float, arcface::kEmbeddingSize> const& query
    ) const {
        MatchResult result{};
        if (gallery.empty()) return result;

        // Collapse multiple templates belonging to the same identity by taking
        // the best cosine score for that identity.
        std::unordered_map<std::string, float> per_identity;
        per_identity.reserve(gallery.size());

        for (auto const& item : gallery) {
            const float score = arcface::cosine_similarity(
                query,
                item.embedding
            );

            auto [it, inserted] = per_identity.emplace(
                item.identity_id,
                score
            );
            if (!inserted && score > it->second) {
                it->second = score;
            }
        }

        for (auto const& [identity, score] : per_identity) {
            if (!result.valid || score > result.best_similarity) {
                result.second_similarity = result.best_similarity;
                result.best_similarity = score;
                result.identity_id = identity;
                result.valid = true;
            } else if (score > result.second_similarity) {
                result.second_similarity = score;
            }
        }

        if (!result.valid) return result;

        if (per_identity.size() < 2) {
            result.second_similarity = -1.0F;
            result.margin = 2.0F;
        } else {
            result.margin =
                result.best_similarity - result.second_similarity;
        }
        return result;
    }

    std::array<float, arcface::kEmbeddingSize> aggregate_embedding(
        TrackState const& state
    ) const {
        if (state.embedding_count == 0) {
            throw std::logic_error("aggregate_embedding tanpa embedding");
        }

        std::array<float, arcface::kEmbeddingSize> mean{};
        const double divisor =
            static_cast<double>(state.embedding_count);

        for (std::size_t i = 0; i < mean.size(); ++i) {
            mean[i] = static_cast<float>(
                state.embedding_sum[i] / divisor
            );
        }
        return arcface::normalize_embedding(std::move(mean));
    }

    void prune_stale_states() {
        for (auto it = states.begin(); it != states.end(); ) {
            const auto age =
                runtime_frame - it->second.last_seen_runtime_frame;

            if (age > cfg.max_missing_runtime_frames) {
                it = states.erase(it);
            } else {
                ++it;
            }
        }
    }

    BiometricTrackSnapshot snapshot(
        bytetrack::TrackResult const& track,
        TrackState const& state
    ) const {
        BiometricTrackSnapshot out{};
        out.track_id = track.track_id;
        out.phase = state.phase;

        if (track.has_observation) {
            out.current_detection_score = track.observation.score;
            out.current_face_width =
                track.observation.x2 - track.observation.x1;
            out.current_face_height =
                track.observation.y2 - track.observation.y1;
        }

        out.pad_attempts = state.pad_votes.attempts;
        out.pad_input_rejections = state.pad_votes.input_rejected;
        out.pad_bona_fide_votes = state.pad_votes.real;
        out.pad_attack_votes = state.pad_votes.spoof;
        out.pad_median_p_real = median(state.pad_p_real);

        out.recognition_attempts = state.recognition_attempts;
        out.identity_id = state.identity_id;
        out.identity_similarity = state.identity_similarity;
        out.identity_margin = state.identity_margin;
        return out;
    }

    BiometricAttendanceFrame process(FaceTrackingFrame tracking_frame) {
        ++runtime_frame;

        BiometricAttendanceFrame out{};
        out.gpu_frame = tracking_frame.gpu_frame;
        out.timing.tracking_ms = tracking_frame.timing.pipeline_total_ms;

        // ---------------------------------------------------------------------
        // 1. Refresh per-Track state from current ByteTrack outputs.
        // ---------------------------------------------------------------------
        for (auto const& track : tracking_frame.tracks) {
            auto [it, inserted] = states.try_emplace(track.track_id);
            auto& state = it->second;

            if (inserted) {
                state.first_seen_runtime_frame = runtime_frame;
            }
            state.last_seen_runtime_frame = runtime_frame;
        }

        prune_stale_states();

        // ---------------------------------------------------------------------
        // 2. Select PAD work. Terminal tracks never receive expensive inference.
        // ---------------------------------------------------------------------
        std::vector<PadWork> pad_work;
        std::vector<FaceDetection> pad_observations;

        for (auto const& track : tracking_frame.tracks) {
            auto& state = states.at(track.track_id);

            const bool terminal =
                state.phase == BiometricTrackPhase::RejectedAttack ||
                state.phase == BiometricTrackPhase::Recognized ||
                state.phase == BiometricTrackPhase::UnknownIdentity ||
                state.phase == BiometricTrackPhase::DuplicateIdentity;

            if (terminal) continue;
            // Once PAD passes, only recognition retries are scheduled.
            if (state.phase == BiometricTrackPhase::Recognizing) continue;

            if (!quality_ok(track, cfg)) {
                if (state.pad_votes.attempts == 0) {
                    state.phase = BiometricTrackPhase::WaitingQuality;
                }
                continue;
            }

            if (state.pad_votes.attempts >= cfg.pad_max_attempts) continue;

            const bool cadence_ready =
                state.last_pad_runtime_frame == 0 ||
                runtime_frame - state.last_pad_runtime_frame >=
                    cfg.pad_every_frames;

            if (!cadence_ready) continue;

            state.phase = BiometricTrackPhase::CheckingPad;

            pad_work.push_back(PadWork{track.track_id, track.observation});
            pad_observations.push_back(track.observation);
        }

        if (!pad_observations.empty()) {
            const auto begin = Clock::now();
            auto results = pad_detector.evaluate_many(
                tracking_frame.gpu_frame,
                pad_observations
            );
            const auto end = Clock::now();
            out.timing.pad_batch_wall_ms = ms(end - begin);

            if (results.size() != pad_work.size()) {
                throw std::runtime_error("PAD batch/result size mismatch");
            }

            for (std::size_t i = 0; i < results.size(); ++i) {
                auto& state = states.at(pad_work[i].track_id);
                const auto& result = results[i];

                state.last_pad_runtime_frame = runtime_frame;
                if (!state.pad_votes.add(result.decision)) {
                    state.phase = BiometricTrackPhase::WaitingQuality;
                    continue;
                }
                if (!result.live_score) throw std::logic_error("Accepted PAD result has no score");
                state.pad_p_real.push_back(*result.live_score);


                if (
                    state.pad_votes.spoof >=
                    cfg.pad_attack_votes_required
                ) {
                    state.phase = BiometricTrackPhase::RejectedAttack;
                    continue;
                }

                if (
                    state.pad_votes.real >=
                    cfg.pad_bona_fide_votes_required
                ) {
                    state.phase = BiometricTrackPhase::Recognizing;
                    continue;
                }

                if (state.pad_votes.attempts >= cfg.pad_max_attempts) {
                    // Fail closed if the configured vote rule is somehow not
                    // resolved by max_attempts.
                    state.phase = BiometricTrackPhase::RejectedAttack;
                }
            }
        }

        // ---------------------------------------------------------------------
        // 3. Select ArcFace work only for tracks whose PAD state passed.
        // ---------------------------------------------------------------------
        std::vector<ArcWork> arc_work;
        std::vector<FaceDetection> arc_observations;

        for (auto const& track : tracking_frame.tracks) {
            auto& state = states.at(track.track_id);

            if (state.phase != BiometricTrackPhase::Recognizing) continue;
            if (!quality_ok(track, cfg)) continue;

            if (
                state.recognition_attempts >=
                cfg.recognition_max_attempts
            ) {
                state.phase = BiometricTrackPhase::UnknownIdentity;
                continue;
            }

            const bool cadence_ready =
                state.last_recognition_runtime_frame == 0 ||
                runtime_frame - state.last_recognition_runtime_frame >=
                    cfg.recognition_every_frames;

            if (!cadence_ready) continue;

            arc_work.push_back(ArcWork{
                track.track_id,
                track.observation
            });
            arc_observations.push_back(track.observation);
        }

        if (!arc_observations.empty()) {
            const auto begin = Clock::now();
            auto results = recognizer.evaluate_many(
                tracking_frame.gpu_frame,
                arc_observations
            );
            const auto end = Clock::now();
            out.timing.arcface_batch_wall_ms = ms(end - begin);

            if (results.size() != arc_work.size()) {
                throw std::runtime_error(
                    "ArcFace batch/result size mismatch"
                );
            }

            for (std::size_t i = 0; i < results.size(); ++i) {
                auto& state = states.at(arc_work[i].track_id);
                auto const& result = results[i];

                state.last_recognition_runtime_frame = runtime_frame;
                ++state.recognition_attempts;
                ++state.embedding_count;

                for (
                    std::size_t d = 0;
                    d < arcface::kEmbeddingSize;
                    ++d
                ) {
                    state.embedding_sum[d] += result.embedding[d];
                }

                const auto aggregate = aggregate_embedding(state);
                const auto match = match_gallery(aggregate);

                if (match.valid) {
                    state.identity_id = match.identity_id;
                    state.identity_similarity = match.best_similarity;
                    state.identity_margin = match.margin;
                }

                const bool similarity_ok =
                    match.valid &&
                    match.best_similarity >=
                        cfg.recognition_similarity_threshold;

                const bool margin_ok =
                    match.valid &&
                    match.margin >= cfg.recognition_min_margin;

                if (similarity_ok && margin_ok) {
                    if (
                        emitted_identities.insert(match.identity_id).second
                    ) {
                        state.phase = BiometricTrackPhase::Recognized;

                        AttendanceEvent event{};
                        event.camera_frame_id =
                            tracking_frame.gpu_frame.frame_id;
                        event.track_id = arc_work[i].track_id;
                        event.identity_id = match.identity_id;
                        event.similarity = match.best_similarity;
                        event.similarity_margin = match.margin;
                        event.pad_median_p_real =
                            median(state.pad_p_real);
                        out.attendance_events.push_back(std::move(event));
                    } else {
                        // Handles track fragmentation / re-entry without
                        // creating duplicate attendance for the same class
                        // session.
                        state.phase =
                            BiometricTrackPhase::DuplicateIdentity;
                    }
                    continue;
                }

                if (
                    state.recognition_attempts >=
                    cfg.recognition_max_attempts
                ) {
                    state.phase =
                        BiometricTrackPhase::UnknownIdentity;
                }
            }
        }

        // ---------------------------------------------------------------------
        // 4. Expose current state for UI/telemetry; no expensive work below.
        // ---------------------------------------------------------------------
        out.tracks.reserve(tracking_frame.tracks.size());
        for (auto const& track : tracking_frame.tracks) {
            out.tracks.push_back(
                snapshot(track, states.at(track.track_id))
            );
        }

        return out;
    }
};

BiometricAttendanceRuntime::BiometricAttendanceRuntime(
    BiometricAttendanceConfig config
)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

BiometricAttendanceRuntime::~BiometricAttendanceRuntime() {
    stop();
}

void BiometricAttendanceRuntime::set_gallery(
    std::vector<GalleryTemplate> gallery
) {
    impl_->set_gallery(std::move(gallery));
}

std::size_t BiometricAttendanceRuntime::gallery_size() const noexcept {
    return impl_->gallery.size();
}

BiometricAttendanceRuntime& BiometricAttendanceRuntime::start() {
    if (!impl_->started) {
        if (impl_->gallery.empty()) {
            throw std::logic_error(
                "Recognition gallery kosong; panggil set_gallery() sebelum start()"
            );
        }

        impl_->states.clear();
        impl_->emitted_identities.clear();
        impl_->runtime_frame = 0;

        impl_->tracking.start();
        impl_->started = true;
    }
    return *this;
}

void BiometricAttendanceRuntime::stop() noexcept {
    if (!impl_ || !impl_->started) return;
    impl_->tracking.stop();
    impl_->started = false;
}

std::optional<BiometricAttendanceFrame>
BiometricAttendanceRuntime::wait_for_frame(
    std::chrono::milliseconds timeout
) {
    if (!impl_->started) {
        throw std::logic_error(
            "BiometricAttendanceRuntime::start() harus dipanggil dahulu"
        );
    }

    auto frame = impl_->tracking.wait_for_frame(timeout);
    if (!frame) return std::nullopt;

    return impl_->process(std::move(*frame));
}

void BiometricAttendanceRuntime::reset_session() {
    impl_->states.clear();
    impl_->emitted_identities.clear();
    impl_->runtime_frame = 0;
    impl_->tracking.reset_tracker();
}

bool BiometricAttendanceRuntime::running() const noexcept {
    return impl_->started;
}

BiometricAttendanceConfig BiometricAttendanceRuntime::config() const {
    return impl_->cfg;
}

} // namespace vision_runtime::pipeline
