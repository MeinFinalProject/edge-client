#include "bytetrack/bytetrack.hpp"

#include "bytetrack/assignment.hpp"
#include "bytetrack/association.hpp"
#include "bytetrack/kalman_filter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace vision_runtime::bytetrack {
namespace {

using Clock = std::chrono::steady_clock;
using TrackPtr = std::shared_ptr<class STrack>;

class STrack {
public:
    explicit STrack(FaceDetection const& detection)
        : tlwh_initial_(tlbr_to_tlwh(detection)),
          score_(detection.score),
          observation_(detection),
          has_observation_(true) {}

    [[nodiscard]] std::array<float,4> tlwh() const {
        if (!has_state_) return tlwh_initial_;
        std::array<float,4> out{
            static_cast<float>(mean_[0]),
            static_cast<float>(mean_[1]),
            static_cast<float>(mean_[2] * mean_[3]),
            static_cast<float>(mean_[3])
        };
        out[0] -= out[2] * 0.5F;
        out[1] -= out[3] * 0.5F;
        return out;
    }

    [[nodiscard]] std::array<float,4> tlbr() const {
        auto out = tlwh();
        out[2] += out[0];
        out[3] += out[1];
        return out;
    }

    [[nodiscard]] KalmanFilter::Measurement xyah() const { return tlwh_to_xyah(tlwh()); }

    static KalmanFilter::Measurement tlwh_to_xyah(std::array<float,4> const& b) {
        const double h = std::max(1e-6F, b[3]);
        return {
            static_cast<double>(b[0] + b[2] * 0.5F),
            static_cast<double>(b[1] + b[3] * 0.5F),
            static_cast<double>(b[2]) / h,
            h
        };
    }

    static std::array<float,4> tlbr_to_tlwh(FaceDetection const& d) {
        return {d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1};
    }

    void predict(KalmanFilter const& kf) {
        if (!has_state_) return;
        auto mean = mean_;
        if (state_ != TrackState::Tracked) mean[7] = 0.0;
        auto state = kf.predict(mean, covariance_);
        mean_ = state.mean;
        covariance_ = state.covariance;
    }

    void activate(KalmanFilter const& kf, int frame_id, std::uint64_t id) {
        track_id_ = id;
        auto state = kf.initiate(tlwh_to_xyah(tlwh_initial_));
        mean_ = state.mean;
        covariance_ = state.covariance;
        has_state_ = true;
        tracklet_len_ = 0;
        state_ = TrackState::Tracked;
        is_activated_ = (frame_id == 1);
        frame_id_ = frame_id;
        start_frame_ = frame_id;
    }

    void re_activate(KalmanFilter const& kf, STrack const& det, int frame_id) {
        auto state = kf.update(mean_, covariance_, det.xyah());
        mean_ = state.mean;
        covariance_ = state.covariance;
        tracklet_len_ = 0;
        state_ = TrackState::Tracked;
        is_activated_ = true;
        frame_id_ = frame_id;
        score_ = det.score_;
        observation_ = det.observation_;
        has_observation_ = det.has_observation_;
    }

    void update(KalmanFilter const& kf, STrack const& det, int frame_id) {
        frame_id_ = frame_id;
        ++tracklet_len_;
        auto state = kf.update(mean_, covariance_, det.xyah());
        mean_ = state.mean;
        covariance_ = state.covariance;
        state_ = TrackState::Tracked;
        is_activated_ = true;
        score_ = det.score_;
        observation_ = det.observation_;
        has_observation_ = det.has_observation_;
    }

    void mark_lost() noexcept { state_ = TrackState::Lost; }
    void mark_removed() noexcept { state_ = TrackState::Removed; }

    [[nodiscard]] AssociationItem association_item() const {
        auto b = tlbr();
        return {b[0], b[1], b[2], b[3], score_};
    }

    [[nodiscard]] TrackResult result() const {
        auto b = tlbr();
        TrackResult out{};
        out.track_id = track_id_;
        out.x1 = b[0];
        out.y1 = b[1];
        out.x2 = b[2];
        out.y2 = b[3];
        out.score = score_;
        out.state = state_;
        out.start_frame = start_frame_;
        out.frame_id = frame_id_;
        out.tracklet_len = tracklet_len_;
        out.has_observation = has_observation_;
        out.observation = observation_;
        return out;
    }

    std::uint64_t track_id_ = 0;
    float score_ = 0.0F;
    TrackState state_ = TrackState::New;
    bool is_activated_ = false;
    int frame_id_ = 0;
    int start_frame_ = 0;
    int tracklet_len_ = 0;

private:
    std::array<float,4> tlwh_initial_{};
    FaceDetection observation_{};
    bool has_observation_ = false;
    bool has_state_ = false;
    KalmanFilter::Mean mean_{};
    KalmanFilter::Covariance covariance_{};
};

std::vector<TrackPtr> joint_tracks(std::vector<TrackPtr> const& a, std::vector<TrackPtr> const& b) {
    std::vector<TrackPtr> out;
    out.reserve(a.size() + b.size());
    std::unordered_set<std::uint64_t> ids;
    for (auto const& t : a) {
        ids.insert(t->track_id_);
        out.push_back(t);
    }
    for (auto const& t : b) {
        if (ids.insert(t->track_id_).second) out.push_back(t);
    }
    return out;
}

std::vector<TrackPtr> subtract_tracks(std::vector<TrackPtr> const& a, std::vector<TrackPtr> const& b) {
    std::unordered_set<std::uint64_t> remove;
    for (auto const& t : b) remove.insert(t->track_id_);
    std::vector<TrackPtr> out;
    out.reserve(a.size());
    for (auto const& t : a) if (!remove.contains(t->track_id_)) out.push_back(t);
    return out;
}

std::vector<AssociationItem> to_items(std::vector<TrackPtr> const& tracks) {
    std::vector<AssociationItem> out;
    out.reserve(tracks.size());
    for (auto const& t : tracks) out.push_back(t->association_item());
    return out;
}

std::pair<std::vector<TrackPtr>, std::vector<TrackPtr>> remove_duplicates(
    std::vector<TrackPtr> const& a,
    std::vector<TrackPtr> const& b,
    double& association_ms,
    std::size_t& association_calls
) {
    if (a.empty() || b.empty()) return {a,b};
    auto ai = to_items(a);
    auto bi = to_items(b);
    std::vector<float> costs;
    AssociationTiming assoc{};
    cost_matrix(ai, bi, false, costs, assoc);
    association_ms += assoc.last_call_ms;
    ++association_calls;

    std::unordered_set<std::size_t> drop_a, drop_b;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (costs[i*b.size()+j] >= 0.15F) continue;
            const int time_a = a[i]->frame_id_ - a[i]->start_frame_;
            const int time_b = b[j]->frame_id_ - b[j]->start_frame_;
            if (time_a > time_b) drop_b.insert(j); else drop_a.insert(i);
        }
    }
    std::vector<TrackPtr> ra, rb;
    for (std::size_t i = 0; i < a.size(); ++i) if (!drop_a.contains(i)) ra.push_back(a[i]);
    for (std::size_t j = 0; j < b.size(); ++j) if (!drop_b.contains(j)) rb.push_back(b[j]);
    return {std::move(ra), std::move(rb)};
}

} // namespace

struct ByteTracker::Impl {
    ByteTrackConfig cfg;
    KalmanFilter kf;
    std::vector<TrackPtr> tracked;
    std::vector<TrackPtr> lost;
    std::vector<TrackPtr> removed;
    int frame = 0;
    int max_time_lost = 30;
    std::uint64_t next_id = 1;
    ByteTrackTiming timing{};

    explicit Impl(ByteTrackConfig c)
        : cfg(c) {
        max_time_lost = static_cast<int>(static_cast<double>(cfg.frame_rate) / 30.0 * cfg.track_buffer);
        if (cfg.new_track_threshold <= 0.0F) cfg.new_track_threshold = cfg.track_threshold + 0.1F;
    }

    AssignmentResult associate(
        std::vector<TrackPtr> const& tracks,
        std::vector<TrackPtr> const& detections,
        bool fuse_score,
        float threshold
    ) {
        auto ti = to_items(tracks);
        auto di = to_items(detections);
        std::vector<float> costs;
        AssociationTiming assoc{};
        cost_matrix(ti, di, fuse_score, costs, assoc);
        timing.association_ms += assoc.last_call_ms;
        ++timing.association_calls;
        return linear_assignment(costs, tracks.size(), detections.size(), threshold);
    }

    std::vector<TrackResult> update(std::vector<FaceDetection> const& input) {
        const auto begin = Clock::now();
        timing = {};
        ++frame;

        std::vector<TrackPtr> activated, refind, newly_lost, newly_removed;
        std::vector<TrackPtr> high, low;
        high.reserve(input.size());
        low.reserve(input.size());
        for (auto const& d : input) {
            if (d.score > cfg.track_threshold) {
                high.push_back(std::make_shared<STrack>(d));
            } else if (d.score > cfg.low_threshold && d.score < cfg.track_threshold) {
                low.push_back(std::make_shared<STrack>(d));
            }
        }

        std::vector<TrackPtr> unconfirmed, confirmed;
        for (auto const& t : tracked) {
            if (!t->is_activated_) unconfirmed.push_back(t);
            else confirmed.push_back(t);
        }

        auto pool = joint_tracks(confirmed, lost);
        for (auto& t : pool) t->predict(kf);

        // First association: high-score detections.
        auto first = associate(pool, high, !cfg.mot20, cfg.match_threshold);
        for (auto const& m : first.matches) {
            auto& track = pool[m.row];
            auto const& det = high[m.col];
            if (track->state_ == TrackState::Tracked) {
                track->update(kf, *det, frame);
                activated.push_back(track);
            } else {
                track->re_activate(kf, *det, frame);
                refind.push_back(track);
            }
        }

        // Second association: unmatched tracked tracks vs low-score detections.
        std::vector<TrackPtr> remaining_tracked;
        remaining_tracked.reserve(first.unmatched_rows.size());
        for (auto idx : first.unmatched_rows) {
            if (pool[idx]->state_ == TrackState::Tracked) remaining_tracked.push_back(pool[idx]);
        }
        auto second = associate(remaining_tracked, low, false, cfg.second_match_threshold);
        for (auto const& m : second.matches) {
            auto& track = remaining_tracked[m.row];
            auto const& det = low[m.col];
            if (track->state_ == TrackState::Tracked) {
                track->update(kf, *det, frame);
                activated.push_back(track);
            } else {
                track->re_activate(kf, *det, frame);
                refind.push_back(track);
            }
        }
        for (auto idx : second.unmatched_rows) {
            auto& track = remaining_tracked[idx];
            if (track->state_ != TrackState::Lost) {
                track->mark_lost();
                newly_lost.push_back(track);
            }
        }

        // Unconfirmed tracks compete only for high-score detections that were
        // unmatched in the first association.
        std::vector<TrackPtr> remaining_high;
        remaining_high.reserve(first.unmatched_cols.size());
        for (auto idx : first.unmatched_cols) remaining_high.push_back(high[idx]);
        auto unc = associate(unconfirmed, remaining_high, !cfg.mot20, cfg.unconfirmed_match_threshold);
        for (auto const& m : unc.matches) {
            auto& track = unconfirmed[m.row];
            track->update(kf, *remaining_high[m.col], frame);
            activated.push_back(track);
        }
        for (auto idx : unc.unmatched_rows) {
            auto& track = unconfirmed[idx];
            track->mark_removed();
            newly_removed.push_back(track);
        }

        // Start tracks from remaining high-score detections.
        for (auto idx : unc.unmatched_cols) {
            auto& track = remaining_high[idx];
            if (track->score_ < cfg.new_track_threshold) continue;
            track->activate(kf, frame, next_id++);
            activated.push_back(track);
        }

        for (auto const& t : lost) {
            if (frame - t->frame_id_ > max_time_lost) {
                t->mark_removed();
                newly_removed.push_back(t);
            }
        }

        std::vector<TrackPtr> tracked_only;
        for (auto const& t : tracked) if (t->state_ == TrackState::Tracked) tracked_only.push_back(t);
        tracked = joint_tracks(tracked_only, activated);
        tracked = joint_tracks(tracked, refind);

        lost = subtract_tracks(lost, tracked);
        lost.insert(lost.end(), newly_lost.begin(), newly_lost.end());
        lost = subtract_tracks(lost, newly_removed);
        removed.insert(removed.end(), newly_removed.begin(), newly_removed.end());

        auto dedup = remove_duplicates(tracked, lost, timing.association_ms, timing.association_calls);
        tracked = std::move(dedup.first);
        lost = std::move(dedup.second);

        std::vector<TrackResult> out;
        for (auto const& t : tracked) {
            if (t->is_activated_) out.push_back(t->result());
        }
        const auto end = Clock::now();
        timing.update_total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
        return out;
    }

    void reset() {
        tracked.clear(); lost.clear(); removed.clear(); frame = 0; next_id = 1; timing = {};
    }
};

ByteTracker::ByteTracker(ByteTrackConfig config)
    : impl_(std::make_unique<Impl>(config)) {}
ByteTracker::~ByteTracker() = default;
ByteTracker::ByteTracker(ByteTracker&&) noexcept = default;
ByteTracker& ByteTracker::operator=(ByteTracker&&) noexcept = default;

std::vector<TrackResult> ByteTracker::update(std::vector<FaceDetection> const& detections) {
    return impl_->update(detections);
}
void ByteTracker::reset() { impl_->reset(); }
ByteTrackTiming ByteTracker::last_timing() const noexcept { return impl_->timing; }
int ByteTracker::frame_id() const noexcept { return impl_->frame; }

} // namespace vision_runtime::bytetrack
