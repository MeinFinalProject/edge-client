#include "pipeline/attendance_processor.hpp"
#include "liveness/vote.hpp"
#include "pipeline/gallery_matcher.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vision_runtime::pipeline
{
namespace
{
using Phase = BiometricTrackPhase;
float median(std::vector<float> v)
{
    if (v.empty())
        return 0;
    std::sort(v.begin(), v.end());
    auto n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) * .5F;
}
double elapsed(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
float quality(bytetrack::TrackResult const &t, BiometricAttendanceConfig const &c)
{
    if (!t.has_observation)
        return 0;
    auto const &o = t.observation;
    const float w = o.x2 - o.x1, h = o.y2 - o.y1;
    if (!std::isfinite(o.score) || o.score > 1 || o.score < c.min_detection_score || !std::isfinite(w) ||
        !std::isfinite(h) || w < c.min_face_width || h < c.min_face_height)
        return 0;
    for (auto p : o.landmarks)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < o.x1 - w * .25F || p.x > o.x2 + w * .25F ||
            p.y < o.y1 - h * .25F || p.y > o.y2 + h * .25F)
            return 0;
    const auto eye = std::hypot(o.landmarks[1].x - o.landmarks[0].x, o.landmarks[1].y - o.landmarks[0].y);
    if (eye < 1)
        return 0;
    try
    {
        const auto a = arcface::estimate_arcface_transform(o);
        if (!std::isfinite(a.rmse) || a.rmse > c.max_alignment_rmse)
            return 0;
        return std::min(w, h) / (1 + a.rmse);
    }
    catch (std::exception const &)
    {
        return 0;
    }
}
} // namespace
void validate_attendance_config(BiometricAttendanceConfig const &c)
{
    auto finite = [](float v) { return std::isfinite(v); };
    if (!finite(c.recognition_similarity_threshold) || c.recognition_similarity_threshold <= -1 ||
        c.recognition_similarity_threshold > 1)
        throw std::invalid_argument("Set an explicit recognition threshold in (-1,1]");
    if (c.calibration_id.empty() && !c.allow_uncalibrated_thresholds)
        throw std::invalid_argument("calibration_id required; development tools must explicitly opt in");
    if (!finite(c.recognition_min_margin) || c.recognition_min_margin < 0 || c.recognition_min_margin > 2 ||
        !finite(c.pad.live_threshold) || c.pad.live_threshold < 0 || c.pad.live_threshold > 1 ||
        !finite(c.min_detection_score) || c.min_detection_score < 0 || c.min_detection_score > 1 ||
        !finite(c.min_face_width) || !finite(c.min_face_height) || c.min_face_width <= 0 || c.min_face_height <= 0 ||
        !finite(c.max_alignment_rmse) || c.max_alignment_rmse <= 0 || !finite(c.retry_quality_gain) ||
        c.retry_quality_gain <= 1)
        throw std::invalid_argument("Invalid biometric quality/threshold configuration");
    if (!c.pad_bona_fide_votes_required || !c.pad_attack_votes_required || !c.pad_max_attempts ||
        c.pad_bona_fide_votes_required > c.pad_max_attempts || c.pad_attack_votes_required > c.pad_max_attempts ||
        !c.pad_every_frames || !c.recognition_every_frames || !c.recognition_max_attempts ||
        !c.max_missing_runtime_frames || c.event_cooldown.count() <= 0 || c.evidence_ttl.count() <= 0)
        throw std::invalid_argument("Invalid biometric cadence/budget");
}
char const *to_string(Phase p) noexcept
{
    switch (p)
    {
    case Phase::WaitingQuality:
        return "WAITING_QUALITY";
    case Phase::CheckingPad:
        return "CHECKING_PAD";
    case Phase::Recognizing:
        return "RECOGNIZING";
    case Phase::RejectedAttack:
        return "REJECTED_ATTACK";
    case Phase::Recognized:
        return "RECOGNIZED";
    case Phase::DuplicateIdentity:
        return "DUPLICATE_IDENTITY";
    case Phase::RecognitionDeferred:
        return "RECOGNITION_DEFERRED";
    case Phase::UnknownIdentity:
        return "UNKNOWN_IDENTITY";
    }
    return "UNKNOWN";
}
struct AttendanceProcessor::Impl
{
    struct State
    {
        Phase phase = Phase::WaitingQuality;
        std::size_t seen = 0, last_pad = 0, last_arc = 0, attempts = 0, extra = 0, count = 0;
        Time evidence{}, terminal{};
        liveness::VoteCounts votes;
        std::vector<float> scores;
        std::array<double, 512> sum{};
        float best_quality = 0;
        GalleryMatch match;
    };
    BiometricAttendanceConfig cfg;
    BiometricEvaluators evaluate;
    GalleryMatcher gallery;
    std::unordered_map<std::uint64_t, State> states;
    std::unordered_map<std::string, Time> emitted;
    std::size_t tick = 0;
    std::uint64_t last_frame = 0;
    Time last_time{};
    bool have_time = false;
    Impl(BiometricAttendanceConfig c, BiometricEvaluators e) : cfg(std::move(c)), evaluate(std::move(e))
    {
        validate_attendance_config(cfg);
        if (!evaluate.pad || !evaluate.recognize)
            throw std::invalid_argument("Missing biometric evaluator");
    }
};
AttendanceProcessor::AttendanceProcessor(BiometricAttendanceConfig c, BiometricEvaluators e)
    : impl_(std::make_unique<Impl>(std::move(c), std::move(e)))
{
}
AttendanceProcessor::~AttendanceProcessor() = default;
void AttendanceProcessor::set_gallery(std::vector<GalleryTemplate> const &g)
{
    impl_->gallery.replace(g);
    reset();
}
std::size_t AttendanceProcessor::gallery_size() const
{
    return impl_->gallery.size();
}
void AttendanceProcessor::reset()
{
    impl_->states.clear();
    impl_->tick = 0;
    impl_->last_frame = 0;
}
BiometricAttendanceFrame AttendanceProcessor::process(FaceTrackingFrame const &f, Time now)
{
    auto &p = *impl_;
    auto const &c = p.cfg;
    if ((p.have_time && now < p.last_time) || f.gpu_frame.frame_id <= p.last_frame)
        throw std::invalid_argument("Frame/time must advance monotonically");
    std::unordered_set<std::uint64_t> unique;
    for (auto const &t : f.tracks)
        if (!unique.insert(t.track_id).second)
            throw std::invalid_argument("Duplicate track ID in frame");
    p.last_time = now;
    p.have_time = true;
    p.last_frame = f.gpu_frame.frame_id;
    ++p.tick;
    for (auto it = p.states.begin(); it != p.states.end();)
    {
        if (p.tick - it->second.seen > c.max_missing_runtime_frames)
            it = p.states.erase(it);
        else
            ++it;
    }
    for (auto it = p.emitted.begin(); it != p.emitted.end();)
    {
        if (now - it->second >= c.event_cooldown)
            it = p.emitted.erase(it);
        else
            ++it;
    }
    BiometricAttendanceFrame out;
    out.gpu_frame = f.gpu_frame;
    out.timing.tracking_ms = f.timing.pipeline_total_ms;
    std::vector<FaceDetection> pad_obs, arc_obs;
    std::vector<std::uint64_t> pad_ids, arc_ids;
    for (auto const &t : f.tracks)
    {
        auto &s = p.states[t.track_id];
        s.seen = p.tick;
        if ((s.phase == Phase::Recognized || s.phase == Phase::DuplicateIdentity) &&
            now - s.terminal >= c.event_cooldown)
        {
            s = {};
            s.seen = p.tick;
        }
        if (s.phase == Phase::Recognized || s.phase == Phase::DuplicateIdentity || s.phase == Phase::RejectedAttack ||
            s.phase == Phase::UnknownIdentity)
            continue;
        if (s.votes.attempts && now - s.evidence > c.evidence_ttl)
        {
            auto seen = s.seen;
            s = {};
            s.seen = seen;
        }
        const float q = quality(t, c);
        if (q <= 0 || !p.gallery.size())
            continue;
        if (s.phase == Phase::RecognitionDeferred)
        {
            if (s.extra >= c.recognition_extra_attempts || q < s.best_quality * c.retry_quality_gain)
                continue;
            ++s.extra;
            s.phase = Phase::Recognizing;
            s.sum = {};
            s.count = 0;
        }
        if (s.phase != Phase::Recognizing && (!s.last_pad || p.tick - s.last_pad >= c.pad_every_frames))
        {
            s.phase = Phase::CheckingPad;
            pad_obs.push_back(t.observation);
            pad_ids.push_back(t.track_id);
        }
    }
    if (!pad_obs.empty())
    {
        auto begin = std::chrono::steady_clock::now();
        auto results = p.evaluate.pad(f.gpu_frame, pad_obs);
        out.timing.pad_batch_wall_ms = elapsed(begin);
        if (results.size() != pad_ids.size())
            throw std::runtime_error("PAD result count mismatch");
        for (std::size_t i = 0; i < results.size(); ++i)
        {
            auto &s = p.states.at(pad_ids[i]);
            auto const &r = results[i];
            s.last_pad = p.tick;
            if (r.decision == liveness::LivenessDecision::InputRejected)
            {
                if (r.live_score)
                    throw std::runtime_error("Rejected PAD input has score");
                s.votes.add(r.decision);
                s.phase = Phase::WaitingQuality;
                continue;
            }
            if (!r.live_score)
                throw std::runtime_error("PAD result missing score");
            const auto decision = liveness::classify_score(*r.live_score, c.pad.live_threshold);
            if (!s.votes.attempts)
                s.evidence = now;
            s.votes.add(decision);
            s.scores.push_back(*r.live_score);
            if (s.votes.spoof >= c.pad_attack_votes_required)
                s.phase = Phase::RejectedAttack;
            else if (s.votes.real >= c.pad_bona_fide_votes_required)
                s.phase = Phase::Recognizing;
            else if (s.votes.attempts >= c.pad_max_attempts)
                s.phase = Phase::RejectedAttack;
        }
    }
    for (auto const &t : f.tracks)
    {
        auto &s = p.states.at(t.track_id);
        const auto q = quality(t, c);
        if (s.phase != Phase::Recognizing || q <= 0 || (s.last_arc && p.tick - s.last_arc < c.recognition_every_frames))
            continue;
        s.best_quality = std::max(s.best_quality, q);
        arc_obs.push_back(t.observation);
        arc_ids.push_back(t.track_id);
    }
    if (!arc_obs.empty())
    {
        auto begin = std::chrono::steady_clock::now();
        auto results = p.evaluate.recognize(f.gpu_frame, arc_obs);
        out.timing.arcface_batch_wall_ms = elapsed(begin);
        if (results.size() != arc_ids.size())
            throw std::runtime_error("Recognition result count mismatch");
        for (std::size_t i = 0; i < results.size(); ++i)
        {
            auto &s = p.states.at(arc_ids[i]);
            s.last_arc = p.tick;
            ++s.attempts;
            ++s.count;
            for (std::size_t j = 0; j < 512; ++j)
                s.sum[j] += results[i].embedding[j];
            std::array<float, 512> mean{};
            for (std::size_t j = 0; j < 512; ++j)
                mean[j] = static_cast<float>(s.sum[j] / s.count);
            s.match = p.gallery.match(mean);
            if (s.match.valid && s.match.best_similarity >= c.recognition_similarity_threshold &&
                s.match.margin >= c.recognition_min_margin)
            {
                s.terminal = now;
                if (p.emitted.contains(s.match.identity_id))
                    s.phase = Phase::DuplicateIdentity;
                else
                {
                    p.emitted[s.match.identity_id] = now;
                    s.phase = Phase::Recognized;
                    AttendanceEvent e;
                    e.occurred_at_utc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::system_clock::now().time_since_epoch())
                                               .count();
                    e.camera_frame_id = f.gpu_frame.frame_id;
                    e.track_id = arc_ids[i];
                    e.identity_id = s.match.identity_id;
                    e.similarity = s.match.best_similarity;
                    e.similarity_margin = s.match.margin;
                    e.pad_median_p_real = median(s.scores);
                    out.attendance_events.push_back(std::move(e));
                }
            }
            else if (s.attempts >= c.recognition_max_attempts + s.extra)
            {
                s.phase = s.extra < c.recognition_extra_attempts ? Phase::RecognitionDeferred : Phase::UnknownIdentity;
            }
        }
    }
    for (auto const &t : f.tracks)
    {
        auto const &s = p.states.at(t.track_id);
        BiometricTrackSnapshot snap;
        snap.track_id = t.track_id;
        snap.phase = s.phase;
        snap.current_detection_score = t.has_observation ? t.observation.score : 0;
        snap.current_face_width = t.observation.x2 - t.observation.x1;
        snap.current_face_height = t.observation.y2 - t.observation.y1;
        snap.pad_attempts = s.votes.attempts;
        snap.pad_input_rejections = s.votes.input_rejected;
        snap.pad_bona_fide_votes = s.votes.real;
        snap.pad_attack_votes = s.votes.spoof;
        snap.pad_median_p_real = median(s.scores);
        snap.recognition_attempts = s.attempts;
        snap.identity_id = s.match.identity_id;
        snap.identity_similarity = s.match.best_similarity;
        snap.identity_margin = s.match.margin;
        out.tracks.push_back(std::move(snap));
    }
    return out;
}
} // namespace vision_runtime::pipeline
