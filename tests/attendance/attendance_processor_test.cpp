#include "pipeline/attendance_processor.hpp"
#include "pipeline/gallery_matcher.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace vision_runtime;
using namespace vision_runtime::pipeline;
namespace
{
void check(bool ok, char const *why)
{
    if (!ok)
        throw std::runtime_error(why);
}
template <class F> void rejects(F f)
{
    bool caught = false;
    try
    {
        f();
    }
    catch (std::exception const &)
    {
        caught = true;
    }
    check(caught, "Expected rejection");
}
std::array<float, 512> basis(int index)
{
    std::array<float, 512> x{};
    x[index] = 1;
    return x;
}
FaceTrackingFrame frame(std::uint64_t id, std::uint64_t track = 1, float scale = 1)
{
    FaceTrackingFrame f;
    f.gpu_frame.frame_id = id;
    bytetrack::TrackResult t;
    t.track_id = track;
    t.has_observation = true;
    auto &o = t.observation;
    o.x1 = 0;
    o.y1 = 0;
    o.x2 = 112 * scale;
    o.y2 = 112 * scale;
    o.score = .9F;
    o.landmarks = {Point2f{38.2946F, 51.6963F}, Point2f{73.5318F, 51.5014F}, Point2f{56.0252F, 71.7366F},
                   Point2f{41.5493F, 92.3655F}, Point2f{70.7299F, 92.2041F}};
    for (auto &p : o.landmarks)
    {
        p.x *= scale;
        p.y *= scale;
    }
    f.tracks.push_back(t);
    return f;
}
struct Fixture
{
    int pad_calls = 0, arc_calls = 0, embedding = 0;
    float score = 1, embedding_sign = 1;
    bool rejected = false;
    BiometricAttendanceConfig config()
    {
        BiometricAttendanceConfig c;
        c.recognition_similarity_threshold = .75F;
        c.allow_uncalibrated_thresholds = true;
        c.pad_every_frames = 1;
        c.recognition_every_frames = 1;
        return c;
    }
    BiometricEvaluators evaluators()
    {
        return {[this](auto const &, auto const &faces) {
                    pad_calls += static_cast<int>(faces.size());
                    std::vector<liveness::LivenessResult> out(faces.size());
                    for (auto &r : out)
                    {
                        if (!rejected)
                        {
                            r.live_score = score;
                            r.decision = liveness::classify_score(score);
                        }
                    }
                    return out;
                },
                [this](auto const &, auto const &faces) {
                    arc_calls += static_cast<int>(faces.size());
                    std::vector<arcface::ArcFaceResult> out(faces.size());
                    for (auto &r : out)
                    {
                        r.embedding = basis(embedding);
                        r.embedding[embedding] *= embedding_sign;
                    }
                    return out;
                }};
    }
};
} // namespace
int main()
{
    try
    {
        using namespace std::chrono_literals;
        auto now = AttendanceProcessor::Time{} + 1s;
        Fixture x;
        rejects([&] { AttendanceProcessor invalid({}, x.evaluators()); });
        AttendanceProcessor p(x.config(), x.evaluators());
        p.set_gallery({{"A", basis(0)}});
        auto no = frame(1);
        no.tracks[0].has_observation = false;
        (void)p.process(no, now);
        check(x.pad_calls == 0, "Prediction must not infer");
        auto bad = frame(2);
        bad.tracks[0].observation.landmarks[0].x = std::numeric_limits<float>::quiet_NaN();
        (void)p.process(bad, now);
        check(x.pad_calls == 0, "Invalid landmarks must not infer");
        x.rejected = true;
        auto r = p.process(frame(3), now);
        check(r.tracks[0].pad_attempts == 0 && r.tracks[0].pad_input_rejections == 1, "Rejected crop consumed vote");
        x.rejected = false;
        (void)p.process(frame(4), now);
        check(x.arc_calls == 0, "ArcFace before two PAD votes");
        r = p.process(frame(5), now);
        check(r.attendance_events.size() == 1 && r.attendance_events[0].occurred_at_utc_ms > 0,
              "Expected timestamped event");
        rejects([&] { (void)p.process(frame(5), now); });
        (void)p.process(frame(6, 2), now + 1s);
        r = p.process(frame(7, 2), now + 1s);
        check(r.attendance_events.empty() && r.tracks[0].phase == BiometricTrackPhase::DuplicateIdentity,
              "Track fragmentation bypassed cooldown");
        p.set_gallery({{"A", basis(0)}});
        (void)p.process(frame(8, 3), now + 2s);
        r = p.process(frame(9, 3), now + 2s);
        check(r.attendance_events.empty(), "Gallery reload cleared cooldown");
        (void)p.process(frame(10, 4), now + 31s);
        r = p.process(frame(11, 4), now + 31s);
        check(r.attendance_events.size() == 1, "Cooldown became permanent dedupe");
        p.set_gallery({});
        r = p.process(frame(12, 5), now + 32s);
        check(r.attendance_events.empty(), "Empty gallery must wait");

        Fixture y;
        y.score = 0;
        AttendanceProcessor spoof(y.config(), y.evaluators());
        spoof.set_gallery({{"A", basis(0)}});
        (void)spoof.process(frame(1), now);
        r = spoof.process(frame(2), now);
        check(y.arc_calls == 0 && r.tracks[0].phase == BiometricTrackPhase::RejectedAttack,
              "Spoof reached recognition");
        Fixture z;
        z.embedding = 1;
        AttendanceProcessor retry(z.config(), z.evaluators());
        retry.set_gallery({{"A", basis(0)}});
        for (int i = 1; i <= 4; ++i)
            r = retry.process(frame(i), now);
        check(r.tracks[0].phase == BiometricTrackPhase::RecognitionDeferred && z.arc_calls == 3,
              "Expected bounded initial attempts");
        (void)retry.process(frame(5), now);
        check(z.arc_calls == 3, "Unchanged quality retried");
        z.embedding = 0;
        r = retry.process(frame(6, 1, 2), now);
        check(r.attendance_events.size() == 1 && z.arc_calls == 4, "Improved quality failed fresh retry");

        Fixture multi;
        AttendanceProcessor m(multi.config(), multi.evaluators());
        m.set_gallery({{"A", basis(0)}});
        auto three = frame(1);
        three.tracks.push_back(frame(1, 2).tracks[0]);
        three.tracks.push_back(frame(1, 3).tracks[0]);
        (void)m.process(three, now);
        three.gpu_frame.frame_id = 2;
        r = m.process(three, now);
        check(r.tracks.size() == 3 && r.attendance_events.size() == 1 && multi.arc_calls == 3,
              "Multi-track state or same-frame dedupe failed");
        Fixture reload;
        AttendanceProcessor g(reload.config(), reload.evaluators());
        g.set_gallery({{"A", basis(0)}});
        (void)g.process(frame(1), now);
        g.set_gallery({{"B", basis(0)}});
        r = g.process(frame(2), now);
        check(r.attendance_events.empty(), "Reload retained stale evidence");
        r = g.process(frame(3), now);
        check(r.attendance_events.at(0).identity_id == "B", "New gallery not applied");
        GalleryMatcher matcher;
        matcher.replace({{"A", basis(0)}, {"A", basis(1)}, {"B", basis(2)}});
        auto match = matcher.match(basis(0));
        check(match.identity_id == "A" && match.margin == 1, "Margin compared templates instead of identities");
        auto invalid = basis(0);
        invalid[3] = std::numeric_limits<float>::infinity();
        rejects([&] { matcher.replace({{"C", invalid}}); });
        check(matcher.match(basis(0)).identity_id == "A", "Failed gallery replacement damaged old gallery");
        check(!matcher.match({}).valid, "Zero aggregate must not produce an identity");
        rejects([&] { (void)matcher.match(invalid); });
        Fixture cancellation;
        cancellation.embedding = 1;
        AttendanceProcessor cancel(cancellation.config(), cancellation.evaluators());
        cancel.set_gallery({{"A", basis(0)}});
        (void)cancel.process(frame(1), now);
        (void)cancel.process(frame(2), now);
        cancellation.embedding_sign = -1;
        r = cancel.process(frame(3), now);
        check(r.attendance_events.empty() && r.tracks[0].recognition_attempts == 2 &&
                  r.tracks[0].phase == BiometricTrackPhase::Recognizing,
              "Cancelling valid embeddings crashed or bypassed the retry budget");
        cancellation.embedding = 0;
        cancellation.embedding_sign = 1;
        r = cancel.process(frame(4), now);
        check(r.attendance_events.size() == 1 && r.tracks[0].recognition_attempts == 3,
              "Recognition could not recover from an inconclusive aggregate");
        Fixture boundary;
        auto cfg = boundary.config();
        cfg.recognition_similarity_threshold = 1;
        cfg.recognition_min_margin = 1;
        boundary.score = cfg.pad.live_threshold;
        AttendanceProcessor equal(cfg, boundary.evaluators());
        equal.set_gallery({{"A", basis(0)}, {"B", basis(1)}});
        (void)equal.process(frame(1), now);
        r = equal.process(frame(2), now);
        check(r.attendance_events.size() == 1, "Threshold or margin equality rejected");
        Fixture expiry;
        AttendanceProcessor ttl(expiry.config(), expiry.evaluators());
        ttl.set_gallery({{"A", basis(0)}});
        (void)ttl.process(frame(1), now);
        r = ttl.process(frame(2), now + 4s);
        check(r.attendance_events.empty() && r.tracks[0].pad_attempts == 1, "Expired PAD evidence reused");
        BiometricEvaluators mixed_models{[](auto const &, auto const &faces) {
                                             std::vector<liveness::LivenessResult> result;
                                             for (auto const &face : faces)
                                             {
                                                 liveness::LivenessResult r;
                                                 r.live_score = face.score == .92F ? 0.F : 1.F;
                                                 r.decision = liveness::classify_score(*r.live_score);
                                                 result.push_back(r);
                                             }
                                             return result;
                                         },
                                         [](auto const &, auto const &faces) {
                                             std::vector<arcface::ArcFaceResult> result;
                                             for (auto const &face : faces)
                                             {
                                                 arcface::ArcFaceResult r;
                                                 r.embedding = basis(face.score == .93F ? 1 : 0);
                                                 result.push_back(r);
                                             }
                                             return result;
                                         }};
        AttendanceProcessor mixed(x.config(), mixed_models);
        mixed.set_gallery({{"A", basis(0)}});
        auto group = frame(1);
        group.tracks.push_back(frame(1, 2).tracks[0]);
        group.tracks.push_back(frame(1, 3).tracks[0]);
        group.tracks[0].observation.score = .91F;
        group.tracks[1].observation.score = .92F;
        group.tracks[2].observation.score = .93F;
        for (int i = 1; i <= 4; ++i)
        {
            group.gpu_frame.frame_id = i;
            r = mixed.process(group, now);
        }
        check(r.tracks[0].phase == BiometricTrackPhase::Recognized &&
                  r.tracks[1].phase == BiometricTrackPhase::RejectedAttack &&
                  r.tracks[2].phase == BiometricTrackPhase::RecognitionDeferred,
              "Three simultaneous tracks leaked evidence across identities");
        auto bigger = frame(5, 3, 2);
        bigger.tracks[0].observation.score = .93F;
        r = mixed.process(bigger, now);
        check(r.tracks[0].phase == BiometricTrackPhase::UnknownIdentity && r.tracks[0].recognition_attempts == 4,
              "Unsuccessful extra retry did not stop");
        bigger.gpu_frame.frame_id = 6;
        r = mixed.process(bigger, now);
        check(r.tracks[0].recognition_attempts == 4, "Extra recognition budget was unbounded");
        std::cout << "Attendance state, quality, PAD, retry, cooldown, multi-track, gallery: PASS\n";
        return 0;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
