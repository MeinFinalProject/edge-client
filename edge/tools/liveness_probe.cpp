#include "liveness/liveness.hpp"
#include "pipeline/face_tracking_runtime.hpp"

#include <Windows.h>
#include <winrt/base.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using vision_runtime::liveness::LivenessDecision;
using vision_runtime::liveness::LivenessDetector;
using vision_runtime::liveness::LivenessResult;
using vision_runtime::pipeline::FaceTrackingRuntime;

struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

Stats summarize(std::vector<double> values) {
    Stats out{};
    if (values.empty()) return out;

    out.mean = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());

    std::sort(values.begin(), values.end());
    const auto n = values.size();
    out.median = (n % 2 == 0)
        ? 0.5 * (values[n / 2 - 1] + values[n / 2])
        : values[n / 2];

    const std::size_t p95_idx = std::min(
        n - 1,
        static_cast<std::size_t>(std::ceil(n * 0.95) - 1)
    );
    out.p95 = values[p95_idx];
    out.min = values.front();
    out.max = values.back();
    return out;
}

std::filesystem::path executable_directory() {
    std::wstring buffer(32768, L'\0');
    const DWORD n = GetModuleFileNameW(
        nullptr,
        buffer.data(),
        static_cast<DWORD>(buffer.size())
    );
    if (n == 0 || n >= buffer.size()) {
        throw std::runtime_error("GetModuleFileNameW failed");
    }
    buffer.resize(n);
    return std::filesystem::path{buffer}.parent_path();
}

std::filesystem::path find_upwards(
    std::filesystem::path root,
    std::filesystem::path const& rel
) {
    for (int i = 0; i < 12; ++i) {
        auto candidate = root / rel;
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate).lexically_normal();
        }
        const auto parent = root.parent_path();
        if (parent.empty() || parent == root) break;
        root = parent;
    }
    return {};
}

struct Args {
    std::filesystem::path scrfd_model;
    std::filesystem::path liveness_model;
    std::string scenario = "live";
    float min_detection_score = 0.50F;
    int check_every = 5;
    int measure_frames = 300;
};

Args parse_args(int argc, char** argv) {
    Args args{};

    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];

        auto need = [&](char const* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    std::string{"Missing value for "} + name
                );
            }
            return argv[++i];
        };

        if (token == "--scrfd") {
            args.scrfd_model = std::filesystem::absolute(need("--scrfd"));
        } else if (token == "--liveness") {
            args.liveness_model = std::filesystem::absolute(need("--liveness"));
        } else if (token == "--scenario") {
            args.scenario = need("--scenario");
        } else if (token == "--min-det-score") {
            args.min_detection_score = std::stof(need("--min-det-score"));
        } else if (token == "--check-every") {
            args.check_every = std::stoi(need("--check-every"));
        } else if (token == "--frames") {
            args.measure_frames = std::stoi(need("--frames"));
        } else if (token == "--help" || token == "-h") {
            std::cout
                << "liveness_probe options:\n"
                << "  --scrfd <det_500m.onnx>\n"
                << "  --liveness <liveness.onnx>\n"
                << "  --scenario <live|photo|display>  label only\n"
                << "  --min-det-score <float>          default 0.50\n"
                << "  --check-every <frames>           default 5\n"
                << "  --frames <count>                 default 300\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + token);
        }
    }

    const auto cwd = std::filesystem::current_path();
    const auto exe = executable_directory();

    if (args.scrfd_model.empty()) {
        const auto rel = std::filesystem::path{
            L"models\\insightface\\buffalo_sc\\det_500m.onnx"
        };
        args.scrfd_model = find_upwards(cwd, rel);
        if (args.scrfd_model.empty()) args.scrfd_model = find_upwards(exe, rel);
    }

    if (args.liveness_model.empty()) {
        const auto rel = std::filesystem::path{
            L"models\\insightface\\addons\\liveness.onnx"
        };
        args.liveness_model = find_upwards(cwd, rel);
        if (args.liveness_model.empty()) args.liveness_model = find_upwards(exe, rel);
    }

    if (args.scrfd_model.empty() || !std::filesystem::exists(args.scrfd_model)) {
        throw std::runtime_error("det_500m.onnx not found; pass --scrfd <path>");
    }
    if (args.liveness_model.empty() || !std::filesystem::exists(args.liveness_model)) {
        throw std::runtime_error(
            "liveness.onnx not found; pass --liveness <path>"
        );
    }
    if (
        args.scenario != "live" &&
        args.scenario != "photo" &&
        args.scenario != "display"
    ) {
        throw std::runtime_error("--scenario must be live, photo, or display");
    }
    if (
        !std::isfinite(args.min_detection_score) ||
        args.min_detection_score < 0.0F ||
        args.min_detection_score > 1.0F
    ) {
        throw std::runtime_error("--min-det-score must be 0..1");
    }
    if (args.check_every < 1) throw std::runtime_error("--check-every must be >= 1");
    if (args.measure_frames < 1) throw std::runtime_error("--frames must be >= 1");

    return args;
}

struct DecisionStats {
    std::size_t total = 0;
    std::size_t live = 0;
    std::size_t spoof = 0;
    std::size_t rejected = 0;
    std::vector<double> scores;
    std::vector<double> total_ms;
    std::vector<double> eval_ms;

    void add(LivenessResult const& r) {
        ++total;
        if (r.decision == LivenessDecision::InputRejected) { ++rejected; return; }
        if (r.decision == LivenessDecision::Live) ++live;
        else ++spoof;
        scores.push_back(r.live_score.value());
        total_ms.push_back(r.timing.total_ms);
        eval_ms.push_back(r.timing.evaluate_ms);
    }
};

} // namespace

int main(int argc, char** argv) {
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        const auto args = parse_args(argc, argv);

        vision_runtime::pipeline::FaceTrackingConfig tracking_cfg{};
        tracking_cfg.camera.width = 1280;
        tracking_cfg.camera.height = 720;
        tracking_cfg.camera.fps = 30;
        tracking_cfg.camera.source_subtype = L"NV12";
        tracking_cfg.camera.output_subtype = L"NV12";

        tracking_cfg.scrfd.model_path = args.scrfd_model;
        tracking_cfg.scrfd.input_width = 640;
        tracking_cfg.scrfd.input_height = 640;
        tracking_cfg.scrfd.score_threshold = 0.10F;
        tracking_cfg.scrfd.nms_threshold = 0.40F;

        tracking_cfg.bytetrack.track_threshold = 0.50F;
        tracking_cfg.bytetrack.low_threshold = 0.10F;
        tracking_cfg.bytetrack.new_track_threshold = 0.60F;
        tracking_cfg.bytetrack.match_threshold = 0.80F;
        tracking_cfg.bytetrack.second_match_threshold = 0.50F;
        tracking_cfg.bytetrack.unconfirmed_match_threshold = 0.70F;
        tracking_cfg.bytetrack.track_buffer = 30;
        tracking_cfg.bytetrack.frame_rate = 30;
        tracking_cfg.bytetrack.mot20 = false;

        vision_runtime::liveness::LivenessConfig liveness_cfg{};
        liveness_cfg.model_path = args.liveness_model;
        liveness_cfg.input_width = 80;
        liveness_cfg.input_height = 80;
        liveness_cfg.live_threshold = vision_runtime::liveness::kTaLiveThreshold;

        std::cout
            << "================================================================================\n"
            << "INSIGHTFACE LIVENESS GPU PROBE\n"
            << "================================================================================\n"
            << "Scenario label     : " << args.scenario << "\n"
            << "SCRFD model        : " << args.scrfd_model.string() << "\n"
            << "Liveness model     : " << args.liveness_model.string() << "\n"
            << "Camera             : 1280x720 @30 NV12\n"
            << "Tracking           : SCRFD DirectML GPU -> ByteTrack native CPU\n"
            << "Liveness backbone  : WinML DirectML GPU\n"
            << "Liveness input     : 1x3x80x80 NCHW RGB float32\n"
            << "Liveness prep      : 5-point affine warp (ArcFace style)\n"
            << "Liveness norm      : pixel / 255.0 (implicit in NV12->RGB)\n"
            << "Liveness output    : live_score (float32)\n"
            << "Quality gate       : SCRFD score >= " << args.min_detection_score << "\n"
            << "Check cadence      : every " << args.check_every << " tracking frames\n"
            << "Decision threshold : " << liveness_cfg.live_threshold << " (default)\n\n";

        FaceTrackingRuntime runtime{tracking_cfg};
        LivenessDetector liveness{liveness_cfg};
        runtime.start();

        constexpr int kWarmup = 35;
        std::cout << "Warm-up            : " << kWarmup << " tracking frames...\n";
        for (int i = 0; i < kWarmup; ++i) {
            auto f = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!f) throw std::runtime_error("tracking warm-up timeout");
        }
        runtime.reset_tracker();

        DecisionStats stats;
        std::size_t sample_frames = 0;
        std::size_t quality_skips = 0;
        std::size_t no_track_frames = 0;
        std::size_t camera_skips = 0;
        std::uint64_t prev_cam = 0;

        const auto wall_begin = std::chrono::steady_clock::now();

        for (int i = 0; i < args.measure_frames; ++i) {
            auto frame = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!frame) throw std::runtime_error("Liveness probe tracking timeout");

            if (prev_cam != 0 && frame->gpu_frame.frame_id > prev_cam + 1) {
                camera_skips += static_cast<std::size_t>(
                    frame->gpu_frame.frame_id - prev_cam - 1
                );
            }
            prev_cam = frame->gpu_frame.frame_id;

            if (frame->tracks.empty()) {
                ++no_track_frames;
                continue;
            }
            if ((i % args.check_every) != 0) continue;

            std::vector<vision_runtime::FaceDetection> observations;
            std::vector<std::uint64_t> track_ids;

            for (auto const& t : frame->tracks) {
                if (!t.has_observation || t.observation.score < args.min_detection_score) {
                    ++quality_skips;
                    continue;
                }
                observations.push_back(t.observation);
                track_ids.push_back(t.track_id);
            }

            if (observations.empty()) continue;

            auto results = liveness.evaluate_many(frame->gpu_frame, observations);
            if (results.size() != observations.size()) {
                throw std::runtime_error("Liveness result count mismatch");
            }

            ++sample_frames;
            std::cout
                << "[CHK " << std::setw(3) << sample_frames << "] "
                << "cam=" << std::setw(5) << frame->gpu_frame.frame_id
                << " active=" << frame->tracks.size();

            for (std::size_t r = 0; r < results.size(); ++r) {
                auto const& out = results[r];
                stats.add(out);
                std::cout
                    << " | ID" << track_ids[r]
                    << " Score=" << std::fixed << std::setprecision(4)
                    << (out.live_score ? std::to_string(*out.live_score) : "n/a")
                    << " oob=" << out.out_of_bounds_ratio
                    << " " << vision_runtime::liveness::to_string(out.decision)
                    << " " << std::setprecision(3)
                    << out.timing.total_ms << "ms";
            }
            std::cout << "\n";
        }

        const auto wall_end = std::chrono::steady_clock::now();
        runtime.stop();

        const double wall_seconds =
            std::chrono::duration<double>(wall_end - wall_begin).count();
        const double tracking_fps = wall_seconds > 0.0
            ? static_cast<double>(args.measure_frames) / wall_seconds
            : 0.0;

        const auto score_stats = summarize(stats.scores);
        const auto total = summarize(stats.total_ms);
        const auto eval = summarize(stats.eval_ms);

        auto percentage = [](std::size_t part, std::size_t total_count) -> double {
            return total_count == 0
                ? 0.0
                : 100.0 * static_cast<double>(part) /
                  static_cast<double>(total_count);
        };

        std::cout
            << "\n================================================================================\n"
            << "RESULT - " << args.scenario << " - Liveness GPU\n"
            << "================================================================================\n"
            << "Tracking throughput          : " << std::fixed << std::setprecision(2)
            << tracking_fps << " FPS\n"
            << "Camera frame IDs skipped     : " << camera_skips << "\n"
            << "Frames without active track  : " << no_track_frames << "\n"
            << "Liveness sample frames       : " << sample_frames << "\n"
            << "Quality-skipped observations : " << quality_skips << "\n"
            << "Liveness evaluations         : " << stats.total << "\n\n"
            << "Input rejected (no score)    : " << stats.rejected << "\n"
            << std::setprecision(6)
            << "Score mean/med/min/max       : "
            << score_stats.mean << " / " << score_stats.median << " / " << score_stats.min << " / " << score_stats.max << "\n"
            << "LIVE / SPOOF votes           : "
            << stats.live << " / " << stats.spoof
            << "  (" << std::setprecision(1)
            << percentage(stats.live, stats.live + stats.spoof) << "% / "
            << percentage(stats.spoof, stats.live + stats.spoof) << "%, accepted inputs only)\n"
            << std::setprecision(3)
            << "Total median / P95           : "
            << total.median << " / " << total.p95 << " ms\n"
            << "WinML Evaluate median / P95  : "
            << eval.median << " / " << eval.p95 << " ms\n\n"
            << "Gate:\n"
            << "- TA decision: LIVE iff score >= " << vision_runtime::liveness::kTaLiveThreshold << ".\n"
            << "- Input rejection has no score and is not a SPOOF vote.\n";

        std::cout << "\nPress ENTER to close...\n";
        std::cin.get();
        return 0;
    } catch (winrt::hresult_error const& e) {
        std::cerr
            << "WinRT ERROR 0x" << std::hex << std::uppercase
            << static_cast<std::uint32_t>(e.code()) << std::dec
            << ": " << winrt::to_string(e.message())
            << "\nPress ENTER to close...\n";
        std::cin.get();
        return 2;
    } catch (std::exception const& e) {
        std::cerr
            << "ERROR: " << e.what()
            << "\nPress ENTER to close...\n";
        std::cin.get();
        return 1;
    }
}
