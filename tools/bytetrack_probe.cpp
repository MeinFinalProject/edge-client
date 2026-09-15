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
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using vision_runtime::pipeline::FaceTrackingFrame;
using vision_runtime::pipeline::FaceTrackingRuntime;

struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
};

Stats summarize(std::vector<double> values) {
    Stats s{};
    if (values.empty()) return s;
    double sum = 0.0;
    for (double v : values) sum += v;
    s.mean = sum / static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    s.median = (n % 2 == 0) ? 0.5 * (values[n/2 - 1] + values[n/2]) : values[n/2];
    const std::size_t p95_index = std::min(n - 1, static_cast<std::size_t>(std::ceil(n * 0.95) - 1));
    s.p95 = values[p95_index];
    return s;
}

std::filesystem::path executable_directory() {
    std::wstring buffer(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (n == 0 || n >= buffer.size()) throw std::runtime_error("GetModuleFileNameW failed");
    buffer.resize(n);
    return std::filesystem::path{buffer}.parent_path();
}

std::filesystem::path find_model(std::filesystem::path root) {
    const auto rel = std::filesystem::path{L"models\\insightface\\buffalo_sc\\det_500m.onnx"};
    for (int i = 0; i < 10; ++i) {
        auto candidate = root / rel;
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate).lexically_normal();
        }
        auto parent = root.parent_path();
        if (parent.empty() || parent == root) break;
        root = parent;
    }
    return {};
}

std::filesystem::path model_path_from_args(int argc, char** argv) {
    if (argc >= 2) {
        auto p = std::filesystem::absolute(argv[1]);
        if (!std::filesystem::exists(p)) throw std::runtime_error("model path argument does not exist");
        return p;
    }
    if (auto p = find_model(std::filesystem::current_path()); !p.empty()) return p;
    if (auto p = find_model(executable_directory()); !p.empty()) return p;
    throw std::runtime_error("det_500m.onnx not found; pass absolute path as argv[1]");
}

std::size_t count_high(std::vector<vision_runtime::FaceDetection> const& detections, float threshold) {
    return static_cast<std::size_t>(std::count_if(
        detections.begin(), detections.end(),
        [threshold](auto const& d) { return d.score > threshold; }
    ));
}

std::size_t count_low(
    std::vector<vision_runtime::FaceDetection> const& detections,
    float low_threshold,
    float track_threshold
) {
    return static_cast<std::size_t>(std::count_if(
        detections.begin(), detections.end(),
        [=](auto const& d) { return d.score > low_threshold && d.score < track_threshold; }
    ));
}

void print_tracks(FaceTrackingFrame const& frame, float track_threshold) {
    if (frame.tracks.empty()) {
        std::cout << "tracks=[]";
        return;
    }
    std::cout << "tracks=[";
    for (std::size_t i = 0; i < frame.tracks.size(); ++i) {
        auto const& t = frame.tracks[i];
        if (i) std::cout << ", ";
        const bool low_recovery = t.has_observation && t.observation.score < track_threshold;
        std::cout << "ID" << t.track_id
                  << " score=" << std::fixed << std::setprecision(3) << t.score
                  << (low_recovery ? " LOW" : " HIGH")
                  << " box=[" << std::setprecision(0)
                  << t.x1 << "," << t.y1 << "->" << t.x2 << "," << t.y2 << "]"
                  << " lm5=" << (t.has_observation ? "yes" : "no");
    }
    std::cout << "]";
}

} // namespace

int main(int argc, char** argv) {
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        const auto model = model_path_from_args(argc, argv);

        vision_runtime::pipeline::FaceTrackingConfig cfg{};
        cfg.camera.width = 1280;
        cfg.camera.height = 720;
        cfg.camera.fps = 30;
        cfg.camera.source_subtype = L"NV12";
        cfg.camera.output_subtype = L"NV12";

        cfg.scrfd.model_path = model;
        cfg.scrfd.input_width = 640;
        cfg.scrfd.input_height = 640;
        cfg.scrfd.score_threshold = 0.10F; // preserve ByteTrack low-score candidates
        cfg.scrfd.nms_threshold = 0.40F;

        cfg.bytetrack.track_threshold = 0.50F;
        cfg.bytetrack.low_threshold = 0.10F;
        cfg.bytetrack.new_track_threshold = 0.60F;
        cfg.bytetrack.match_threshold = 0.80F;
        cfg.bytetrack.second_match_threshold = 0.50F;
        cfg.bytetrack.unconfirmed_match_threshold = 0.70F;
        cfg.bytetrack.track_buffer = 30;
        cfg.bytetrack.frame_rate = 30;
        cfg.bytetrack.mot20 = false;

        std::cout << "========================================================\n";
        std::cout << "NATIVE BYTE TRACK LIVE PROBE v4.2\n";
        std::cout << "========================================================\n";
        std::cout << "Model             : " << model.string() << "\n";
        std::cout << "Camera            : 1280x720 @30 NV12 (validated native path)\n";
        std::cout << "SCRFD             : original det_500m.onnx, 640x640, GPU/DirectML\n";
        std::cout << "SCRFD candidates  : score > 0.10 for ByteTrack low-score association\n";
        std::cout << "ByteTrack         : native C++ tracking\n";
        std::cout << "Image round-trip  : no OpenCV/NumPy production image path\n";
        std::cout << "Downstream bridge : current GpuFrame + tracked SCRFD 5-point landmarks preserved\n\n";

        FaceTrackingRuntime runtime{cfg};
        runtime.start();

        constexpr int kWarmFrames = 35;
        std::cout << "Warm-up           : " << kWarmFrames << " full pipeline frames...\n";
        for (int i = 0; i < kWarmFrames; ++i) {
            auto result = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!result) throw std::runtime_error("full-pipeline warm-up timeout");
        }
        runtime.reset_tracker();

        constexpr int kMeasureFrames = 300;
        std::cout << "Measure           : " << kMeasureFrames << " frames (~10 s).\n";
        std::cout << "Move naturally. Jika bisa, buat wajah cepat/menyamping agar low-score recovery muncul.\n\n";

        std::vector<double> detector_ms;
        std::vector<double> tracker_ms;
        std::vector<double> association_ms;
        std::vector<double> pipeline_ms;
        detector_ms.reserve(kMeasureFrames);
        tracker_ms.reserve(kMeasureFrames);
        association_ms.reserve(kMeasureFrames);
        pipeline_ms.reserve(kMeasureFrames);

        std::set<std::uint64_t> unique_track_ids;
        std::map<std::uint64_t, std::size_t> track_output_frames;
        std::size_t total_candidates = 0;
        std::size_t total_high_candidates = 0;
        std::size_t total_low_candidates = 0;
        std::size_t total_track_outputs = 0;
        std::size_t total_low_recovery_outputs = 0;
        std::size_t max_active_tracks = 0;
        std::size_t max_candidates = 0;
        std::size_t skipped_camera_ids = 0;
        std::size_t frames_with_tracks = 0;
        std::size_t frames_with_low_candidates = 0;
        std::size_t frames_with_low_recovery = 0;
        std::size_t missing_track_observations = 0;

        std::uint64_t previous_frame_id = 0;
        const auto wall_begin = Clock::now();

        for (int i = 0; i < kMeasureFrames; ++i) {
            auto result = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!result) throw std::runtime_error("full-pipeline measurement timeout");
            auto const& f = *result;

            if (previous_frame_id != 0 && f.gpu_frame.frame_id > previous_frame_id + 1) {
                skipped_camera_ids += static_cast<std::size_t>(f.gpu_frame.frame_id - previous_frame_id - 1);
            }
            previous_frame_id = f.gpu_frame.frame_id;

            const auto high = count_high(f.detections, cfg.bytetrack.track_threshold);
            const auto low = count_low(f.detections, cfg.bytetrack.low_threshold, cfg.bytetrack.track_threshold);

            total_candidates += f.detections.size();
            total_high_candidates += high;
            total_low_candidates += low;
            max_candidates = std::max(max_candidates, f.detections.size());
            max_active_tracks = std::max(max_active_tracks, f.tracks.size());
            if (!f.tracks.empty()) ++frames_with_tracks;
            if (low > 0) ++frames_with_low_candidates;

            bool frame_low_recovery = false;
            for (auto const& t : f.tracks) {
                ++total_track_outputs;
                unique_track_ids.insert(t.track_id);
                ++track_output_frames[t.track_id];
                if (!t.has_observation) ++missing_track_observations;
                if (t.has_observation && t.observation.score < cfg.bytetrack.track_threshold) {
                    ++total_low_recovery_outputs;
                    frame_low_recovery = true;
                }
            }
            if (frame_low_recovery) ++frames_with_low_recovery;

            detector_ms.push_back(f.timing.detector_total_ms);
            tracker_ms.push_back(f.timing.tracker_total_ms);
            association_ms.push_back(f.timing.tracker_association_ms);
            pipeline_ms.push_back(f.timing.pipeline_total_ms);

            const bool print_line = (i == 0) || ((i + 1) % 15 == 0) || low > 0 || frame_low_recovery;
            if (print_line) {
                std::cout << "[" << std::setw(3) << (i + 1) << "/" << kMeasureFrames << "] "
                          << "cam=" << std::setw(5) << f.gpu_frame.frame_id
                          << " cand=" << f.detections.size()
                          << " (H" << high << "/L" << low << ")"
                          << " active=" << f.tracks.size()
                          << " | det=" << std::fixed << std::setprecision(2) << f.timing.detector_total_ms << " ms"
                          << " bt=" << std::setprecision(4) << f.timing.tracker_total_ms << " ms"
                          << " | ";
                print_tracks(f, cfg.bytetrack.track_threshold);
                std::cout << "\n";
            }
        }

        const auto wall_end = Clock::now();
        runtime.stop();

        const double wall_seconds = std::chrono::duration<double>(wall_end - wall_begin).count();
        const double throughput = wall_seconds > 0.0 ? static_cast<double>(kMeasureFrames) / wall_seconds : 0.0;

        const auto det_stats = summarize(std::move(detector_ms));
        const auto bt_stats = summarize(std::move(tracker_ms));
        const auto assoc_stats = summarize(std::move(association_ms));
        const auto pipe_stats = summarize(std::move(pipeline_ms));

        std::size_t longest_track_output = 0;
        for (auto const& [id, count] : track_output_frames) {
            (void)id;
            longest_track_output = std::max(longest_track_output, count);
        }

        std::cout << "\n========================================================\n";
        std::cout << "BYTE TRACK LIVE RESULT - v4.2\n";
        std::cout << "========================================================\n";
        std::cout << std::fixed;
        std::cout << "Pipeline throughput          : " << std::setprecision(2) << throughput << " FPS\n";
        std::cout << "Camera frame IDs skipped     : " << skipped_camera_ids << "\n";
        std::cout << "Failed camera reads          : " << runtime.failed_camera_reads() << "\n";
        std::cout << "Detector candidates total    : " << total_candidates
                  << " (high=" << total_high_candidates << ", low=" << total_low_candidates << ")\n";
        std::cout << "Max candidates / frame       : " << max_candidates << "\n";
        std::cout << "Frames with active tracks    : " << frames_with_tracks << "/" << kMeasureFrames << "\n";
        std::cout << "Max simultaneous tracks      : " << max_active_tracks << "\n";
        std::cout << "Unique temporal Track IDs    : " << unique_track_ids.size() << "\n";
        std::cout << "Longest Track-ID output span : " << longest_track_output << " measured frames\n";
        std::cout << "Total active track outputs   : " << total_track_outputs << "\n";
        std::cout << "Frames with low candidates   : " << frames_with_low_candidates << "\n";
        std::cout << "Low-score track recoveries   : " << total_low_recovery_outputs
                  << " outputs across " << frames_with_low_recovery << " frames\n";
        std::cout << "Missing current observations : " << missing_track_observations << "\n\n";

        std::cout << std::setprecision(4);
        std::cout << "Detector total mean/med/p95  : " << det_stats.mean << " / " << det_stats.median << " / " << det_stats.p95 << " ms\n";
        std::cout << "ByteTrack mean/med/p95       : " << bt_stats.mean << " / " << bt_stats.median << " / " << bt_stats.p95 << " ms\n";
        std::cout << "BT IoU association m/m/p95   : " << assoc_stats.mean << " / " << assoc_stats.median << " / " << assoc_stats.p95 << " ms\n";
        std::cout << "Pipeline mean/med/p95        : " << pipe_stats.mean << " / " << pipe_stats.median << " / " << pipe_stats.p95 << " ms\n";
        if (pipe_stats.median > 0.0) {
            std::cout << "ByteTrack share of median    : " << std::setprecision(3)
                      << (100.0 * bt_stats.median / pipe_stats.median) << " %\n";
        }

        std::cout << "\nTracking notes:\n";
        std::cout << "- Track ID is temporal continuity only; it is NOT student identity.\n";
        std::cout << "- GpuFrame + matched SCRFD landmarks are preserved for the next PAD stage.\n";

        if (missing_track_observations != 0) {
            throw std::runtime_error("active ByteTrack output lost current SCRFD observation metadata");
        }

        std::cout << "\nByteTrack live tracking verdict: PASS\n";
        std::cout << "Press ENTER to close...\n";
        std::cin.get();
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        std::cout << "Press ENTER to close...\n";
        std::cin.get();
        return 1;
    }
}
