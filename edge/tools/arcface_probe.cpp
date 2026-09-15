#include "arcface/arcface.hpp"
#include "pipeline/face_tracking_runtime.hpp"

#include <Windows.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/base.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using vision_runtime::arcface::ArcFaceRecognizer;
using vision_runtime::arcface::ArcFaceResult;
using vision_runtime::arcface::kEmbeddingSize;
using vision_runtime::pipeline::FaceTrackingRuntime;

struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double p05 = 0.0;
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

    auto quantile_nearest = [&](double q) {
        const auto idx = std::min(
            n - 1,
            static_cast<std::size_t>(std::ceil(q * static_cast<double>(n)) - 1.0)
        );
        return values[idx];
    };
    out.p05 = quantile_nearest(0.05);
    out.p95 = quantile_nearest(0.95);
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
    std::filesystem::path arcface_model;
    float min_detection_score = 0.50F;
    int arcface_every = 5;
    int enroll_samples = 10;
    int verify_samples = 60;
    int max_tracking_frames = 2500;
};

Args parse_args(int argc, char** argv) {
    Args args{};

    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        auto need = [&](char const* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string{"Missing value for "} + name);
            }
            return argv[++i];
        };

        if (token == "--scrfd") {
            args.scrfd_model = std::filesystem::absolute(need("--scrfd"));
        } else if (token == "--arcface") {
            args.arcface_model = std::filesystem::absolute(need("--arcface"));
        } else if (token == "--min-det-score") {
            args.min_detection_score = std::stof(need("--min-det-score"));
        } else if (token == "--every") {
            args.arcface_every = std::stoi(need("--every"));
        } else if (token == "--enroll") {
            args.enroll_samples = std::stoi(need("--enroll"));
        } else if (token == "--verify") {
            args.verify_samples = std::stoi(need("--verify"));
        } else if (token == "--max-frames") {
            args.max_tracking_frames = std::stoi(need("--max-frames"));
        } else if (token == "--help" || token == "-h") {
            std::cout
                << "arcface_probe v4.7 options:\n"
                << "  --scrfd <det_500m.onnx>\n"
                << "  --arcface <w600k_r50.onnx>\n"
                << "  --min-det-score <float>  default 0.50\n"
                << "  --every <frames>         default 5\n"
                << "  --enroll <samples>       default 10\n"
                << "  --verify <samples>       default 60\n"
                << "  --max-frames <count>     default 2500\n";
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

    if (args.arcface_model.empty()) {
        const auto rel = std::filesystem::path{
            L"models\\insightface\\buffalo_l\\w600k_r50.onnx"
        };
        args.arcface_model = find_upwards(cwd, rel);
        if (args.arcface_model.empty()) args.arcface_model = find_upwards(exe, rel);
    }

    if (args.scrfd_model.empty() || !std::filesystem::exists(args.scrfd_model)) {
        throw std::runtime_error("det_500m.onnx not found; pass --scrfd <path>");
    }
    if (args.arcface_model.empty() || !std::filesystem::exists(args.arcface_model)) {
        throw std::runtime_error(
            "w600k_r50.onnx not found; expected models\\insightface\\buffalo_l\\w600k_r50.onnx or pass --arcface <path>"
        );
    }
    if (
        !std::isfinite(args.min_detection_score) ||
        args.min_detection_score < 0.0F ||
        args.min_detection_score > 1.0F
    ) {
        throw std::runtime_error("--min-det-score must be 0..1");
    }
    if (args.arcface_every < 1) throw std::runtime_error("--every must be >= 1");
    if (args.enroll_samples < 1) throw std::runtime_error("--enroll must be >= 1");
    if (args.verify_samples < 1) throw std::runtime_error("--verify must be >= 1");
    if (args.max_tracking_frames < 1) throw std::runtime_error("--max-frames must be >= 1");
    return args;
}

std::array<float, kEmbeddingSize> finalize_reference(
    std::array<double, kEmbeddingSize> const& sum,
    int count
) {
    if (count <= 0) throw std::runtime_error("No enrollment embeddings");
    std::array<float, kEmbeddingSize> ref{};
    for (std::size_t i = 0; i < ref.size(); ++i) {
        ref[i] = static_cast<float>(sum[i] / static_cast<double>(count));
    }
    return vision_runtime::arcface::normalize_embedding(std::move(ref));
}

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

        vision_runtime::arcface::ArcFaceConfig arc_cfg{};
        arc_cfg.model_path = args.arcface_model;
        arc_cfg.input_width = 112;
        arc_cfg.input_height = 112;

        std::cout
            << "================================================================================\n"
            << "NATIVE ARCFACE GPU GATE v4.7\n"
            << "================================================================================\n"
            << "SCRFD model         : " << args.scrfd_model.string() << "\n"
            << "ArcFace model       : " << args.arcface_model.string() << "\n"
            << "Camera              : 1280x720 @30 NV12\n"
            << "Tracking            : SCRFD DirectML GPU -> ByteTrack native CPU\n"
            << "Recognition         : w600k_r50 / WinML DirectML GPU\n"
            << "Alignment           : SCRFD 5-point -> InsightFace 112x112 similarity transform\n"
            << "ArcFace input       : 1x3x112x112 NCHW RGB float32\n"
            << "Normalization       : (pixel - 127.5) / 127.5\n"
            << "Embedding           : 512 float -> CPU L2 normalize\n"
            << "ArcFace cadence     : every " << args.arcface_every << " tracking frames\n"
            << "Quality gate        : SCRFD score >= " << args.min_detection_score << "\n"
            << "Enrollment samples  : " << args.enroll_samples << "\n"
            << "Verification samples: " << args.verify_samples << "\n"
            << "Identity threshold  : NONE; this is a same-person stability gate\n"
            << "Image CPU roundtrip : NONE in native ArcFace image path\n\n";

        FaceTrackingRuntime runtime{tracking_cfg};
        ArcFaceRecognizer recognizer{arc_cfg};
        runtime.start();

        constexpr int kWarmup = 35;
        std::cout << "Warm-up             : " << kWarmup << " tracking frames...\n";
        for (int i = 0; i < kWarmup; ++i) {
            auto f = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!f) throw std::runtime_error("tracking warm-up timeout");
        }
        runtime.reset_tracker();

        std::array<double, kEmbeddingSize> enroll_sum{};
        int enrolled = 0;
        int verified = 0;
        bool reference_ready = false;
        std::array<float, kEmbeddingSize> reference{};

        std::vector<double> similarities;
        std::vector<double> raw_norms;
        std::vector<double> align_rmse;
        std::vector<double> total_ms;
        std::vector<double> eval_ms;

        std::size_t tracking_frames = 0;
        std::size_t camera_skips = 0;
        std::size_t no_track_skips = 0;
        std::size_t ambiguous_track_skips = 0;
        std::size_t quality_skips = 0;
        std::size_t cadence_skips = 0;
        std::uint64_t prev_cam = 0;
        std::set<std::uint64_t> track_ids;

        const auto wall_begin = std::chrono::steady_clock::now();

        while (
            verified < args.verify_samples &&
            tracking_frames < static_cast<std::size_t>(args.max_tracking_frames)
        ) {
            auto frame = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!frame) throw std::runtime_error("ArcFace tracking timeout");
            ++tracking_frames;

            if (prev_cam != 0 && frame->gpu_frame.frame_id > prev_cam + 1) {
                camera_skips += static_cast<std::size_t>(
                    frame->gpu_frame.frame_id - prev_cam - 1
                );
            }
            prev_cam = frame->gpu_frame.frame_id;

            if (frame->tracks.empty()) {
                ++no_track_skips;
                continue;
            }
            if (frame->tracks.size() != 1) {
                ++ambiguous_track_skips;
                continue;
            }
            if (((tracking_frames - 1) % static_cast<std::size_t>(args.arcface_every)) != 0) {
                ++cadence_skips;
                continue;
            }

            auto const& track = frame->tracks.front();
            if (
                !track.has_observation ||
                track.observation.score < args.min_detection_score
            ) {
                ++quality_skips;
                continue;
            }
            track_ids.insert(track.track_id);

            const ArcFaceResult result = recognizer.evaluate(
                frame->gpu_frame,
                track.observation
            );

            raw_norms.push_back(result.raw_embedding_norm);
            align_rmse.push_back(result.transform.rmse);
            total_ms.push_back(result.timing.total_ms);
            eval_ms.push_back(result.timing.evaluate_ms);

            if (!reference_ready) {
                for (std::size_t i = 0; i < kEmbeddingSize; ++i) {
                    enroll_sum[i] += result.embedding[i];
                }
                ++enrolled;
                std::cout
                    << "[ENROLL " << std::setw(2) << enrolled << '/' << args.enroll_samples << "] "
                    << "cam=" << std::setw(5) << frame->gpu_frame.frame_id
                    << " ID" << track.track_id
                    << " det=" << std::fixed << std::setprecision(3) << track.observation.score
                    << " alignRMSE=" << result.transform.rmse
                    << " rawNorm=" << result.raw_embedding_norm
                    << " arc=" << result.timing.total_ms << "ms\n";

                if (enrolled == args.enroll_samples) {
                    reference = finalize_reference(enroll_sum, enrolled);
                    reference_ready = true;
                    std::cout
                        << "REFERENCE LOCKED    : mean of " << enrolled
                        << " L2-normalized embeddings -> L2 normalize\n\n";
                }
                continue;
            }

            const float sim = vision_runtime::arcface::cosine_similarity(
                reference,
                result.embedding
            );
            similarities.push_back(sim);
            ++verified;

            std::cout
                << "[VERIFY " << std::setw(2) << verified << '/' << args.verify_samples << "] "
                << "cam=" << std::setw(5) << frame->gpu_frame.frame_id
                << " ID" << track.track_id
                << " det=" << std::fixed << std::setprecision(3) << track.observation.score
                << " cosine=" << std::setprecision(6) << sim
                << " alignRMSE=" << std::setprecision(3) << result.transform.rmse
                << " arc=" << result.timing.total_ms << "ms\n";
        }

        const auto wall_end = std::chrono::steady_clock::now();
        runtime.stop();

        const double wall_seconds =
            std::chrono::duration<double>(wall_end - wall_begin).count();
        const double tracking_fps = wall_seconds > 0.0
            ? static_cast<double>(tracking_frames) / wall_seconds
            : 0.0;

        const auto sim = summarize(similarities);
        const auto norm = summarize(raw_norms);
        const auto rmse = summarize(align_rmse);
        const auto total = summarize(total_ms);
        const auto eval = summarize(eval_ms);

        std::cout
            << "\n================================================================================\n"
            << "RESULT - ARCFACE GPU v4.7 SAME-PERSON STABILITY\n"
            << "================================================================================\n"
            << "Enrollment embeddings      : " << enrolled << '/' << args.enroll_samples << "\n"
            << "Verification embeddings    : " << verified << '/' << args.verify_samples << "\n"
            << "Unique ByteTrack IDs seen  : " << track_ids.size() << "\n"
            << "Tracking frames consumed   : " << tracking_frames << "\n"
            << "Tracking throughput        : " << std::fixed << std::setprecision(2)
            << tracking_fps << " FPS\n"
            << "Camera frame IDs skipped   : " << camera_skips << "\n"
            << "No-track skips             : " << no_track_skips << "\n"
            << "Ambiguous-track skips      : " << ambiguous_track_skips << "\n"
            << "Quality-gate skips         : " << quality_skips << "\n"
            << "Cadence skips              : " << cadence_skips << "\n\n"
            << std::setprecision(6)
            << "Cosine mean/median         : " << sim.mean << " / " << sim.median << "\n"
            << "Cosine P05/min/max         : " << sim.p05 << " / " << sim.min << " / " << sim.max << "\n"
            << std::setprecision(3)
            << "Alignment RMSE med/P95     : " << rmse.median << " / " << rmse.p95 << " px\n"
            << "Raw embedding norm med/P95 : " << norm.median << " / " << norm.p95 << "\n"
            << "ArcFace total med/P95      : " << total.median << " / " << total.p95 << " ms\n"
            << "WinML Evaluate med/P95     : " << eval.median << " / " << eval.p95 << " ms\n\n"
            << "Gate:\n"
            << "- This probe verifies native GPU alignment/inference stability for ONE identity.\n"
            << "- Do not freeze an identity-match threshold from this self-test.\n"
            << "- Photo of the same person is EXPECTED to match ArcFace; PAD is the presentation-attack gate.\n";

        if (verified < args.verify_samples) {
            std::cout
                << "\nWARNING: verification target was not reached before --max-frames.\n";
        }
        return verified > 0 ? 0 : 1;
    } catch (winrt::hresult_error const& e) {
        std::cerr
            << "WinRT ERROR 0x" << std::hex << std::uppercase
            << static_cast<std::uint32_t>(e.code()) << std::dec
            << ": " << winrt::to_string(e.message()) << "\n";
        return 2;
    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
