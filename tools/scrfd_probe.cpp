#include "camera/camera.hpp"
#include "scrfd/scrfd.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Windows.h>

#include <winrt/base.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.AI.MachineLearning.h>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kCameraWarmupFrames = 30;
constexpr std::size_t kScrfdWarmupFrames = 5;
constexpr std::size_t kMeasureFrames = 60;

struct TimingStats {
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
};

struct CameraCandidate {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    char const* label = "";
};

struct ProbeResult {
    CameraCandidate candidate{};
    vision_runtime::CameraMode actual_mode{};

    double detector_fps = 0.0;
    double source_fps = 0.0;
    std::uint64_t skipped_camera_ids = 0;
    std::uint64_t failed_reads = 0;

    std::size_t detected_frames = 0;
    std::size_t total_detections = 0;
    std::size_t max_faces = 0;
    double average_top_score = 0.0;

    TimingStats total{};
    TimingStats evaluate{};
    TimingStats postprocess{};
    TimingStats source_interval{};
};

TimingStats summarize(std::vector<double> values) {
    TimingStats result{};
    if (values.empty()) {
        return result;
    }

    result.mean = std::accumulate(
        values.begin(),
        values.end(),
        0.0
    ) / static_cast<double>(values.size());

    std::sort(values.begin(), values.end());
    result.median = values[values.size() / 2];

    const std::size_t p95_index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(values.size() * 0.95)
    );
    result.p95 = values[p95_index];

    return result;
}

std::filesystem::path executable_directory() {
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        nullptr,
        buffer.data(),
        static_cast<DWORD>(buffer.size())
    );

    if (length == 0 || length >= buffer.size()) {
        throw std::runtime_error(
            "GetModuleFileNameW gagal saat mencari lokasi executable"
        );
    }

    buffer.resize(length);
    return std::filesystem::path{buffer}.parent_path();
}

std::filesystem::path find_model_from_root(
    std::filesystem::path root
) {
    const auto relative = std::filesystem::path{
        L"models\\insightface\\buffalo_sc\\det_500m.onnx"
    };

    for (int depth = 0; depth < 10; ++depth) {
        const auto candidate = root / relative;
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate).lexically_normal();
        }

        const auto parent = root.parent_path();
        if (parent.empty() || parent == root) {
            break;
        }
        root = parent;
    }

    return {};
}

std::filesystem::path default_model_path() {
    if (auto candidate = find_model_from_root(std::filesystem::current_path());
        !candidate.empty()) {
        return candidate;
    }

    if (auto candidate = find_model_from_root(executable_directory());
        !candidate.empty()) {
        return candidate;
    }

    throw std::runtime_error(
        "det_500m.onnx tidak ditemukan otomatis. "
        "Berikan path model sebagai argumen pertama."
    );
}

struct RuntimePreflightResult {
    bool passed = false;
    std::string error;
};

RuntimePreflightResult run_runtime_preflight(
    std::filesystem::path const& model_path,
    winrt::Windows::AI::MachineLearning::LearningModelDeviceKind kind
) {
    using namespace winrt::Windows::AI::MachineLearning;

    RuntimePreflightResult result{};

    try {
        auto model = LearningModel::LoadFromFilePath(
            winrt::hstring(std::filesystem::absolute(model_path).wstring())
        );

        auto device = LearningModelDevice{kind};
        auto session = LearningModelSession{model, device};
        auto binding = LearningModelBinding{session};

        auto inputs = model.InputFeatures();
        if (inputs.Size() != 1) {
            throw std::runtime_error("preflight expects exactly one model input");
        }

        std::vector<std::int64_t> shape{1, 3, 640, 640};
        std::vector<float> zeros(
            static_cast<std::size_t>(1) * 3 * 640 * 640,
            0.0F
        );

        auto tensor = TensorFloat::CreateFromArray(
            shape,
            winrt::array_view<float const>{zeros}
        );

        binding.Bind(inputs.GetAt(0).Name(), tensor);
        (void)session.Evaluate(binding, L"scrfd-preflight");
        result.passed = true;
    } catch (winrt::hresult_error const& error) {
        std::ostringstream message;
        message
            << "HRESULT 0x"
            << std::hex << std::uppercase
            << static_cast<std::uint32_t>(error.code())
            << std::dec
            << " (" << winrt::to_string(error.message()) << ")";
        result.error = message.str();
    } catch (std::exception const& error) {
        result.error = error.what();
    }

    return result;
}

ProbeResult run_camera_candidate(
    std::filesystem::path const& model_path,
    CameraCandidate const& candidate
) {
    std::cout << "\n========================================================\n";
    std::cout << "CAMERA SOURCE TEST - " << candidate.label << "\n";
    std::cout << "========================================================\n";
    std::cout
        << "Requested camera : "
        << candidate.width << "x" << candidate.height << " @ 30 FPS NV12\n";
    std::cout << "SCRFD input      : 640x640 (FROZEN baseline)\n";

    vision_runtime::CameraConfig camera_config{};
    camera_config.width = candidate.width;
    camera_config.height = candidate.height;
    camera_config.fps = 30;
    camera_config.source_subtype = L"NV12";
    camera_config.output_subtype = L"NV12";
    camera_config.force_constant_framerate = true;
    camera_config.restore_legacy_control_on_stop = true;

    vision_runtime::Camera camera{camera_config};
    camera.start();

    const auto mode = camera.mode();
    std::wcout
        << L"Actual camera    : "
        << mode.width << L"x" << mode.height
        << L" @ " << std::fixed << std::setprecision(2) << mode.fps
        << L" | source=" << mode.subtype
        << L" | output=" << mode.output_subtype
        << L"\n";

    if (mode.width != candidate.width || mode.height != candidate.height) {
        camera.stop();
        throw std::runtime_error(
            "Camera melakukan fallback resolution pada candidate " +
            std::string{candidate.label}
        );
    }

    vision_runtime::ScrfdConfig detector_config{};
    detector_config.model_path = model_path;
    detector_config.input_width = 640;
    detector_config.input_height = 640;
    detector_config.score_threshold = 0.50F;
    detector_config.nms_threshold = 0.40F;

    vision_runtime::ScrfdDetector detector{detector_config};

    std::uint64_t last_frame_id = 0;

    std::cout << "Camera warm-up   : " << kCameraWarmupFrames << " frames...\n";
    for (std::size_t i = 0; i < kCameraWarmupFrames; ++i) {
        auto frame = camera.wait_for_frame(
            last_frame_id,
            std::chrono::milliseconds{1500}
        );
        if (!frame) {
            camera.stop();
            throw std::runtime_error("Timeout saat camera warm-up");
        }
        last_frame_id = frame->frame_id;
    }

    std::cout << "SCRFD warm-up    : " << kScrfdWarmupFrames << " inference...\n";
    for (std::size_t i = 0; i < kScrfdWarmupFrames; ++i) {
        auto frame = camera.wait_for_frame(
            last_frame_id,
            std::chrono::milliseconds{1500}
        );
        if (!frame) {
            camera.stop();
            throw std::runtime_error("Timeout saat SCRFD warm-up");
        }
        last_frame_id = frame->frame_id;
        (void)detector.detect(*frame);
    }

    std::cout
        << "Measure           : " << kMeasureFrames
        << " latest frames. Jaga posisi/scene sebisa mungkin sama antar-test.\n";

    std::vector<double> total_times;
    std::vector<double> evaluate_times;
    std::vector<double> post_times;
    std::vector<double> source_intervals;
    total_times.reserve(kMeasureFrames);
    evaluate_times.reserve(kMeasureFrames);
    post_times.reserve(kMeasureFrames);
    source_intervals.reserve(kMeasureFrames - 1);

    std::uint64_t previous_frame_id = last_frame_id;
    std::uint64_t previous_source_ns = 0;
    std::uint64_t first_source_ns = 0;
    std::uint64_t last_source_ns = 0;
    std::uint64_t skipped_camera_ids = 0;

    std::size_t detected_frames = 0;
    std::size_t total_detections = 0;
    std::size_t max_faces = 0;
    double top_score_sum = 0.0;
    std::size_t top_score_count = 0;

    const auto begin = Clock::now();

    for (std::size_t i = 1; i <= kMeasureFrames; ++i) {
        auto frame = camera.wait_for_frame(
            previous_frame_id,
            std::chrono::milliseconds{2000}
        );
        if (!frame) {
            camera.stop();
            throw std::runtime_error("Timeout menunggu camera frame untuk SCRFD");
        }

        if (frame->frame_id > previous_frame_id + 1) {
            skipped_camera_ids += frame->frame_id - previous_frame_id - 1;
        }
        previous_frame_id = frame->frame_id;

        if (frame->source_timestamp_ns != 0) {
            if (first_source_ns == 0) {
                first_source_ns = frame->source_timestamp_ns;
            }
            last_source_ns = frame->source_timestamp_ns;

            if (
                previous_source_ns != 0 &&
                frame->source_timestamp_ns > previous_source_ns
            ) {
                source_intervals.push_back(
                    static_cast<double>(
                        frame->source_timestamp_ns - previous_source_ns
                    ) / 1'000'000.0
                );
            }
            previous_source_ns = frame->source_timestamp_ns;
        }

        auto detections = detector.detect(*frame);
        const auto timing = detector.last_timing();

        total_times.push_back(timing.total_ms);
        evaluate_times.push_back(timing.evaluate_ms);
        post_times.push_back(timing.postprocess_ms);

        if (!detections.empty()) {
            ++detected_frames;
            top_score_sum += detections.front().score;
            ++top_score_count;
        }

        total_detections += detections.size();
        max_faces = std::max(max_faces, detections.size());

        if (i == 1 || i % 10 == 0 || i == kMeasureFrames) {
            std::cout
                << "  [" << std::setw(2) << i << "/" << kMeasureFrames << "]"
                << " faces=" << detections.size()
                << " | total=" << std::fixed << std::setprecision(2)
                << timing.total_ms << " ms";

            if (!detections.empty()) {
                std::cout
                    << " | top=" << std::setprecision(3)
                    << detections.front().score;
            }
            std::cout << "\n";
        }
    }

    const auto end = Clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(end - begin).count();

    ProbeResult result{};
    result.candidate = candidate;
    result.actual_mode = mode;
    result.detector_fps = static_cast<double>(kMeasureFrames) / elapsed_seconds;
    result.skipped_camera_ids = skipped_camera_ids;
    result.failed_reads = camera.failed_reads();
    result.detected_frames = detected_frames;
    result.total_detections = total_detections;
    result.max_faces = max_faces;
    result.average_top_score =
        top_score_count == 0
            ? 0.0
            : top_score_sum / static_cast<double>(top_score_count);
    result.total = summarize(total_times);
    result.evaluate = summarize(evaluate_times);
    result.postprocess = summarize(post_times);
    result.source_interval = summarize(source_intervals);

    if (
        first_source_ns != 0 &&
        last_source_ns > first_source_ns &&
        kMeasureFrames > 1
    ) {
        const double source_seconds =
            static_cast<double>(last_source_ns - first_source_ns) /
            1'000'000'000.0;
        result.source_fps =
            static_cast<double>(kMeasureFrames - 1) / source_seconds;
    }

    camera.stop();

    std::cout << "\nRESULT - " << candidate.label << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Detector throughput      : "
              << std::fixed << std::setprecision(2)
              << result.detector_fps << " FPS\n";
    std::cout << "Source timestamp FPS     : "
              << result.source_fps << " FPS\n";
    std::cout << "Detected frames          : "
              << result.detected_frames << "/" << kMeasureFrames << "\n";
    std::cout << "Average top confidence   : "
              << std::setprecision(3) << result.average_top_score << "\n";
    std::cout << "Max simultaneous faces   : "
              << result.max_faces << "\n";
    std::cout << "Camera frame IDs skipped : "
              << result.skipped_camera_ids << "\n";
    std::cout << "Failed camera reads      : "
              << result.failed_reads << "\n";
    std::cout << "Total med/p95            : "
              << std::setprecision(2)
              << result.total.median << " / "
              << result.total.p95 << " ms\n";
    std::cout << "Evaluate med/p95         : "
              << result.evaluate.median << " / "
              << result.evaluate.p95 << " ms\n";
    std::cout << "Postprocess med/p95      : "
              << result.postprocess.median << " / "
              << result.postprocess.p95 << " ms\n";
    std::cout << "Source dt med/p95        : "
              << result.source_interval.median << " / "
              << result.source_interval.p95 << " ms\n";

    return result;
}

void print_comparison(
    std::array<ProbeResult, 2> const& results
) {
    std::cout << "\n========================================================\n";
    std::cout << "CAMERA SOURCE A/B SUMMARY - PROBE ONLY\n";
    std::cout << "========================================================\n";
    std::cout << "SCRFD model/input remain frozen: original det_500m.onnx / 640x640\n";
    std::cout << "Production runtime remains v3.4 until this probe is evaluated.\n\n";

    std::cout
        << std::left << std::setw(27) << "Metric"
        << std::right << std::setw(16) << results[0].candidate.label
        << std::setw(16) << results[1].candidate.label
        << "\n";
    std::cout << std::string(59, '-') << "\n";

    auto row = [&](char const* name, double a, double b, int precision = 2) {
        std::cout
            << std::left << std::setw(27) << name
            << std::right << std::fixed << std::setprecision(precision)
            << std::setw(16) << a
            << std::setw(16) << b
            << "\n";
    };

    row("Source FPS", results[0].source_fps, results[1].source_fps);
    row("Detector throughput FPS", results[0].detector_fps, results[1].detector_fps);
    row("Total median ms", results[0].total.median, results[1].total.median);
    row("Total P95 ms", results[0].total.p95, results[1].total.p95);
    row("Evaluate median ms", results[0].evaluate.median, results[1].evaluate.median);
    row("Evaluate P95 ms", results[0].evaluate.p95, results[1].evaluate.p95);
    row("Post median ms", results[0].postprocess.median, results[1].postprocess.median, 3);
    row("Detected-frame ratio %",
        100.0 * static_cast<double>(results[0].detected_frames) / kMeasureFrames,
        100.0 * static_cast<double>(results[1].detected_frames) / kMeasureFrames,
        1);
    row("Average top confidence", results[0].average_top_score, results[1].average_top_score, 3);

    std::cout << "\nInterpretation guard:\n";
    std::cout << "- Ini A/B camera SOURCE resolution, bukan perubahan input SCRFD.\n";
    std::cout << "- 640x480 memakai aspect ratio 4:3; framing/FOV dapat berbeda dari 1280x720 (16:9).\n";
    std::cout << "- Jangan port 640x480 ke production hanya karena latency lebih rendah.\n";
    std::cout << "  Validasi multi-face, wajah jauh, sisi frame, dan pose menyamping tetap diperlukan.\n";
    std::cout << "- Perubahan production baru dilakukan setelah hasil probe dipilih.\n";
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);

        const std::filesystem::path model_path =
            argc >= 2
                ? std::filesystem::path{argv[1]}
                : default_model_path();

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(
                "Model file tidak ditemukan: " +
                std::filesystem::absolute(model_path).string()
            );
        }

        std::cout << "========================================================\n";
        std::cout << "NATIVE SCRFD RESEARCH PROBE v3.5\n";
        std::cout << "========================================================\n";
        std::wcout << L"Model          : "
                   << std::filesystem::absolute(model_path).wstring() << L"\n";
        std::cout << "Runtime base   : v3.4 (UNCHANGED)\n";
        std::cout << "Research       : camera source 1280x720 vs 640x480\n";
        std::cout << "SCRFD input    : 640x640 for BOTH candidates\n";
        std::cout << "Image path     : GPU-backed NV12; no OpenCV/NumPy runtime path\n\n";

        std::cout << "Runtime preflight (diagnostic only):\n";
        const auto cpu_preflight = run_runtime_preflight(
            model_path,
            winrt::Windows::AI::MachineLearning::LearningModelDeviceKind::Cpu
        );
        std::cout << "  WinML CPU model test : "
                  << (cpu_preflight.passed ? "PASS" : "FAIL") << "\n";
        if (!cpu_preflight.passed) {
            std::cout << "    " << cpu_preflight.error << "\n";
        }

        const auto gpu_preflight = run_runtime_preflight(
            model_path,
            winrt::Windows::AI::MachineLearning::LearningModelDeviceKind::DirectX
        );
        std::cout << "  WinML GPU model test : "
                  << (gpu_preflight.passed ? "PASS" : "FAIL") << "\n";
        if (!gpu_preflight.passed) {
            std::cout << "    " << gpu_preflight.error << "\n";
        }

        if (!gpu_preflight.passed) {
            throw std::runtime_error(
                "WinML DirectX preflight gagal; A/B test dibatalkan"
            );
        }

        constexpr std::array<CameraCandidate, 2> candidates{{
            {1280, 720, "1280x720"},
            {640, 480, "640x480"},
        }};

        std::array<ProbeResult, 2> results{};

        std::cout << "\nIMPORTANT: A/B akan dijalankan berurutan.\n";
        std::cout << "Usahakan pencahayaan, jarak, posisi wajah, dan jumlah orang tetap sama.\n";

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            results[i] = run_camera_candidate(
                model_path,
                candidates[i]
            );
        }

        print_comparison(results);

        std::cout << "\nOpenCV / NumPy production path: NOT USED\n";
        std::cout << "Runtime source files changed by this experiment: NONE\n";
        std::cout << "\nPress ENTER to close...";
        std::cin.get();
        return 0;
    } catch (winrt::hresult_error const& error) {
        std::cerr
            << "WinRT error 0x"
            << std::hex << static_cast<std::uint32_t>(error.code())
            << std::dec
            << ": "
            << winrt::to_string(error.message())
            << "\n";
    } catch (std::exception const& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
    }

    std::cerr << "Press ENTER to close...";
    std::cin.get();
    return 1;
}
