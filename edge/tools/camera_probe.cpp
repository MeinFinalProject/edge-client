#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Media.Capture.h>
#include <winrt/Windows.Media.Capture.Frames.h>
#include <winrt/Windows.Media.MediaProperties.h>
#include <winrt/base.h>

#include "camera/camera.hpp"

namespace {

using winrt::Windows::Media::Capture::MediaStreamType;
using winrt::Windows::Media::Capture::Frames::MediaFrameFormat;
using winrt::Windows::Media::Capture::Frames::MediaFrameSourceGroup;
using winrt::Windows::Media::Capture::Frames::MediaFrameSourceKind;

constexpr std::uint32_t kCameraIndex = 0;
constexpr std::uint32_t kWidth = 1280;
constexpr std::uint32_t kHeight = 720;
constexpr std::uint32_t kFps = 30;
constexpr int kWarmupFrames = 60;
constexpr int kBenchmarkFrames = 120;

struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double max = 0.0;
};

std::uint64_t steady_now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

double frame_rate(MediaFrameFormat const& format) {
    auto ratio = format.FrameRate();
    auto denominator = ratio.Denominator();
    if (denominator == 0) {
        return 0.0;
    }
    return static_cast<double>(ratio.Numerator()) /
           static_cast<double>(denominator);
}

std::vector<MediaFrameSourceGroup> eligible_groups() {
    auto groups = MediaFrameSourceGroup::FindAllAsync().get();
    std::vector<MediaFrameSourceGroup> eligible;
    eligible.reserve(groups.Size());

    for (auto const& group : groups) {
        bool has_color_video = false;
        for (auto const& info : group.SourceInfos()) {
            auto stream = info.MediaStreamType();
            bool is_video =
                stream == MediaStreamType::VideoPreview ||
                stream == MediaStreamType::VideoRecord;

            if (
                info.SourceKind() == MediaFrameSourceKind::Color &&
                is_video
            ) {
                has_color_video = true;
                break;
            }
        }

        if (has_color_video) {
            eligible.push_back(group);
        }
    }

    return eligible;
}

Stats summarize(std::vector<double> values) {
    Stats result;
    if (values.empty()) {
        return result;
    }

    std::sort(values.begin(), values.end());
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                  static_cast<double>(values.size());

    const auto n = values.size();
    result.median = n % 2 == 0
        ? (values[n / 2 - 1] + values[n / 2]) / 2.0
        : values[n / 2];

    auto p95_index = static_cast<std::size_t>(
        std::ceil(static_cast<double>(n) * 0.95)
    ) - 1;
    p95_index = std::min(p95_index, n - 1);

    result.p95 = values[p95_index];
    result.max = values.back();
    return result;
}

double fps_from_intervals(std::vector<double> const& intervals_ms) {
    if (intervals_ms.empty()) {
        return 0.0;
    }

    const double total_ms = std::accumulate(
        intervals_ms.begin(), intervals_ms.end(), 0.0
    );

    return total_ms > 0.0
        ? static_cast<double>(intervals_ms.size()) * 1000.0 / total_ms
        : 0.0;
}

void print_driver_modes(MediaFrameSourceGroup const& group) {
    winrt::Windows::Media::Capture::MediaCaptureInitializationSettings settings;
    settings.SourceGroup(group);
    settings.SharingMode(
        winrt::Windows::Media::Capture::MediaCaptureSharingMode::ExclusiveControl
    );
    settings.MemoryPreference(
        winrt::Windows::Media::Capture::MediaCaptureMemoryPreference::Auto
    );
    settings.StreamingCaptureMode(
        winrt::Windows::Media::Capture::StreamingCaptureMode::Video
    );

    winrt::Windows::Media::Capture::MediaCapture capture;
    capture.InitializeAsync(settings).get();
    auto sources = capture.FrameSources();

    struct Line {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        double fps = 0.0;
        std::wstring subtype;
    };

    std::vector<Line> lines;

    for (auto const& info : group.SourceInfos()) {
        auto stream = info.MediaStreamType();
        bool is_video =
            stream == MediaStreamType::VideoPreview ||
            stream == MediaStreamType::VideoRecord;

        if (
            info.SourceKind() != MediaFrameSourceKind::Color ||
            !is_video ||
            !sources.HasKey(info.Id())
        ) {
            continue;
        }

        auto source = sources.Lookup(info.Id());
        for (auto const& format : source.SupportedFormats()) {
            auto video = format.VideoFormat();
            if (!video) {
                continue;
            }

            Line line;
            line.width = video.Width();
            line.height = video.Height();
            line.fps = frame_rate(format);
            line.subtype = std::wstring(format.Subtype().c_str());
            lines.push_back(std::move(line));
        }
    }

    std::sort(
        lines.begin(), lines.end(),
        [](Line const& a, Line const& b) {
            if (a.width != b.width) return a.width < b.width;
            if (a.height != b.height) return a.height < b.height;
            if (std::abs(a.fps - b.fps) > 0.001) return a.fps < b.fps;
            return a.subtype < b.subtype;
        }
    );

    lines.erase(
        std::unique(
            lines.begin(), lines.end(),
            [](Line const& a, Line const& b) {
                return a.width == b.width &&
                       a.height == b.height &&
                       std::abs(a.fps - b.fps) < 0.001 &&
                       a.subtype == b.subtype;
            }
        ),
        lines.end()
    );

    std::wcout << L"Driver-reported supported modes:\n";
    for (auto const& line : lines) {
        std::wcout
            << L"  " << line.width << L"x" << line.height
            << L" @ " << std::fixed << std::setprecision(2) << line.fps
            << L" | " << line.subtype << L"\n";
    }

    capture.Close();
}

} // namespace

int main() {
    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);

        auto groups = eligible_groups();
        if (kCameraIndex >= groups.size()) {
            throw std::runtime_error("Camera index tidak tersedia");
        }

        auto group = groups[kCameraIndex];

        std::wcout << L"========================================\n";
        std::wcout << L"NATIVE PRODUCTION CAMERA PROBE v3.0\n";
        std::wcout << L"========================================\n";
        std::wcout << L"Device: " << group.DisplayName().c_str() << L"\n\n";

        print_driver_modes(group);

        vision_runtime::CameraConfig config;
        config.index = kCameraIndex;
        config.width = kWidth;
        config.height = kHeight;
        config.fps = kFps;
        config.acquisition_mode = vision_runtime::CameraAcquisitionMode::Realtime;
        config.source_subtype = L"NV12";
        config.output_subtype = L"NV12";
        config.force_constant_framerate = true;
        config.restore_legacy_control_on_stop = true;

        vision_runtime::Camera camera(config);
        camera.start();

        auto mode = camera.mode();
        auto controls = camera.control_status();

        std::wcout << L"\nProduction camera selected:\n";
        std::wcout
            << L"  " << mode.width << L"x" << mode.height
            << L" @ " << std::fixed << std::setprecision(2) << mode.fps
            << L" | source=" << mode.subtype
            << L" | output=" << mode.output_subtype << L"\n";

        std::cout << "\nConstant-frame-rate control:\n";
        std::cout << "  Legacy property #19 supported : "
                  << (controls.legacy_low_light_supported ? "YES" : "NO")
                  << "\n";
        std::cout << "  Original value                : "
                  << controls.legacy_low_light_original_value << "\n";
        std::cout << "  Current value                 : "
                  << controls.legacy_low_light_current_value << "\n";
        std::cout << "  Forced OFF                    : "
                  << (controls.legacy_low_light_forced_off ? "YES" : "NO")
                  << "\n";
        std::cout << "  Restore on stop               : "
                  << (controls.legacy_low_light_restore_pending ? "YES" : "NO")
                  << "\n";

        std::cout << "\nWarming up " << kWarmupFrames << " frames...\n";

        std::uint64_t last_id = 0;
        std::optional<vision_runtime::GpuFrame> first;

        for (int i = 0; i < kWarmupFrames; ++i) {
            auto frame = camera.wait_for_frame(
                last_id,
                std::chrono::milliseconds{1500}
            );
            if (!frame) {
                throw std::runtime_error("Timeout saat warm-up");
            }
            last_id = frame->frame_id;
            if (!first) {
                first = frame;
            }
        }

        if (!first) {
            throw std::runtime_error("Tidak mendapat GPU frame");
        }

        std::cout << "First GPU frame:\n";
        std::cout << "  size       : " << first->width << "x" << first->height << "\n";
        std::cout << "  DXGI fmt   : " << static_cast<int>(first->dxgi_format) << "\n";
        std::cout << "  GPU-backed : " << (first->valid() ? "YES" : "NO") << "\n";

        std::cout << "\nBenchmarking " << kBenchmarkFrames
                  << " steady-state frames...\n";

        std::vector<double> source_dt_ms;
        std::vector<double> callback_dt_ms;
        std::vector<double> delivery_age_ms;
        source_dt_ms.reserve(kBenchmarkFrames - 1);
        callback_dt_ms.reserve(kBenchmarkFrames - 1);
        delivery_age_ms.reserve(kBenchmarkFrames);

        std::uint64_t prev_source = 0;
        std::uint64_t prev_callback = 0;
        std::uint64_t first_consumer_ns = 0;
        std::uint64_t last_consumer_ns = 0;
        std::uint64_t first_frame_id = 0;
        std::uint64_t last_frame_id = 0;

        for (int i = 0; i < kBenchmarkFrames; ++i) {
            auto frame = camera.wait_for_frame(
                last_id,
                std::chrono::milliseconds{1500}
            );
            if (!frame) {
                throw std::runtime_error("Timeout saat benchmark");
            }

            const auto consumer_ns = steady_now_ns();
            last_id = frame->frame_id;

            if (i == 0) {
                first_consumer_ns = consumer_ns;
                first_frame_id = frame->frame_id;
            }
            last_consumer_ns = consumer_ns;
            last_frame_id = frame->frame_id;

            if (prev_source != 0 && frame->source_timestamp_ns != 0) {
                source_dt_ms.push_back(
                    static_cast<double>(frame->source_timestamp_ns - prev_source) /
                    1'000'000.0
                );
            }
            if (frame->source_timestamp_ns != 0) {
                prev_source = frame->source_timestamp_ns;
            }

            if (prev_callback != 0) {
                callback_dt_ms.push_back(
                    static_cast<double>(frame->callback_entry_ns - prev_callback) /
                    1'000'000.0
                );
            }
            prev_callback = frame->callback_entry_ns;

            delivery_age_ms.push_back(
                static_cast<double>(consumer_ns - frame->timestamp_ns) /
                1'000'000.0
            );
        }

        const auto source_stats = summarize(source_dt_ms);
        const auto callback_stats = summarize(callback_dt_ms);
        const auto delivery_stats = summarize(delivery_age_ms);
        const double source_fps = fps_from_intervals(source_dt_ms);
        const double callback_fps = fps_from_intervals(callback_dt_ms);

        double consumer_fps = 0.0;
        if (last_consumer_ns > first_consumer_ns && kBenchmarkFrames > 1) {
            consumer_fps =
                static_cast<double>(kBenchmarkFrames - 1) * 1'000'000'000.0 /
                static_cast<double>(last_consumer_ns - first_consumer_ns);
        }

        std::uint64_t skipped_ids = 0;
        if (last_frame_id >= first_frame_id) {
            const auto id_span = last_frame_id - first_frame_id + 1;
            if (id_span > static_cast<std::uint64_t>(kBenchmarkFrames)) {
                skipped_ids = id_span - static_cast<std::uint64_t>(kBenchmarkFrames);
            }
        }

        std::cout << "\n========================================\n";
        std::cout << "RESULT - PRODUCTION v3.0\n";
        std::cout << "========================================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Frames                 : " << kBenchmarkFrames << "\n";
        std::cout << "Consumer skipped IDs   : " << skipped_ids << "\n";
        std::cout << "Camera callback FPS    : " << callback_fps << "\n";
        std::cout << "Source timestamp FPS   : " << source_fps << "\n";
        std::cout << "Consumer FPS           : " << consumer_fps << "\n";
        std::cout << "Target achievement     : "
                  << (source_fps / static_cast<double>(kFps) * 100.0)
                  << " %\n";
        std::cout << "Source dt median/P95   : "
                  << source_stats.median << " / " << source_stats.p95 << " ms\n";
        std::cout << "Callback dt median/P95 : "
                  << callback_stats.median << " / " << callback_stats.p95 << " ms\n";
        std::cout << "Delivery median/P95    : "
                  << delivery_stats.median << " / " << delivery_stats.p95 << " ms\n";
        std::cout << "Failed reads           : " << camera.failed_reads() << "\n";
        std::cout << "GPU-backed             : " << (first->valid() ? "YES" : "NO") << "\n";
        std::cout << "OpenCV                 : NOT USED\n";

        const bool pass =
            source_fps >= 28.5 &&
            callback_fps >= 28.5 &&
            consumer_fps >= 28.5 &&
            camera.failed_reads() == 0 &&
            first->valid();

        std::cout << "Production verdict     : " << (pass ? "PASS" : "BELOW TARGET") << "\n";

        camera.stop();

        std::cout << "\nCamera stopped. Original legacy property #19 was requested to be restored.\n";
        std::cout << "Press ENTER to close...";
        std::cin.get();

        winrt::uninit_apartment();
        return pass ? 0 : 2;

    } catch (winrt::hresult_error const& e) {
        std::cerr << "WinRT error: " << winrt::to_string(e.message()) << "\n";
    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
    }

    std::cout << "Press ENTER to close...";
    std::cin.get();
    return 1;
}
