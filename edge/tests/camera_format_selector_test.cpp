#include "camera/camera.hpp"
#include "camera/format_selector.hpp"

#include <iostream>
#include <limits>

namespace {

int failures = 0;

void expect(bool condition, char const* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

} // namespace

int main() {
    using vision_runtime::CameraStreamPreference;
    using vision_runtime::camera_detail::FormatScore;
    using vision_runtime::camera_detail::is_better_candidate;
    using vision_runtime::camera_detail::is_supported_fps;
    using vision_runtime::camera_detail::normalize_subtype;
    using vision_runtime::camera_detail::stream_matches;
    using vision_runtime::camera_detail::stream_rank;
    using vision_runtime::camera_detail::subtype_rank;

    expect(normalize_subtype(L"nV12") == L"NV12", "normalize subtype");

    expect(subtype_rank(L"NV12") == 0, "NV12 is first choice");
    expect(subtype_rank(L"yuy2") == 1, "ranking is case-insensitive");
    expect(subtype_rank(L"BGRA8") == 2, "BGRA8 rank");
    expect(subtype_rank(L"unknown") == 20, "unknown uncompressed rank");
    expect(subtype_rank(L"MJPG") == 50, "MJPG decode penalty");
    expect(subtype_rank(L"H264") == 60, "H264 decode penalty");

    expect(
        stream_rank(CameraStreamPreference::VideoPreview) == 0,
        "preview stream rank"
    );
    expect(
        stream_rank(CameraStreamPreference::VideoRecord) == 1,
        "record stream rank"
    );
    expect(
        stream_matches(
            CameraStreamPreference::VideoRecord,
            CameraStreamPreference::Auto
        ),
        "auto accepts record"
    );
    expect(
        stream_matches(
            CameraStreamPreference::VideoPreview,
            CameraStreamPreference::VideoPreview
        ),
        "preview accepts preview"
    );
    expect(
        !stream_matches(
            CameraStreamPreference::VideoPreview,
            CameraStreamPreference::VideoRecord
        ),
        "record rejects preview"
    );

    expect(
        is_better_candidate({0.0, 1, 50}, {0.5, 0, 0}),
        "FPS error wins before format preferences"
    );
    expect(
        is_better_candidate({0.50005, 0, 50}, {0.5, 1, 0}),
        "near-equal FPS uses stream tie-break"
    );
    expect(
        is_better_candidate({0.5, 0, 0}, {0.5, 0, 50}),
        "equal stream uses subtype tie-break"
    );
    expect(
        !is_better_candidate({0.5, 1, 50}, {0.5, 0, 0}),
        "worse tie-break does not replace incumbent"
    );

    expect(is_supported_fps(1.0), "FPS tolerance includes boundary");
    expect(!is_supported_fps(1.0001), "FPS tolerance rejects above boundary");
    expect(
        is_supported_fps(std::numeric_limits<double>::quiet_NaN()),
        "selector preserves legacy NaN behavior"
    );

    if (failures != 0) {
        std::cerr << failures << " camera format selector test(s) failed\n";
        return 1;
    }

    std::cout << "camera format selector tests passed\n";
    return 0;
}
