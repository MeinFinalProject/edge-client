#include "camera/format_selector.hpp"

#include "camera/camera.hpp"

#include <algorithm>
#include <cmath>
#include <cwctype>

namespace vision_runtime::camera_detail {
namespace {

constexpr double kMaxFpsError = 1.0;
constexpr double kTieEpsilon = 0.0001;

} // namespace

std::wstring normalize_subtype(std::wstring value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](wchar_t c) { return static_cast<wchar_t>(std::towupper(c)); }
    );
    return value;
}

int subtype_rank(std::wstring_view subtype) {
    auto normalized = normalize_subtype(std::wstring(subtype));

    // Production preference: uncompressed format suitable for a GPU surface.
    if (normalized == L"NV12") return 0;
    if (normalized == L"YUY2") return 1;
    if (normalized == L"BGRA8") return 2;
    if (normalized == L"ARGB32") return 3;
    if (normalized == L"RGB32") return 4;

    // Compressed formats are lower priority because they add a decode stage.
    if (normalized == L"MJPG") return 50;
    if (normalized == L"H264") return 60;

    return 20;
}

int stream_rank(CameraStreamPreference stream) noexcept {
    if (stream == CameraStreamPreference::VideoPreview) return 0;
    if (stream == CameraStreamPreference::VideoRecord) return 1;
    return 10;
}

bool stream_matches(
    CameraStreamPreference actual,
    CameraStreamPreference requested
) noexcept {
    return requested == CameraStreamPreference::Auto || actual == requested;
}

bool is_better_candidate(
    FormatScore candidate,
    FormatScore incumbent
) noexcept {
    return candidate.fps_error < incumbent.fps_error ||
           (
               std::abs(candidate.fps_error - incumbent.fps_error) <
                   kTieEpsilon &&
               candidate.stream < incumbent.stream
           ) ||
           (
               std::abs(candidate.fps_error - incumbent.fps_error) <
                   kTieEpsilon &&
               candidate.stream == incumbent.stream &&
               candidate.subtype < incumbent.subtype
           );
}

bool is_supported_fps(double fps_error) noexcept {
    // Preserve the original rejection predicate exactly. Media Foundation
    // yields finite rates, but this also keeps behavior stable for NaN input.
    return !(fps_error > kMaxFpsError);
}

} // namespace vision_runtime::camera_detail
