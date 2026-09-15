#pragma once

#include <string>
#include <string_view>

namespace vision_runtime {

enum class CameraStreamPreference;

} // namespace vision_runtime

namespace vision_runtime::camera_detail {

struct FormatScore {
    double fps_error = 0.0;
    int stream = 0;
    int subtype = 0;
};

[[nodiscard]] std::wstring normalize_subtype(std::wstring value);
[[nodiscard]] int subtype_rank(std::wstring_view subtype);
[[nodiscard]] int stream_rank(CameraStreamPreference stream) noexcept;
[[nodiscard]] bool stream_matches(
    CameraStreamPreference actual,
    CameraStreamPreference requested
) noexcept;
[[nodiscard]] bool is_better_candidate(
    FormatScore candidate,
    FormatScore incumbent
) noexcept;
[[nodiscard]] bool is_supported_fps(double fps_error) noexcept;

} // namespace vision_runtime::camera_detail
