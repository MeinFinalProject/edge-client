#pragma once

namespace vision_runtime::liveness {
// TA selection locked on 2026-09-15. Evidence: blueprint/progres/riwayat-pad.md.
// Threshold fitted on S01 development sessions; held-out performance is untested.
inline constexpr float kTaLiveThreshold = 0.95F;
inline constexpr char kModelRelativePath[] = "insightface/addons/liveness.onnx";
inline constexpr char kModelSha256[] = "87a9ac1dbb16a61eec212957e5095e62a8769c1e188af9b0198f253302c4afdb";
inline constexpr char kPreprocessContract[] = "addons_v2_align80_rgb8_linear32_replicate_gpu_oob30";
} // namespace vision_runtime::liveness
