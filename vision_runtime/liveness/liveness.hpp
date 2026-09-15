#pragma once

#include "arcface/arcface.hpp"     // SimilarityTransform2D
#include "common/gpu_frame.hpp"
#include "common/vision_types.hpp"
#include "liveness/model_contract.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vision_runtime::liveness {

struct LivenessConfig {
    std::filesystem::path model_path;

    // InsightFace liveness.onnx contract.
    std::uint32_t input_width  = 80;
    std::uint32_t input_height = 80;

    // TA development-data operating point; not a population accuracy claim.
    float live_threshold = kTaLiveThreshold;
};

struct LivenessTiming {
    double evaluate_ms = 0.0;
    double total_ms    = 0.0;
};

enum class LivenessDecision {
    Live,
    Spoof,
    InputRejected,
};

[[nodiscard]] inline LivenessDecision classify_score(float score, float threshold = kTaLiveThreshold) {
    if (!std::isfinite(score) || score < 0 || score > 1 ||
        !std::isfinite(threshold) || threshold < 0 || threshold > 1)
        throw std::invalid_argument("Liveness score/threshold must be finite probabilities");
    return score >= threshold ? LivenessDecision::Live : LivenessDecision::Spoof;
}

struct LivenessResult {
    // Raw ONNX probability [0,1]; empty for InputRejected (inference not run).
    std::optional<float> live_score;

    LivenessDecision decision = LivenessDecision::InputRejected;
    float out_of_bounds_ratio = 0.0F;

    // The similarity transform used for alignment.
    arcface::SimilarityTransform2D transform{};

    LivenessTiming timing{};
};

[[nodiscard]] char const* to_string(LivenessDecision decision) noexcept;

// InsightFace liveness 5-point reference template for 80x80 aligned faces.
// Same landmark ordering as ArcFace: left eye, right eye, nose, left mouth,
// right mouth. Different target coordinates optimized for the liveness model.
[[nodiscard]] arcface::SimilarityTransform2D estimate_liveness_transform(
    FaceDetection const& observation,
    std::uint32_t output_width  = 80,
    std::uint32_t output_height = 80
);

// InsightFace liveness evaluator.
//
// Contract from the official InsightFace Python addon adapter:
//   input       : [batch,3,80,80] float32
//   channel     : RGB
//   normalize   : pixel / 255.0
//   crop        : bilinear uint8, replicated border; reject >30% missing area
//   output      : [batch] float32 — live_score (already a probability)
//   decision    : live_score >= threshold → Live, else Spoof
//
// GPU path:
//   camera NV12 D3D11 texture
//     -> shared NV12 staging texture
//     -> D3D11On12 compute: 5-point aligned warp + NV12->RGB + /255 + NCHW
//     -> D3D12 TensorFloat
//     -> 4-byte coverage ratio to CPU; skip inference for rejected crops
//     -> WinML / DirectML for accepted crops
//     -> single live_score read back to CPU for accepted crops
class LivenessDetector {
  public:
    explicit LivenessDetector(LivenessConfig config);
    ~LivenessDetector();

    LivenessDetector(LivenessDetector const&) = delete;
    LivenessDetector& operator=(LivenessDetector const&) = delete;
    LivenessDetector(LivenessDetector&&) = delete;
    LivenessDetector& operator=(LivenessDetector&&) = delete;

    [[nodiscard]] LivenessResult evaluate(
        GpuFrame const& frame,
        FaceDetection const& observation
    );

    [[nodiscard]] std::vector<LivenessResult> evaluate_many(
        GpuFrame const& frame,
        std::vector<FaceDetection> const& observations
    );

    [[nodiscard]] LivenessConfig config() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime::liveness
