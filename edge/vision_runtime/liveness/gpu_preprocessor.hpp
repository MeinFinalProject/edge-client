#pragma once

#include "arcface/arcface.hpp"     // SimilarityTransform2D
#include "common/gpu_frame.hpp"

#include <cstdint>
#include <memory>

#include <winrt/Windows.AI.MachineLearning.h>

namespace vision_runtime::liveness {

struct LivenessPreprocessConfig {
    std::uint32_t input_width  = 80;
    std::uint32_t input_height = 80;
};

// Only this scalar is read back; the aligned image/tensor stays on the GPU.
struct LivenessPreprocessResult {
    winrt::Windows::AI::MachineLearning::TensorFloat tensor{nullptr};
    float out_of_bounds_ratio = 0.0F;
    [[nodiscard]] bool input_rejected() const noexcept { return out_of_bounds_ratio > 0.30F; }
};

// GPU aligned-face tensorizer for InsightFace liveness.onnx.
//
// Same D3D11On12 cross-API pipeline as ArcFace but with:
// - 80x80 output instead of 112x112
// - pixel / 255.0 normalization (the NV12→RGB conversion already yields [0,1])
// - InsightFace liveness destination landmarks
// - BORDER_REPLICATE and bilinear uint8 crop semantics
// - GPU mask reduction; only a 4-byte missing-area ratio returns to CPU
class LivenessGpuPreprocessor {
  public:
    explicit LivenessGpuPreprocessor(
        GpuFrame const& first_frame,
        LivenessPreprocessConfig config = {}
    );
    ~LivenessGpuPreprocessor();

    LivenessGpuPreprocessor(LivenessGpuPreprocessor const&) = delete;
    LivenessGpuPreprocessor& operator=(LivenessGpuPreprocessor const&) = delete;
    LivenessGpuPreprocessor(LivenessGpuPreprocessor&&) = delete;
    LivenessGpuPreprocessor& operator=(LivenessGpuPreprocessor&&) = delete;

    void prepare_frame(GpuFrame const& frame);

    [[nodiscard]] LivenessPreprocessResult
    preprocess_aligned(arcface::SimilarityTransform2D const& transform);

    [[nodiscard]] winrt::Windows::AI::MachineLearning::LearningModelDevice
    learning_model_device() const;

    [[nodiscard]] LivenessPreprocessConfig config() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime::liveness
