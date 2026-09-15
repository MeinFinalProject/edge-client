#pragma once

#include "arcface/arcface.hpp"

#include <cstdint>
#include <memory>

#include <winrt/Windows.AI.MachineLearning.h>

namespace vision_runtime::arcface {

struct ArcFacePreprocessConfig {
    std::uint32_t input_width = 112;
    std::uint32_t input_height = 112;
};

// GPU aligned-face tensorizer for InsightFace w600k_r50.
//
// Input: GPU-backed camera NV12.
// Warp : inverse similarity transform derived from the current SCRFD 5 points.
// Color: NV12 -> RGB.
// Norm : (pixel_0_255 - 127.5) / 127.5.
// Layout: NCHW float32, 1x3x112x112.
class ArcFaceGpuPreprocessor {
public:
    explicit ArcFaceGpuPreprocessor(
        GpuFrame const& first_frame,
        ArcFacePreprocessConfig config = {}
    );
    ~ArcFaceGpuPreprocessor();

    ArcFaceGpuPreprocessor(ArcFaceGpuPreprocessor const&) = delete;
    ArcFaceGpuPreprocessor& operator=(ArcFaceGpuPreprocessor const&) = delete;
    ArcFaceGpuPreprocessor(ArcFaceGpuPreprocessor&&) = delete;
    ArcFaceGpuPreprocessor& operator=(ArcFaceGpuPreprocessor&&) = delete;

    void prepare_frame(GpuFrame const& frame);

    [[nodiscard]] winrt::Windows::AI::MachineLearning::TensorFloat
    preprocess_aligned(SimilarityTransform2D const& transform);

    [[nodiscard]] winrt::Windows::AI::MachineLearning::LearningModelDevice
    learning_model_device() const;

    [[nodiscard]] ArcFacePreprocessConfig config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime::arcface
