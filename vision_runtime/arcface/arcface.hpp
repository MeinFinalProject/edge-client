#pragma once

#include "common/gpu_frame.hpp"
#include "common/vision_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace vision_runtime::arcface {

constexpr std::size_t kEmbeddingSize = 512;

struct SimilarityTransform2D {
    // Forward mapping from source camera coordinates to ArcFace output pixels:
    //   u = m00*x + m01*y + m02
    //   v = m10*x + m11*y + m12
    std::array<float, 6> src_to_dst{};

    // Inverse mapping used by the GPU warp:
    //   x = i00*u + i01*v + i02
    //   y = i10*u + i11*v + i12
    std::array<float, 6> dst_to_src{};

    // RMS landmark fitting error in ArcFace output-pixel coordinates.
    float rmse = 0.0F;
};

struct ArcFaceConfig {
    std::filesystem::path model_path;
    std::uint32_t input_width = 112;
    std::uint32_t input_height = 112;
};

struct ArcFaceTiming {
    // Wall time around WinML Evaluate(). Because GPU preprocessing is queued on
    // the same D3D12 command queue, this can include queued preprocess work and
    // output synchronization as well as model inference.
    double evaluate_ms = 0.0;

    // alignment calculation -> GPU preprocess submission -> 512-float readback
    // -> CPU L2 normalization.
    double total_ms = 0.0;
};

struct ArcFaceResult {
    std::array<float, kEmbeddingSize> embedding{}; // L2-normalized
    float raw_embedding_norm = 0.0F;
    SimilarityTransform2D transform{};
    ArcFaceTiming timing{};
};

// InsightFace ArcFace 5-point reference template for 112x112 aligned faces:
//   left eye, right eye, nose, left mouth, right mouth.
[[nodiscard]] SimilarityTransform2D estimate_arcface_transform(
    FaceDetection const& observation,
    std::uint32_t output_width = 112,
    std::uint32_t output_height = 112
);

[[nodiscard]] float cosine_similarity(
    std::array<float, kEmbeddingSize> const& a,
    std::array<float, kEmbeddingSize> const& b
) noexcept;

[[nodiscard]] std::array<float, kEmbeddingSize> normalize_embedding(
    std::array<float, kEmbeddingSize> embedding
);

// Native InsightFace w600k_r50 ArcFace evaluator.
//
// Official InsightFace recognition path for this model family:
//   current SCRFD 5-point landmarks
//     -> similarity alignment to the 112x112 ArcFace template
//     -> RGB
//     -> (pixel - 127.5) / 127.5
//     -> NCHW float32
//     -> w600k_r50.onnx / WinML / DirectML
//     -> 512-float embedding
//     -> CPU L2 normalization / cosine similarity
//
// The native path performs the image warp and tensorization from the same
// GPU-backed NV12 camera frame; it does not create an OpenCV/NumPy image in the
// application processing path. Only the 512-float embedding is read to CPU.
class ArcFaceRecognizer {
public:
    explicit ArcFaceRecognizer(ArcFaceConfig config);
    ~ArcFaceRecognizer();

    ArcFaceRecognizer(ArcFaceRecognizer const&) = delete;
    ArcFaceRecognizer& operator=(ArcFaceRecognizer const&) = delete;
    ArcFaceRecognizer(ArcFaceRecognizer&&) = delete;
    ArcFaceRecognizer& operator=(ArcFaceRecognizer&&) = delete;

    [[nodiscard]] ArcFaceResult evaluate(
        GpuFrame const& frame,
        FaceDetection const& observation
    );

    // Copy the current camera frame into the shareable NV12 staging texture
    // once, then evaluate multiple aligned faces from that exact frame.
    [[nodiscard]] std::vector<ArcFaceResult> evaluate_many(
        GpuFrame const& frame,
        std::vector<FaceDetection> const& observations
    );

    [[nodiscard]] ArcFaceConfig config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime::arcface
