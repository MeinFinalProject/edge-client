#include "scrfd/scrfd.hpp"

#include "scrfd/gpu_preprocessor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <utility>
#include <vector>

#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.Collections.h>

namespace vision_runtime {
namespace {

using namespace winrt::Windows::AI::MachineLearning;
using Clock = std::chrono::steady_clock;

struct TensorBlob {
    std::wstring name;
    std::vector<std::int64_t> shape;
    std::vector<float> data;
};

struct LevelBlobs {
    int stride = 0;
    std::size_t anchors = 0;
    TensorBlob scores;
    TensorBlob bbox;
    TensorBlob kps;
    bool has_scores = false;
    bool has_bbox = false;
    bool has_kps = false;
};

struct ModelCandidate {
    FaceDetection det;
};

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

std::vector<std::int64_t> copy_shape(
    winrt::Windows::Foundation::Collections::IVectorView<std::int64_t> const& shape
) {
    std::vector<std::int64_t> result;
    result.reserve(shape.Size());

    for (std::uint32_t i = 0; i < shape.Size(); ++i) {
        result.push_back(shape.GetAt(i));
    }

    return result;
}

std::vector<float> copy_tensor_data(TensorFloat const& tensor) {
    auto view = tensor.GetAsVectorView();
    std::vector<float> data(view.Size());

    if (!data.empty()) {
        const auto copied = view.GetMany(
            0,
            winrt::array_view<float>(data)
        );

        if (copied != data.size()) {
            throw std::runtime_error(
                "WinML hanya mengembalikan sebagian output tensor"
            );
        }
    }

    return data;
}

std::size_t last_dimension(TensorBlob const& blob) {
    if (blob.shape.empty()) {
        return 0;
    }

    const auto value = blob.shape.back();
    if (value <= 0) {
        return 0;
    }

    return static_cast<std::size_t>(value);
}

std::size_t row_count(TensorBlob const& blob) {
    const auto dim = last_dimension(blob);
    if (dim == 0 || blob.data.size() % dim != 0) {
        return 0;
    }
    return blob.data.size() / dim;
}

int stride_for_anchor_count(
    std::size_t anchors,
    std::uint32_t input_width,
    std::uint32_t input_height
) {
    constexpr std::array<int, 3> strides{8, 16, 32};
    constexpr std::size_t num_anchors = 2;

    for (const int stride : strides) {
        const auto expected =
            static_cast<std::size_t>(input_width / stride) *
            static_cast<std::size_t>(input_height / stride) *
            num_anchors;

        if (anchors == expected) {
            return stride;
        }
    }

    return 0;
}

float clampf(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

float iou_inclusive(FaceDetection const& a, FaceDetection const& b) {
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);

    const float width = std::max(0.0F, xx2 - xx1 + 1.0F);
    const float height = std::max(0.0F, yy2 - yy1 + 1.0F);
    const float intersection = width * height;

    const float area_a =
        std::max(0.0F, a.x2 - a.x1 + 1.0F) *
        std::max(0.0F, a.y2 - a.y1 + 1.0F);
    const float area_b =
        std::max(0.0F, b.x2 - b.x1 + 1.0F) *
        std::max(0.0F, b.y2 - b.y1 + 1.0F);

    const float denominator = area_a + area_b - intersection;
    if (denominator <= 0.0F) {
        return 0.0F;
    }

    return intersection / denominator;
}

std::string describe_hresult(winrt::hresult_error const& error) {
    std::ostringstream message;
    message << "HRESULT 0x"
            << std::hex << std::uppercase
            << static_cast<std::uint32_t>(error.code())
            << std::dec
            << " (" << winrt::to_string(error.message()) << ")";
    return message.str();
}

[[noreturn]] void rethrow_stage(
    char const* stage,
    winrt::hresult_error const& error
) {
    throw std::runtime_error(
        std::string{"SCRFD stage '"} + stage + "' gagal: " +
        describe_hresult(error)
    );
}

std::vector<ModelCandidate> nms(
    std::vector<ModelCandidate> candidates,
    float threshold
) {
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](ModelCandidate const& lhs, ModelCandidate const& rhs) {
            return lhs.det.score > rhs.det.score;
        }
    );

    std::vector<ModelCandidate> kept;
    kept.reserve(candidates.size());

    for (auto const& candidate : candidates) {
        bool suppressed = false;

        for (auto const& accepted : kept) {
            if (iou_inclusive(candidate.det, accepted.det) > threshold) {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            kept.push_back(candidate);
        }
    }

    return kept;
}

} // namespace

struct ScrfdDetector::Impl {
    explicit Impl(ScrfdConfig cfg) : config(std::move(cfg)) {
        validate_config();
        load_model();
    }

    ScrfdConfig config{};
    ScrfdTiming timing{};

    LearningModel model{nullptr};
    LearningModelSession session{nullptr};
    LearningModelBinding binding{nullptr};
    winrt::hstring input_name;

    std::unique_ptr<ScrfdGpuPreprocessor> preprocessor;

    void validate_config() {
        if (config.model_path.empty()) {
            throw std::invalid_argument(
                "ScrfdConfig.model_path belum diisi"
            );
        }

        if (!std::filesystem::exists(config.model_path)) {
            throw std::invalid_argument(
                "Model SCRFD tidak ditemukan: " +
                config.model_path.string()
            );
        }

        if (
            config.input_width != 640 ||
            config.input_height != 640
        ) {
            throw std::invalid_argument(
                "SCRFD v3.1 baseline menggunakan original input 640x640"
            );
        }

        if (
            config.score_threshold < 0.0F ||
            config.score_threshold > 1.0F
        ) {
            throw std::invalid_argument(
                "score_threshold harus berada pada rentang 0..1"
            );
        }

        if (
            config.nms_threshold < 0.0F ||
            config.nms_threshold > 1.0F
        ) {
            throw std::invalid_argument(
                "nms_threshold harus berada pada rentang 0..1"
            );
        }
    }

    void load_model() {
        const auto absolute_path = std::filesystem::absolute(config.model_path);

        model = LearningModel::LoadFromFilePath(
            winrt::hstring(absolute_path.wstring())
        );

        auto inputs = model.InputFeatures();
        if (inputs.Size() != 1) {
            throw std::runtime_error(
                "det_500m.onnx diharapkan memiliki tepat satu input"
            );
        }

        auto input = inputs.GetAt(0);
        if (input.Kind() != LearningModelFeatureKind::Tensor) {
            throw std::runtime_error(
                "Input original SCRFD bukan TensorFeatureDescriptor"
            );
        }

        input_name = input.Name();

        auto outputs = model.OutputFeatures();
        if (outputs.Size() != 9) {
            throw std::runtime_error(
                "SCRFD det_500m diharapkan memiliki 9 output, ditemukan " +
                std::to_string(outputs.Size())
            );
        }
    }

    void initialize_runtime(GpuFrame const& first_frame) {
        ScrfdPreprocessConfig preprocess_config{};
        preprocess_config.input_width = config.input_width;
        preprocess_config.input_height = config.input_height;

        try {
            preprocessor = std::make_unique<ScrfdGpuPreprocessor>(
                first_frame,
                preprocess_config
            );
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("create GPU preprocessor", error);
        } catch (std::exception const& error) {
            throw std::runtime_error(
                std::string{"SCRFD stage 'create GPU preprocessor' gagal: "} +
                error.what()
            );
        }

        try {
            session = LearningModelSession{
                model,
                preprocessor->learning_model_device()
            };
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("LearningModelSession(GPU device)", error);
        }

        try {
            binding = LearningModelBinding{session};
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("LearningModelBinding", error);
        }

        TensorFloat tensor{nullptr};
        try {
            tensor = preprocessor->preprocess(first_frame);
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("first GPU preprocess", error);
        } catch (std::exception const& error) {
            throw std::runtime_error(
                std::string{"SCRFD stage 'first GPU preprocess' gagal: "} +
                error.what()
            );
        }

        try {
            binding.Bind(input_name, tensor);
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("Bind input tensor", error);
        }
    }

    std::vector<TensorBlob> evaluate_outputs() {
        LearningModelEvaluationResult result{nullptr};
        try {
            result = session.Evaluate(binding, L"scrfd");
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("LearningModelSession::Evaluate", error);
        }
        auto output_map = result.Outputs();
        auto features = model.OutputFeatures();

        std::vector<TensorBlob> blobs;
        blobs.reserve(features.Size());

        for (std::uint32_t i = 0; i < features.Size(); ++i) {
            auto feature = features.GetAt(i);
            auto value = output_map.Lookup(feature.Name());

            TensorFloat tensor{nullptr};
            try {
                tensor = value.as<TensorFloat>();
            } catch (...) {
                throw std::runtime_error(
                    "Output SCRFD bukan TensorFloat: " +
                    winrt::to_string(feature.Name())
                );
            }

            TensorBlob blob{};
            blob.name = feature.Name().c_str();
            blob.shape = copy_shape(tensor.Shape());
            blob.data = copy_tensor_data(tensor);
            blobs.push_back(std::move(blob));
        }

        return blobs;
    }

    std::array<LevelBlobs, 3> organize_outputs(
        std::vector<TensorBlob> blobs
    ) const {
        std::map<int, LevelBlobs> levels;

        for (auto& blob : blobs) {
            const auto dim = last_dimension(blob);
            const auto rows = row_count(blob);
            const int stride = stride_for_anchor_count(
                rows,
                config.input_width,
                config.input_height
            );

            if (stride == 0) {
                throw std::runtime_error(
                    "Shape output SCRFD tidak cocok dengan stride 8/16/32"
                );
            }

            auto& level = levels[stride];
            level.stride = stride;
            level.anchors = rows;

            if (dim == 1) {
                level.scores = std::move(blob);
                level.has_scores = true;
            } else if (dim == 4) {
                level.bbox = std::move(blob);
                level.has_bbox = true;
            } else if (dim == 10) {
                level.kps = std::move(blob);
                level.has_kps = true;
            } else {
                throw std::runtime_error(
                    "Output SCRFD memiliki dimensi terakhir yang tidak dikenal: " +
                    std::to_string(dim)
                );
            }
        }

        std::array<LevelBlobs, 3> ordered{};
        constexpr std::array<int, 3> strides{8, 16, 32};

        for (std::size_t i = 0; i < strides.size(); ++i) {
            const int stride = strides[i];
            auto found = levels.find(stride);

            if (found == levels.end()) {
                throw std::runtime_error(
                    "Output SCRFD untuk stride " +
                    std::to_string(stride) +
                    " tidak ditemukan"
                );
            }

            if (
                !found->second.has_scores ||
                !found->second.has_bbox ||
                !found->second.has_kps
            ) {
                throw std::runtime_error(
                    "Output SCRFD stride " +
                    std::to_string(stride) +
                    " tidak lengkap"
                );
            }

            ordered[i] = std::move(found->second);
        }

        return ordered;
    }

    std::vector<FaceDetection> decode(
        std::array<LevelBlobs, 3> const& levels,
        GpuFrame const& frame
    ) const {
        std::vector<ModelCandidate> candidates;

        for (auto const& level : levels) {
            const int stride = level.stride;
            const std::size_t feat_width = config.input_width / stride;
            constexpr std::size_t num_anchors = 2;

            const auto& scores = level.scores.data;
            const auto& bbox = level.bbox.data;
            const auto& kps = level.kps.data;

            if (
                scores.size() != level.anchors ||
                bbox.size() != level.anchors * 4 ||
                kps.size() != level.anchors * 10
            ) {
                throw std::runtime_error(
                    "Jumlah elemen output SCRFD tidak konsisten"
                );
            }

            for (std::size_t i = 0; i < level.anchors; ++i) {
                const float score = scores[i];
                if (score < config.score_threshold) {
                    continue;
                }

                // Official SCRFD center order: setiap grid center diulang dua
                // kali karena _num_anchors == 2.
                const std::size_t cell = i / num_anchors;
                const std::size_t grid_x = cell % feat_width;
                const std::size_t grid_y = cell / feat_width;

                const float anchor_x =
                    static_cast<float>(grid_x * stride);
                const float anchor_y =
                    static_cast<float>(grid_y * stride);

                const std::size_t bbox_offset = i * 4;
                const float left = bbox[bbox_offset + 0] * stride;
                const float top = bbox[bbox_offset + 1] * stride;
                const float right = bbox[bbox_offset + 2] * stride;
                const float bottom = bbox[bbox_offset + 3] * stride;

                ModelCandidate candidate{};
                candidate.det.x1 = anchor_x - left;
                candidate.det.y1 = anchor_y - top;
                candidate.det.x2 = anchor_x + right;
                candidate.det.y2 = anchor_y + bottom;
                candidate.det.score = score;

                const std::size_t kps_offset = i * 10;
                for (std::size_t p = 0; p < 5; ++p) {
                    candidate.det.landmarks[p].x =
                        anchor_x + kps[kps_offset + p * 2] * stride;
                    candidate.det.landmarks[p].y =
                        anchor_y + kps[kps_offset + p * 2 + 1] * stride;
                }

                candidates.push_back(candidate);
            }
        }

        auto kept = nms(
            std::move(candidates),
            config.nms_threshold
        );

        const auto transform = preprocessor->transform();
        const float inverse_scale = 1.0F / transform.scale;
        const float max_x = static_cast<float>(frame.width - 1);
        const float max_y = static_cast<float>(frame.height - 1);

        std::vector<FaceDetection> detections;
        detections.reserve(kept.size());

        for (auto& item : kept) {
            auto& det = item.det;

            det.x1 = clampf(
                (det.x1 - transform.pad_x) * inverse_scale,
                0.0F,
                max_x
            );
            det.y1 = clampf(
                (det.y1 - transform.pad_y) * inverse_scale,
                0.0F,
                max_y
            );
            det.x2 = clampf(
                (det.x2 - transform.pad_x) * inverse_scale,
                0.0F,
                max_x
            );
            det.y2 = clampf(
                (det.y2 - transform.pad_y) * inverse_scale,
                0.0F,
                max_y
            );

            for (auto& point : det.landmarks) {
                point.x = clampf(
                    (point.x - transform.pad_x) * inverse_scale,
                    0.0F,
                    max_x
                );
                point.y = clampf(
                    (point.y - transform.pad_y) * inverse_scale,
                    0.0F,
                    max_y
                );
            }

            // Abaikan prediksi yang runtuh sepenuhnya setelah mapping/clamp.
            if (det.x2 > det.x1 && det.y2 > det.y1) {
                detections.push_back(det);
            }
        }

        return detections;
    }

    std::vector<FaceDetection> detect(GpuFrame const& frame) {
        validate_gpu_frame(frame);

        const auto total_begin = Clock::now();

        if (!preprocessor) {
            initialize_runtime(frame);
        } else {
            // Tensor yang dibind ke session adalah resource persisten yang sama.
            // preprocess() hanya memperbarui isi GPU resource tersebut.
            (void)preprocessor->preprocess(frame);
        }

        const auto evaluate_begin = Clock::now();
        auto blobs = evaluate_outputs();
        const auto evaluate_end = Clock::now();

        const auto post_begin = Clock::now();
        auto levels = organize_outputs(std::move(blobs));
        auto detections = decode(levels, frame);
        const auto post_end = Clock::now();

        timing.evaluate_ms = milliseconds(evaluate_end - evaluate_begin);
        timing.postprocess_ms = milliseconds(post_end - post_begin);
        timing.total_ms = milliseconds(post_end - total_begin);

        return detections;
    }
};

ScrfdDetector::ScrfdDetector(ScrfdConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

ScrfdDetector::~ScrfdDetector() = default;

std::vector<FaceDetection> ScrfdDetector::detect(
    GpuFrame const& frame
) {
    return impl_->detect(frame);
}

ScrfdTiming ScrfdDetector::last_timing() const noexcept {
    return impl_->timing;
}

ScrfdConfig ScrfdDetector::config() const {
    return impl_->config;
}

} // namespace vision_runtime
