#include "liveness/liveness.hpp"
#include "liveness/gpu_preprocessor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.AI.MachineLearning.h>

namespace vision_runtime::liveness {
namespace {

using Clock = std::chrono::steady_clock;
using namespace winrt::Windows::AI::MachineLearning;

constexpr vision_runtime::Point2f kLivenessDst80[5] = {
    {32.03F, 38.06F},
    {47.89F, 37.98F},
    {40.01F, 47.08F},
    {33.50F, 56.36F},
    {46.63F, 56.29F},
};

template <class Rep, class Period>
double ms(std::chrono::duration<Rep, Period> d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

std::string describe_hresult(winrt::hresult_error const& error) {
    std::ostringstream message;
    message << "HRESULT 0x" << std::hex << std::uppercase
            << static_cast<std::uint32_t>(error.code()) << std::dec << " ("
            << winrt::to_string(error.message()) << ")";
    return message.str();
}

[[noreturn]] void rethrow_stage(char const* stage, winrt::hresult_error const& error) {
    throw std::runtime_error(std::string{"Liveness stage '"} + stage +
                             "' gagal: " + describe_hresult(error));
}

bool dynamic_or_one(std::int64_t v) noexcept {
    return v == 1 || v <= 0;
}

void validate_input_shape(ILearningModelFeatureDescriptor const& feature) {
    auto tensor = feature.as<TensorFeatureDescriptor>();
    auto shape = tensor.Shape();
    if (shape.Size() != 4) {
        throw std::runtime_error("Input Liveness harus rank-4 NCHW");
    }
    if (!dynamic_or_one(shape.GetAt(0)) || shape.GetAt(1) != 3 ||
        shape.GetAt(2) != 80 || shape.GetAt(3) != 80) {
        throw std::runtime_error("Input Liveness.onnx harus kompatibel dengan [1,3,80,80]");
    }
}

void validate_output_shape(ILearningModelFeatureDescriptor const& feature) {
    auto tensor = feature.as<TensorFeatureDescriptor>();
    auto shape = tensor.Shape();
    // Liveness model outputs [batch_size] float
    if (shape.Size() != 1 && shape.Size() != 2) {
        throw std::runtime_error("Output Liveness harus scalar/1D");
    }
    if (!dynamic_or_one(shape.GetAt(0))) {
        throw std::runtime_error("Output Liveness.onnx harus batch size 1");
    }
}

float copy_live_score(TensorFloat const& tensor) {
    auto view = tensor.GetAsVectorView();
    if (view.Size() != 1) {
        throw std::runtime_error("Output Liveness.onnx diharapkan 1 float, ditemukan " +
                                 std::to_string(view.Size()));
    }

    std::array<float, 1> out{};
    const auto copied = view.GetMany(0, winrt::array_view<float>(out));
    if (copied != out.size()) {
        throw std::runtime_error("WinML hanya mengembalikan sebagian output Liveness");
    }

    float v = out[0];
    if (!std::isfinite(v)) {
        throw std::runtime_error("Output Liveness mengandung NaN/Inf");
    }
    if (v < -1.0e-4F || v > 1.0001F) {
        throw std::runtime_error("Output Liveness bukan probabilitas [0,1]");
    }

    return std::clamp(v, 0.0F, 1.0F);
}

// Reuse the exact same 4x4 solver algorithm from ArcFace.
std::array<double, 4> solve4(std::array<std::array<double, 4>, 4> a, std::array<double, 4> b) {
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        double best = std::abs(a[col][col]);
        for (int row = col + 1; row < 4; ++row) {
            const double cand = std::abs(a[row][col]);
            if (cand > best) {
                best = cand;
                pivot = row;
            }
        }
        if (best < 1.0e-10) {
            throw std::runtime_error("Landmark Liveness menghasilkan transform degenerat");
        }
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(b[pivot], b[col]);
        }

        const double diag = a[col][col];
        for (int j = col; j < 4; ++j) a[col][j] /= diag;
        b[col] /= diag;

        for (int row = 0; row < 4; ++row) {
            if (row == col) continue;
            const double factor = a[row][col];
            if (std::abs(factor) < 1.0e-18) continue;
            for (int j = col; j < 4; ++j) {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    return b;
}

} // namespace

char const* to_string(LivenessDecision decision) noexcept {
    if (decision == LivenessDecision::InputRejected) return "INPUT_REJECTED";
    return decision == LivenessDecision::Live ? "LIVE" : "SPOOF";
}

arcface::SimilarityTransform2D estimate_liveness_transform(FaceDetection const& observation,
                                                           std::uint32_t output_width,
                                                           std::uint32_t output_height) {
    if (output_width != 80 || output_height != 80) {
        throw std::invalid_argument("Liveness alignment dikunci ke template InsightFace 80x80");
    }

    std::array<std::array<double, 4>, 4> ata{};
    std::array<double, 4> atb{};

    auto accumulate_row = [&](std::array<double, 4> const& row, double target) {
        for (int i = 0; i < 4; ++i) {
            atb[i] += row[i] * target;
            for (int j = 0; j < 4; ++j) {
                ata[i][j] += row[i] * row[j];
            }
        }
    };

    for (std::size_t i = 0; i < 5; ++i) {
        const double x = observation.landmarks[i].x;
        const double y = observation.landmarks[i].y;
        const double u = kLivenessDst80[i].x;
        const double v = kLivenessDst80[i].y;

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(u) || !std::isfinite(v)) {
            throw std::invalid_argument("Landmark Liveness mengandung NaN/Inf");
        }

        accumulate_row({x, -y, 1.0, 0.0}, u);
        accumulate_row({y, x, 0.0, 1.0}, v);
    }

    const auto p = solve4(ata, atb);
    const double a = p[0];
    const double b = p[1];
    const double tx = p[2];
    const double ty = p[3];
    const double denom = a * a + b * b;
    if (denom < 1.0e-12) {
        throw std::runtime_error("Scale similarity transform Liveness mendekati nol");
    }

    arcface::SimilarityTransform2D out{};
    out.src_to_dst = {
        static_cast<float>(a),  static_cast<float>(-b), static_cast<float>(tx),
        static_cast<float>(b),  static_cast<float>(a),  static_cast<float>(ty),
    };

    const double i00 = a / denom;
    const double i01 = b / denom;
    const double i10 = -b / denom;
    const double i11 = a / denom;
    const double i02 = -(i00 * tx + i01 * ty);
    const double i12 = -(i10 * tx + i11 * ty);

    out.dst_to_src = {
        static_cast<float>(i00), static_cast<float>(i01), static_cast<float>(i02),
        static_cast<float>(i10), static_cast<float>(i11), static_cast<float>(i12),
    };

    double sq = 0.0;
    for (std::size_t i = 0; i < 5; ++i) {
        const double x = observation.landmarks[i].x;
        const double y = observation.landmarks[i].y;
        const double uhat = a * x - b * y + tx;
        const double vhat = b * x + a * y + ty;
        const double du = uhat - kLivenessDst80[i].x;
        const double dv = vhat - kLivenessDst80[i].y;
        sq += du * du + dv * dv;
    }
    out.rmse = static_cast<float>(std::sqrt(sq / 5.0));
    return out;
}

struct LivenessDetector::Impl {
    explicit Impl(LivenessConfig cfg) : config(std::move(cfg)) {
        validate_config();
        load_model();
    }

    LivenessConfig config{};

    LearningModel model{nullptr};
    LearningModelSession session{nullptr};
    LearningModelBinding binding{nullptr};
    winrt::hstring input_name;
    winrt::hstring output_name;

    std::unique_ptr<LivenessGpuPreprocessor> preprocessor;
    bool input_bound = false;

    void validate_config() {
        if (config.model_path.empty()) {
            throw std::invalid_argument("LivenessConfig.model_path belum diisi");
        }
        if (!std::filesystem::exists(config.model_path)) {
            throw std::invalid_argument("Model Liveness tidak ditemukan: " + config.model_path.string());
        }
        if (config.input_width != 80 || config.input_height != 80) {
            throw std::invalid_argument("InsightFace liveness dikunci ke 80x80");
        }
        if (!std::isfinite(config.live_threshold) || config.live_threshold < 0.0F ||
            config.live_threshold > 1.0F) {
            throw std::invalid_argument("Liveness threshold harus 0..1");
        }
    }

    void load_model() {
        const auto absolute = std::filesystem::absolute(config.model_path);
        try {
            model = LearningModel::LoadFromFilePath(winrt::hstring(absolute.wstring()));
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("LearningModel::LoadFromFilePath", error);
        }

        auto inputs = model.InputFeatures();
        auto outputs = model.OutputFeatures();
        if (inputs.Size() != 1) {
            throw std::runtime_error("Liveness.onnx diharapkan tepat 1 input");
        }
        if (outputs.Size() != 1) {
            throw std::runtime_error("Liveness.onnx diharapkan tepat 1 output");
        }

        auto in = inputs.GetAt(0);
        auto out = outputs.GetAt(0);
        if (in.Kind() != LearningModelFeatureKind::Tensor) {
            throw std::runtime_error("Input Liveness bukan tensor");
        }
        if (out.Kind() != LearningModelFeatureKind::Tensor) {
            throw std::runtime_error("Output Liveness bukan tensor");
        }

        validate_input_shape(in);
        validate_output_shape(out);
        input_name = in.Name();
        output_name = out.Name();
    }

    void ensure_runtime(GpuFrame const& frame) {
        if (preprocessor) return;

        LivenessPreprocessConfig pcfg{};
        pcfg.input_width = config.input_width;
        pcfg.input_height = config.input_height;

        try {
            preprocessor = std::make_unique<LivenessGpuPreprocessor>(frame, pcfg);
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("create GPU preprocessor", error);
        } catch (std::exception const& error) {
            throw std::runtime_error(std::string{"Liveness stage 'create GPU preprocessor' gagal: "} +
                                     error.what());
        }

        try {
            session = LearningModelSession{model, preprocessor->learning_model_device()};
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("LearningModelSession(GPU device)", error);
        }

        try {
            binding = LearningModelBinding{session};
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("LearningModelBinding", error);
        }
    }

    LivenessResult evaluate_prepared(FaceDetection const& observation) {
        const auto total_begin = Clock::now();
        const auto transform = estimate_liveness_transform(observation, config.input_width, config.input_height);

        LivenessPreprocessResult prepared;
        try {
            prepared = preprocessor->preprocess_aligned(transform);
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("GPU aligned-face preprocessing", error);
        }

        LivenessResult result{};
        result.transform = transform;
        result.out_of_bounds_ratio = prepared.out_of_bounds_ratio;
        if (prepared.input_rejected()) {
            result.timing.total_ms = ms(Clock::now() - total_begin);
            return result; // No inference, no probability, no spoof decision.
        }
        if (!input_bound) {
            try {
                binding.Bind(input_name, prepared.tensor);
                input_bound = true;
            } catch (winrt::hresult_error const& error) {
                rethrow_stage("Bind input tensor", error);
            }
        }

        LearningModelEvaluationResult eval_result{nullptr};
        const auto eval_begin = Clock::now();
        try {
            eval_result = session.Evaluate(binding, L"insightface-liveness");
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("LearningModelSession::Evaluate", error);
        }
        const auto eval_end = Clock::now();

        auto value = eval_result.Outputs().Lookup(output_name);
        TensorFloat output{nullptr};
        try {
            output = value.as<TensorFloat>();
        } catch (...) {
            throw std::runtime_error("Output Liveness bukan TensorFloat");
        }

        result.live_score = copy_live_score(output);

        // InsightFace contract: live_score >= threshold is Live.
        result.decision = classify_score(*result.live_score, config.live_threshold);

        const auto total_end = Clock::now();
        result.timing.evaluate_ms = ms(eval_end - eval_begin);
        result.timing.total_ms = ms(total_end - total_begin);
        return result;
    }

    std::vector<LivenessResult> evaluate_many(GpuFrame const& frame,
                                              std::vector<FaceDetection> const& observations) {
        if (!frame.valid()) {
            throw std::invalid_argument("GpuFrame Liveness tidak valid");
        }
        if (observations.empty()) return {};

        ensure_runtime(frame);

        try {
            preprocessor->prepare_frame(frame);
        } catch (winrt::hresult_error const& error) {
            rethrow_stage("prepare GPU frame", error);
        }

        std::vector<LivenessResult> results;
        results.reserve(observations.size());
        for (auto const& obs : observations) {
            results.push_back(evaluate_prepared(obs));
        }
        return results;
    }
};

LivenessDetector::LivenessDetector(LivenessConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

LivenessDetector::~LivenessDetector() = default;

LivenessResult LivenessDetector::evaluate(GpuFrame const& frame, FaceDetection const& observation) {
    auto results = impl_->evaluate_many(frame, std::vector<FaceDetection>{observation});
    return std::move(results.front());
}

std::vector<LivenessResult> LivenessDetector::evaluate_many(
    GpuFrame const& frame, std::vector<FaceDetection> const& observations) {
    return impl_->evaluate_many(frame, observations);
}

LivenessConfig LivenessDetector::config() const {
    return impl_->config;
}

} // namespace vision_runtime::liveness
