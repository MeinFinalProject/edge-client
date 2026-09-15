#include "arcface/arcface.hpp"
#include "arcface/gpu_preprocessor.hpp"

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

namespace vision_runtime::arcface
{
    namespace
    {

        using Clock = std::chrono::steady_clock;
        using namespace winrt::Windows::AI::MachineLearning;

        constexpr std::array<Point2f, 5> kArcFaceDst112{{
            {38.2946F, 51.6963F},
            {73.5318F, 51.5014F},
            {56.0252F, 71.7366F},
            {41.5493F, 92.3655F},
            {70.7299F, 92.2041F},
        }};

        template <class Rep, class Period>
        double ms(std::chrono::duration<Rep, Period> d)
        {
            return std::chrono::duration<double, std::milli>(d).count();
        }

        std::string describe_hresult(winrt::hresult_error const &error)
        {
            std::ostringstream message;
            message
                << "HRESULT 0x"
                << std::hex << std::uppercase
                << static_cast<std::uint32_t>(error.code())
                << std::dec
                << " (" << winrt::to_string(error.message()) << ")";
            return message.str();
        }

        [[noreturn]] void rethrow_stage(
            char const *stage,
            winrt::hresult_error const &error)
        {
            throw std::runtime_error(
                std::string{"ArcFace stage '"} + stage + "' gagal: " +
                describe_hresult(error));
        }

        bool dynamic_or_one(std::int64_t v) noexcept
        {
            return v == 1 || v <= 0;
        }

        void validate_input_shape(ILearningModelFeatureDescriptor const &feature)
        {
            auto tensor = feature.as<TensorFeatureDescriptor>();
            auto shape = tensor.Shape();
            if (shape.Size() != 4)
            {
                throw std::runtime_error("Input ArcFace harus rank-4 NCHW");
            }
            if (
                !dynamic_or_one(shape.GetAt(0)) ||
                shape.GetAt(1) != 3 ||
                shape.GetAt(2) != 112 ||
                shape.GetAt(3) != 112)
            {
                throw std::runtime_error(
                    "Input w600k_r50 harus kompatibel dengan [1,3,112,112]");
            }
        }

        void validate_output_shape(ILearningModelFeatureDescriptor const &feature)
        {
            auto tensor = feature.as<TensorFeatureDescriptor>();
            auto shape = tensor.Shape();
            if (shape.Size() != 2)
            {
                throw std::runtime_error("Output ArcFace harus rank-2 [N,512]");
            }
            if (!dynamic_or_one(shape.GetAt(0)) || shape.GetAt(1) != 512)
            {
                throw std::runtime_error(
                    "Output w600k_r50 harus kompatibel dengan [1,512]");
            }
        }

        std::array<float, kEmbeddingSize> copy_embedding(TensorFloat const &tensor)
        {
            auto view = tensor.GetAsVectorView();
            if (view.Size() != kEmbeddingSize)
            {
                throw std::runtime_error(
                    "Output ArcFace diharapkan 512 float, ditemukan " +
                    std::to_string(view.Size()));
            }

            std::array<float, kEmbeddingSize> out{};
            const auto copied = view.GetMany(0, winrt::array_view<float>(out));
            if (copied != out.size())
            {
                throw std::runtime_error("WinML hanya mengembalikan sebagian embedding ArcFace");
            }

            for (float v : out)
            {
                if (!std::isfinite(v))
                {
                    throw std::runtime_error("Embedding ArcFace mengandung NaN/Inf");
                }
            }
            return out;
        }

        // Solve a 4x4 linear system using partial-pivot Gaussian elimination.
        std::array<double, 4> solve4(
            std::array<std::array<double, 4>, 4> a,
            std::array<double, 4> b)
        {
            for (int col = 0; col < 4; ++col)
            {
                int pivot = col;
                double best = std::abs(a[col][col]);
                for (int row = col + 1; row < 4; ++row)
                {
                    const double cand = std::abs(a[row][col]);
                    if (cand > best)
                    {
                        best = cand;
                        pivot = row;
                    }
                }
                if (best < 1.0e-10)
                {
                    throw std::runtime_error("Landmark ArcFace menghasilkan transform degenerat");
                }
                if (pivot != col)
                {
                    std::swap(a[pivot], a[col]);
                    std::swap(b[pivot], b[col]);
                }

                const double diag = a[col][col];
                for (int j = col; j < 4; ++j)
                    a[col][j] /= diag;
                b[col] /= diag;

                for (int row = 0; row < 4; ++row)
                {
                    if (row == col)
                        continue;
                    const double factor = a[row][col];
                    if (std::abs(factor) < 1.0e-18)
                        continue;
                    for (int j = col; j < 4; ++j)
                    {
                        a[row][j] -= factor * a[col][j];
                    }
                    b[row] -= factor * b[col];
                }
            }
            return b;
        }

    } // namespace

    SimilarityTransform2D estimate_arcface_transform(
        FaceDetection const &observation,
        std::uint32_t output_width,
        std::uint32_t output_height)
    {
        if (output_width != 112 || output_height != 112)
        {
            throw std::invalid_argument(
                "v4.7 ArcFace alignment dikunci ke template InsightFace 112x112");
        }

        // Least-squares similarity transform:
        //   u = a*x - b*y + tx
        //   v = b*x + a*y + ty
        // parameter vector = [a, b, tx, ty].
        std::array<std::array<double, 4>, 4> ata{};
        std::array<double, 4> atb{};

        auto accumulate_row = [&](std::array<double, 4> const &row, double target)
        {
            for (int i = 0; i < 4; ++i)
            {
                atb[i] += row[i] * target;
                for (int j = 0; j < 4; ++j)
                {
                    ata[i][j] += row[i] * row[j];
                }
            }
        };

        for (std::size_t i = 0; i < 5; ++i)
        {
            const double x = observation.landmarks[i].x;
            const double y = observation.landmarks[i].y;
            const double u = kArcFaceDst112[i].x;
            const double v = kArcFaceDst112[i].y;

            if (
                !std::isfinite(x) || !std::isfinite(y) ||
                !std::isfinite(u) || !std::isfinite(v))
            {
                throw std::invalid_argument("Landmark ArcFace mengandung NaN/Inf");
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
        if (denom < 1.0e-12)
        {
            throw std::runtime_error("Scale similarity transform ArcFace mendekati nol");
        }

        SimilarityTransform2D out{};
        out.src_to_dst = {
            static_cast<float>(a),
            static_cast<float>(-b),
            static_cast<float>(tx),
            static_cast<float>(b),
            static_cast<float>(a),
            static_cast<float>(ty),
        };

        const double i00 = a / denom;
        const double i01 = b / denom;
        const double i10 = -b / denom;
        const double i11 = a / denom;
        const double i02 = -(i00 * tx + i01 * ty);
        const double i12 = -(i10 * tx + i11 * ty);

        out.dst_to_src = {
            static_cast<float>(i00),
            static_cast<float>(i01),
            static_cast<float>(i02),
            static_cast<float>(i10),
            static_cast<float>(i11),
            static_cast<float>(i12),
        };

        double sq = 0.0;
        for (std::size_t i = 0; i < 5; ++i)
        {
            const double x = observation.landmarks[i].x;
            const double y = observation.landmarks[i].y;
            const double uhat = a * x - b * y + tx;
            const double vhat = b * x + a * y + ty;
            const double du = uhat - kArcFaceDst112[i].x;
            const double dv = vhat - kArcFaceDst112[i].y;
            sq += du * du + dv * dv;
        }
        out.rmse = static_cast<float>(std::sqrt(sq / 5.0));
        return out;
    }

    float cosine_similarity(
        std::array<float, kEmbeddingSize> const &a,
        std::array<float, kEmbeddingSize> const &b) noexcept
    {
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (std::size_t i = 0; i < kEmbeddingSize; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            na += static_cast<double>(a[i]) * a[i];
            nb += static_cast<double>(b[i]) * b[i];
        }
        if (na <= 0.0 || nb <= 0.0)
            return 0.0F;
        return static_cast<float>(dot / std::sqrt(na * nb));
    }

    std::array<float, kEmbeddingSize> normalize_embedding(
        std::array<float, kEmbeddingSize> embedding)
    {
        double sq = 0.0;
        for (float v : embedding)
            sq += static_cast<double>(v) * v;
        const double norm = std::sqrt(sq);
        if (!std::isfinite(norm) || norm < 1.0e-12)
        {
            throw std::runtime_error("Norm embedding ArcFace nol/tidak valid");
        }
        const float inv = static_cast<float>(1.0 / norm);
        for (auto &v : embedding)
            v *= inv;
        return embedding;
    }

    struct ArcFaceRecognizer::Impl
    {
        explicit Impl(ArcFaceConfig cfg) : config(std::move(cfg))
        {
            validate_config();
            load_model();
        }

        ArcFaceConfig config{};

        LearningModel model{nullptr};
        LearningModelSession session{nullptr};
        LearningModelBinding binding{nullptr};
        winrt::hstring input_name;
        winrt::hstring output_name;

        std::unique_ptr<ArcFaceGpuPreprocessor> preprocessor;
        bool input_bound = false;

        void validate_config()
        {
            if (config.model_path.empty())
            {
                throw std::invalid_argument("ArcFaceConfig.model_path belum diisi");
            }
            if (!std::filesystem::exists(config.model_path))
            {
                throw std::invalid_argument(
                    "Model ArcFace tidak ditemukan: " + config.model_path.string());
            }
            if (config.input_width != 112 || config.input_height != 112)
            {
                throw std::invalid_argument("w600k_r50 native path dikunci ke 112x112");
            }
        }

        void load_model()
        {
            const auto absolute = std::filesystem::absolute(config.model_path);
            try
            {
                model = LearningModel::LoadFromFilePath(
                    winrt::hstring(absolute.wstring()));
            }
            catch (winrt::hresult_error const &error)
            {
                rethrow_stage("LearningModel::LoadFromFilePath", error);
            }

            auto inputs = model.InputFeatures();
            auto outputs = model.OutputFeatures();
            if (inputs.Size() != 1)
            {
                throw std::runtime_error("w600k_r50 diharapkan tepat 1 input");
            }
            if (outputs.Size() != 1)
            {
                throw std::runtime_error("w600k_r50 diharapkan tepat 1 output");
            }

            auto in = inputs.GetAt(0);
            auto out = outputs.GetAt(0);
            if (in.Kind() != LearningModelFeatureKind::Tensor)
            {
                throw std::runtime_error("Input ArcFace bukan tensor");
            }
            if (out.Kind() != LearningModelFeatureKind::Tensor)
            {
                throw std::runtime_error("Output ArcFace bukan tensor");
            }

            validate_input_shape(in);
            validate_output_shape(out);
            input_name = in.Name();
            output_name = out.Name();
        }

        void ensure_runtime(GpuFrame const &frame)
        {
            if (preprocessor)
                return;

            ArcFacePreprocessConfig pcfg{};
            pcfg.input_width = config.input_width;
            pcfg.input_height = config.input_height;

            try
            {
                preprocessor = std::make_unique<ArcFaceGpuPreprocessor>(frame, pcfg);
            }
            catch (winrt::hresult_error const &error)
            {
                rethrow_stage("create GPU preprocessor", error);
            }
            catch (std::exception const &error)
            {
                throw std::runtime_error(
                    std::string{"ArcFace stage 'create GPU preprocessor' gagal: "} +
                    error.what());
            }

            try
            {
                session = LearningModelSession{
                    model,
                    preprocessor->learning_model_device()};
            }
            catch (winrt::hresult_error const &error)
            {
                rethrow_stage("LearningModelSession(GPU device)", error);
            }

            try
            {
                binding = LearningModelBinding{session};
            }
            catch (winrt::hresult_error const &error)
            {
                rethrow_stage("LearningModelBinding", error);
            }
        }

        ArcFaceResult evaluate_prepared(FaceDetection const &observation)
        {
            const auto total_begin = Clock::now();
            const auto transform = estimate_arcface_transform(
                observation,
                config.input_width,
                config.input_height);

            TensorFloat tensor{nullptr};
            try
            {
                tensor = preprocessor->preprocess_aligned(transform);
            }
            catch (winrt::hresult_error const &error)
            {
                rethrow_stage("GPU aligned-face preprocessing", error);
            }

            if (!input_bound)
            {
                try
                {
                    binding.Bind(input_name, tensor);
                    input_bound = true;
                }
                catch (winrt::hresult_error const &error)
                {
                    rethrow_stage("Bind input tensor", error);
                }
            }

            LearningModelEvaluationResult eval_result{nullptr};
            const auto eval_begin = Clock::now();
            try
            {
                eval_result = session.Evaluate(binding, L"arcface-w600k-r50");
            }
            catch (winrt::hresult_error const &error)
            {
                rethrow_stage("LearningModelSession::Evaluate", error);
            }
            const auto eval_end = Clock::now();

            auto value = eval_result.Outputs().Lookup(output_name);
            TensorFloat output{nullptr};
            try
            {
                output = value.as<TensorFloat>();
            }
            catch (...)
            {
                throw std::runtime_error("Output ArcFace bukan TensorFloat");
            }

            ArcFaceResult result{};
            result.transform = transform;
            auto raw = copy_embedding(output);

            double sq = 0.0;
            for (float v : raw)
                sq += static_cast<double>(v) * v;
            result.raw_embedding_norm = static_cast<float>(std::sqrt(sq));
            result.embedding = normalize_embedding(std::move(raw));

            const auto total_end = Clock::now();
            result.timing.evaluate_ms = ms(eval_end - eval_begin);
            result.timing.total_ms = ms(total_end - total_begin);
            return result;
        }

        std::vector<ArcFaceResult> evaluate_many(
            GpuFrame const &frame,
            std::vector<FaceDetection> const &observations)
        {
            if (!frame.valid())
            {
                throw std::invalid_argument("GpuFrame ArcFace tidak valid");
            }
            if (observations.empty())
                return {};

            ensure_runtime(frame);

            try
            {
                preprocessor->prepare_frame(frame);
            }
            catch (winrt::hresult_error const &error)
            {
                rethrow_stage("prepare GPU frame", error);
            }

            std::vector<ArcFaceResult> results;
            results.reserve(observations.size());
            for (auto const &obs : observations)
            {
                results.push_back(evaluate_prepared(obs));
            }
            return results;
        }
    };

    ArcFaceRecognizer::ArcFaceRecognizer(ArcFaceConfig config)
        : impl_(std::make_unique<Impl>(std::move(config))) {}

    ArcFaceRecognizer::~ArcFaceRecognizer() = default;

    ArcFaceResult ArcFaceRecognizer::evaluate(
        GpuFrame const &frame,
        FaceDetection const &observation)
    {
        auto results = impl_->evaluate_many(
            frame,
            std::vector<FaceDetection>{observation});
        return std::move(results.front());
    }

    std::vector<ArcFaceResult> ArcFaceRecognizer::evaluate_many(
        GpuFrame const &frame,
        std::vector<FaceDetection> const &observations)
    {
        return impl_->evaluate_many(frame, observations);
    }

    ArcFaceConfig ArcFaceRecognizer::config() const
    {
        return impl_->config;
    }

} // namespace vision_runtime::arcface
