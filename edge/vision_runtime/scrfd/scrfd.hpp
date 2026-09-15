#pragma once

#include "scrfd/detector.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace vision_runtime {

struct ScrfdConfig {
    std::filesystem::path model_path;

    std::uint32_t input_width = 640;
    std::uint32_t input_height = 640;

    float score_threshold = 0.50F;
    float nms_threshold = 0.40F;
};

struct ScrfdTiming {
    // Waktu wall-clock Evaluate(). Nilai ini mencakup GPU wait untuk
    // preprocessing yang disubmit sebelumnya, inferensi, dan sinkronisasi
    // output WinML kembali ke CPU untuk post-processing.
    double evaluate_ms = 0.0;

    // Decode anchor + threshold + NMS + mapping ke koordinat camera.
    double postprocess_ms = 0.0;

    // detect() end-to-end dari submit preprocessing sampai hasil final.
    double total_ms = 0.0;
};

class ScrfdDetector final : public IDetector {
public:
    explicit ScrfdDetector(ScrfdConfig config);
    ~ScrfdDetector() override;

    ScrfdDetector(ScrfdDetector const&) = delete;
    ScrfdDetector& operator=(ScrfdDetector const&) = delete;
    ScrfdDetector(ScrfdDetector&&) = delete;
    ScrfdDetector& operator=(ScrfdDetector&&) = delete;

    std::vector<FaceDetection> detect(
        GpuFrame const& frame
    ) override;

    [[nodiscard]] ScrfdTiming last_timing() const noexcept;
    [[nodiscard]] ScrfdConfig config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vision_runtime
