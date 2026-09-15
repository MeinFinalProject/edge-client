#include "scrfd/detector.hpp"

#include <stdexcept>

namespace vision_runtime {

void validate_gpu_frame(GpuFrame const& frame) {
    if (!frame.valid()) {
        throw std::invalid_argument(
            "Detector menerima GpuFrame yang tidak valid"
        );
    }

    if (frame.width == 0 || frame.height == 0) {
        throw std::invalid_argument(
            "Detector menerima GpuFrame dengan dimensi nol"
        );
    }

    if (frame.dxgi_format == DXGI_FORMAT_UNKNOWN) {
        throw std::invalid_argument(
            "Detector menerima GpuFrame dengan DXGI format UNKNOWN"
        );
    }
}

} // namespace vision_runtime
