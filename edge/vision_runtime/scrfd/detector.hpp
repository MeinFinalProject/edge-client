#pragma once

#include "common/gpu_frame.hpp"
#include "common/vision_types.hpp"

#include <cstdint>
#include <vector>

namespace vision_runtime {

// Boundary native pipeline.
// Camera menghasilkan GpuFrame; concrete SCRFD detector nantinya menerima
// GpuFrame secara langsung sehingga raw image tidak perlu kembali ke Python.
class IDetector {
public:
    virtual ~IDetector() = default;

    virtual std::vector<FaceDetection> detect(
        GpuFrame const& frame
    ) = 0;
};

// Utility guard yang dipakai concrete detector sebelum dispatch GPU.
void validate_gpu_frame(GpuFrame const& frame);

} // namespace vision_runtime
