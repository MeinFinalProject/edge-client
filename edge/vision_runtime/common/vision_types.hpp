#pragma once

#include <array>

namespace vision_runtime {

struct Point2f {
    float x = 0.0F;
    float y = 0.0F;
};

// Coordinate system: original camera frame, XYXY bounding box.
// Landmarks are SCRFD 5-point landmarks in the same coordinate system.
struct FaceDetection {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float score = 0.0F;
    std::array<Point2f, 5> landmarks{};
};

} // namespace vision_runtime
