#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace vision_runtime::bytetrack {

struct AssociationItem {
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float score = 1.0F;
};

struct AssociationTiming {
    double last_call_ms = 0.0;
};

// Native CPU association used by production ByteTrack.
// Output is row-major [tracks x detections]. With fuse_score=true this
// computes ByteTrack's 1 - IoU * detection_score; otherwise 1 - IoU.
void cost_matrix(
    std::span<AssociationItem const> tracks,
    std::span<AssociationItem const> detections,
    bool fuse_score,
    std::vector<float>& output,
    AssociationTiming& timing
);

} // namespace vision_runtime::bytetrack
