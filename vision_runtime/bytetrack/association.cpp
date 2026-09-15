#include "bytetrack/association.hpp"

#include <algorithm>
#include <chrono>

namespace vision_runtime::bytetrack {
namespace {

float iou(AssociationItem const& a, AssociationItem const& b) noexcept {
    const float left = std::max(a.x1, b.x1);
    const float top = std::max(a.y1, b.y1);
    const float right = std::min(a.x2, b.x2);
    const float bottom = std::min(a.y2, b.y2);
    const float iw = std::max(0.0F, right - left);
    const float ih = std::max(0.0F, bottom - top);
    const float inter = iw * ih;
    const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0.0F ? inter / uni : 0.0F;
}

} // namespace

void cost_matrix(
    std::span<AssociationItem const> tracks,
    std::span<AssociationItem const> detections,
    bool fuse_score,
    std::vector<float>& output,
    AssociationTiming& timing
) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();

    output.resize(tracks.size() * detections.size());
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        for (std::size_t j = 0; j < detections.size(); ++j) {
            const float overlap = iou(tracks[i], detections[j]);
            output[i * detections.size() + j] =
                fuse_score ? (1.0F - overlap * detections[j].score)
                           : (1.0F - overlap);
        }
    }

    const auto end = Clock::now();
    timing.last_call_ms = std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace vision_runtime::bytetrack
