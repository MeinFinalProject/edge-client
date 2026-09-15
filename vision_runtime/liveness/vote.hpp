#pragma once
#include "liveness/liveness.hpp"
#include <cstddef>

namespace vision_runtime::liveness {
// Counts only evaluated observations toward a temporal decision.
struct VoteCounts {
    std::size_t attempts = 0;
    std::size_t real = 0;
    std::size_t spoof = 0;
    std::size_t input_rejected = 0;

    bool add(LivenessDecision decision) noexcept {
        if (decision == LivenessDecision::InputRejected) { ++input_rejected; return false; }
        ++attempts;
        if (decision == LivenessDecision::Live) ++real; else ++spoof;
        return true;
    }
};
} // namespace vision_runtime::liveness
