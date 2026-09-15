#pragma once

#include <cstddef>
#include <vector>

namespace vision_runtime::bytetrack {

struct Match {
    std::size_t row = 0;
    std::size_t col = 0;
};

struct AssignmentResult {
    std::vector<Match> matches;
    std::vector<std::size_t> unmatched_rows;
    std::vector<std::size_t> unmatched_cols;
};

// Dense min-cost assignment with a cost limit. The solver uses a rectangular
// Hungarian formulation augmented with dummy columns, so any pair above
// max_cost remains unmatched. This matches ByteTrack's required semantics
// (lapjv(..., cost_limit=threshold)) without a Python/lap dependency.
AssignmentResult linear_assignment(
    std::vector<float> const& row_major_costs,
    std::size_t rows,
    std::size_t cols,
    float max_cost
);

} // namespace vision_runtime::bytetrack
