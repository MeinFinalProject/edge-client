#include "bytetrack/assignment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vision_runtime::bytetrack {

AssignmentResult linear_assignment(
    std::vector<float> const& costs,
    std::size_t rows,
    std::size_t cols,
    float max_cost
) {
    AssignmentResult result{};
    if (costs.size() != rows * cols) {
        throw std::invalid_argument("linear_assignment: cost matrix size mismatch");
    }
    if (rows == 0 || cols == 0) {
        result.unmatched_rows.resize(rows);
        result.unmatched_cols.resize(cols);
        for (std::size_t i = 0; i < rows; ++i) result.unmatched_rows[i] = i;
        for (std::size_t j = 0; j < cols; ++j) result.unmatched_cols[j] = j;
        return result;
    }

    // We solve rows x (real cols + one dummy pool of `rows` columns), so the
    // Hungarian implementation always has columns >= rows. Dummy cost is just
    // above max_cost; invalid real pairs are made much more expensive.
    const std::size_t aug_cols = cols + rows;
    const double dummy_cost = static_cast<double>(max_cost) + 1e-6;
    const double invalid_cost = dummy_cost + 1e6;

    // 1-indexed standard rectangular Hungarian algorithm (n <= m).
    std::vector<double> u(rows + 1, 0.0), v(aug_cols + 1, 0.0);
    std::vector<std::size_t> p(aug_cols + 1, 0), way(aug_cols + 1, 0);

    auto a = [&](std::size_t i1, std::size_t j1) -> double {
        const std::size_t i = i1 - 1;
        const std::size_t j = j1 - 1;
        if (j >= cols) return dummy_cost;
        const float c = costs[i * cols + j];
        if (!std::isfinite(c) || c > max_cost) return invalid_cost;
        return static_cast<double>(c);
    };

    for (std::size_t i = 1; i <= rows; ++i) {
        p[0] = i;
        std::size_t j0 = 0;
        std::vector<double> minv(aug_cols + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(aug_cols + 1, false);
        do {
            used[j0] = true;
            const std::size_t i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            std::size_t j1 = 0;
            for (std::size_t j = 1; j <= aug_cols; ++j) {
                if (used[j]) continue;
                const double cur = a(i0, j) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (std::size_t j = 0; j <= aug_cols; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const std::size_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int> row_to_col(rows, -1);
    std::vector<int> col_to_row(cols, -1);
    for (std::size_t j = 1; j <= aug_cols; ++j) {
        if (p[j] == 0) continue;
        const std::size_t row = p[j] - 1;
        const std::size_t col = j - 1;
        if (col < cols) {
            const float c = costs[row * cols + col];
            if (std::isfinite(c) && c <= max_cost) {
                row_to_col[row] = static_cast<int>(col);
                col_to_row[col] = static_cast<int>(row);
                result.matches.push_back({row, col});
            }
        }
    }

    for (std::size_t i = 0; i < rows; ++i) {
        if (row_to_col[i] < 0) result.unmatched_rows.push_back(i);
    }
    for (std::size_t j = 0; j < cols; ++j) {
        if (col_to_row[j] < 0) result.unmatched_cols.push_back(j);
    }
    return result;
}

} // namespace vision_runtime::bytetrack
