#include "domain/gallery_snapshot.hpp"
#include <cmath>
#include <stdexcept>
namespace edge_app
{
Embedding normalize_embedding(Embedding embedding)
{
    double square = 0;
    for (float value : embedding)
        square += static_cast<double>(value) * value;
    const double norm = std::sqrt(square);
    if (!std::isfinite(norm) || norm < 1.0e-12)
        throw std::invalid_argument("Invalid gallery embedding norm");
    const float inverse = static_cast<float>(1.0 / norm);
    for (auto &value : embedding)
        value *= inverse;
    return embedding;
}
} // namespace edge_app
