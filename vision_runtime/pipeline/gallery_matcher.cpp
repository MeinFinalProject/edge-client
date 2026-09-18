#include "pipeline/gallery_matcher.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision_runtime::pipeline
{
void GalleryMatcher::replace(std::vector<GalleryTemplate> const &templates)
{
    GalleryMatcher next;
    for (auto const &t : templates)
    {
        if (t.identity_id.empty())
            throw std::invalid_argument("Empty gallery identity");
        next.identities_.push_back(t.identity_id);
    }
    std::sort(next.identities_.begin(), next.identities_.end());
    next.identities_.erase(std::unique(next.identities_.begin(), next.identities_.end()), next.identities_.end());
    next.matrix_.reserve(templates.size());
    for (auto const &t : templates)
    {
        for (float v : t.embedding)
            if (!std::isfinite(v))
                throw std::invalid_argument("Non-finite embedding");
        next.matrix_.push_back(arcface::normalize_embedding(t.embedding));
        next.owners_.push_back(std::lower_bound(next.identities_.begin(), next.identities_.end(), t.identity_id) -
                               next.identities_.begin());
    }
    next.best_.resize(next.identities_.size());
    *this = std::move(next);
}
GalleryMatch GalleryMatcher::match(std::array<float, arcface::kEmbeddingSize> const &input)
{
    GalleryMatch out;
    if (matrix_.empty())
        return out;
    double squared_norm = 0;
    for (float v : input)
    {
        if (!std::isfinite(v))
            throw std::invalid_argument("Non-finite query");
        squared_norm += double(v) * v;
    }
    const auto norm = std::sqrt(squared_norm);
    // Individually valid observations can cancel when averaged. There is no
    // direction to compare in that case; consume the attempt without a match.
    if (norm < 1.0e-12)
        return out;
    auto query = input;
    const float inverse_norm = static_cast<float>(1.0 / norm);
    for (auto &v : query)
        v *= inverse_norm;
    std::fill(best_.begin(), best_.end(), -1.F);
    for (std::size_t n = 0; n < matrix_.size(); ++n)
    {
        double dot = 0;
        for (std::size_t j = 0; j < query.size(); ++j)
            dot += double(query[j]) * matrix_[n][j];
        best_[owners_[n]] = std::max(best_[owners_[n]], std::clamp(static_cast<float>(dot), -1.F, 1.F));
    }
    for (std::size_t i = 0; i < best_.size(); ++i)
    {
        if (!out.valid || best_[i] > out.best_similarity)
        {
            out.second_similarity = out.best_similarity;
            out.best_similarity = best_[i];
            out.identity_id = identities_[i];
            out.valid = true;
        }
        else
            out.second_similarity = std::max(out.second_similarity, best_[i]);
    }
    out.margin = identities_.size() == 1 ? 2.F : out.best_similarity - out.second_similarity;
    return out;
}
} // namespace vision_runtime::pipeline
