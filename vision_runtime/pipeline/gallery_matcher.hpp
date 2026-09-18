#pragma once
#include "pipeline/biometric_attendance_runtime.hpp"

namespace vision_runtime::pipeline
{
struct GalleryMatch
{
    bool valid = false;
    std::string identity_id;
    float best_similarity = -1, second_similarity = -1, margin = 0;
};

// Owner-thread only. Template math is contiguous; string grouping happens at load.
class GalleryMatcher
{
  public:
    void replace(std::vector<GalleryTemplate> const &templates);
    // A finite near-zero aggregate has no direction and returns valid=false.
    // Invalid/non-finite inputs and invalid stored templates still throw.
    GalleryMatch match(std::array<float, arcface::kEmbeddingSize> const &query);
    std::size_t size() const noexcept
    {
        return matrix_.size();
    }

  private:
    std::vector<std::array<float, arcface::kEmbeddingSize>> matrix_;
    std::vector<std::size_t> owners_;
    std::vector<std::string> identities_;
    std::vector<float> best_;
};
} // namespace vision_runtime::pipeline
