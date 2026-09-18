#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <vector>
namespace edge_app
{
inline constexpr std::size_t embedding_dimension = 512;
using Embedding = std::array<float, embedding_dimension>;
struct GalleryTemplate
{
    std::string identity_id;
    Embedding embedding{};
};
struct GallerySnapshot
{
    std::string version, etag, model_id, model_sha256;
    std::vector<std::string> template_ids;
    std::vector<GalleryTemplate> templates;
};
// Small revision information used for conditional HTTP requests; no embeddings.
struct GalleryMetadata
{
    std::string version, etag, model_id, model_sha256;
};
// Same float32 normalization contract as the runtime, without a vision dependency.
Embedding normalize_embedding(Embedding embedding);
} // namespace edge_app
