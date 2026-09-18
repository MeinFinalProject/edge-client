#include "storage/gallery_repository.hpp"
#include "storage/database_internal.hpp"
#include <cstring>
namespace edge_app
{
using namespace detail;
void GalleryRepository::replace(GallerySnapshot const &g)
{
    if (g.template_ids.size() != g.templates.size() || g.version.empty())
        throw std::invalid_argument("Invalid gallery snapshot");
    // Validate/normalize before opening the write transaction, preserving old data on failure.
    std::vector<GalleryTemplate> values = g.templates;
    for (auto &t : values)
    {
        if (t.identity_id.empty())
            throw std::invalid_argument("Empty identity");
        t.embedding = normalize_embedding(t.embedding);
    }
    std::lock_guard lock(database_.impl_->mutex);
    Transaction tx(database_.impl_->db);
    sql(database_.impl_->db, "DELETE FROM gallery_template; DELETE FROM gallery_state;");
    Statement state(database_.impl_->db, "INSERT INTO gallery_state VALUES(1,?,?,?,?)");
    state.text(1, g.version);
    state.text(2, g.etag);
    state.text(3, g.model_id);
    state.text(4, g.model_sha256);
    state.step();
    Statement s(database_.impl_->db, "INSERT INTO gallery_template VALUES(?,?,?)");
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        s.text(1, g.template_ids[i]);
        s.text(2, values[i].identity_id);
        s.blob(3, values[i].embedding.data(), 2048);
        s.step();
        s.reset();
    }
    tx.commit();
}
GallerySnapshot GalleryRepository::load()
{
    std::lock_guard lock(database_.impl_->mutex);
    GallerySnapshot g;
    Statement s(database_.impl_->db, "SELECT version,etag,model_id,model_hash FROM gallery_state WHERE singleton=1");
    if (!s.step())
        return g;
    g.version = s.string(0);
    g.etag = s.string(1);
    g.model_id = s.string(2);
    g.model_sha256 = s.string(3);
    Statement t(database_.impl_->db,
                "SELECT template_id,identity_id,embedding FROM gallery_template ORDER BY template_id");
    while (t.step())
    {
        GalleryTemplate item;
        item.identity_id = t.string(1);
        if (sqlite3_column_bytes(t.s, 2) != 2048)
            throw std::runtime_error("Corrupt gallery blob");
        std::memcpy(item.embedding.data(), sqlite3_column_blob(t.s, 2), 2048);
        g.template_ids.push_back(t.string(0));
        g.templates.push_back(std::move(item));
    }
    return g;
}
GalleryMetadata GalleryRepository::load_metadata()
{
    std::lock_guard lock(database_.impl_->mutex);
    Statement s(database_.impl_->db, "SELECT version,etag,model_id,model_hash FROM gallery_state WHERE singleton=1");
    if (!s.step())
        return {};
    return {s.string(0), s.string(1), s.string(2), s.string(3)};
}

} // namespace edge_app
