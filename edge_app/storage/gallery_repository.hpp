#pragma once
#include "domain/gallery_snapshot.hpp"
#include "storage/database.hpp"
namespace edge_app
{
class GalleryRepository
{
  public:
    explicit GalleryRepository(Database &database) : database_(database)
    {
    }
    void replace(GallerySnapshot const &gallery);
    GallerySnapshot load();
    GalleryMetadata load_metadata();

  private:
    Database &database_;
};
} // namespace edge_app
