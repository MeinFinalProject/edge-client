#pragma once
#include "storage/gallery_repository.hpp"
#include "sync/http_client.hpp"
#include <atomic>
namespace edge_app
{
class GallerySync
{
  public:
    GallerySync(GalleryRepository &gallery, Transport &http, std::string model_hash);
    SyncResult step(std::int64_t now);
    void reset_poll()
    {
        gallery_due_ = 0;
    }
    bool take_dirty()
    {
        return dirty_.exchange(false);
    }

  private:
    GalleryRepository &gallery_;
    Transport &http_;
    std::string hash_;
    std::atomic_bool dirty_{false};
    std::int64_t gallery_due_ = 0;
    int gallery_attempts_ = 0;
};
} // namespace edge_app
