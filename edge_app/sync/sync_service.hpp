#pragma once
#include "sync/attendance_sync.hpp"
#include "sync/gallery_sync.hpp"
namespace edge_app
{
// Single-worker orchestrator. Only dirty/auth flags are read from another thread.
class SyncService
{
  public:
    SyncService(OutboxRepository &outbox, GalleryRepository &gallery, Transport &http, std::string device,
                std::string model_hash);
    void set_token(std::string const &token);
    void step(std::int64_t now);
    bool take_gallery_dirty()
    {
        return gallery_.take_dirty();
    }
    bool auth_blocked() const
    {
        return blocked_.load();
    }

  private:
    Transport &http_;
    AttendanceSync attendance_;
    GallerySync gallery_;
    std::string token_;
    std::atomic_bool blocked_{false};
};
} // namespace edge_app
