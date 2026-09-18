#pragma once
#include "pipeline/biometric_attendance_runtime.hpp"
#include "storage/gallery_repository.hpp"
#include "storage/outbox_repository.hpp"
#include <functional>

namespace edge_app
{
// Owner-thread only. Connects vision results to durable application records;
// neither repositories nor sync components hold a reference to the runtime.
class AttendanceCoordinator
{
  public:
    using GalleryInstaller = std::function<void(std::vector<vision_runtime::pipeline::GalleryTemplate>)>;
    AttendanceCoordinator(OutboxRepository &outbox, GalleryRepository &gallery, std::string device,
                          std::string model_hash);
    // Call between frames. Publish the new version only after installation succeeds.
    void refresh_gallery(GalleryInstaller const &install);
    // Returns only after COMMIT; caller can then print success/wake the worker.
    AttendanceRecord record(vision_runtime::pipeline::AttendanceEvent const &event);
    std::string const &gallery_version() const
    {
        return version_;
    }
    std::size_t gallery_size() const
    {
        return template_count_;
    }

  private:
    OutboxRepository &outbox_;
    GalleryRepository &gallery_;
    std::string device_, model_hash_, version_;
    std::size_t template_count_ = 0;
};
} // namespace edge_app
