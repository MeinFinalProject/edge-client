#include "app/attendance_coordinator.hpp"
#include "sync/wire_format.hpp"
#include <stdexcept>

namespace edge_app
{
AttendanceCoordinator::AttendanceCoordinator(OutboxRepository &outbox, GalleryRepository &gallery, std::string device,
                                             std::string model_hash)
    : outbox_(outbox), gallery_(gallery), device_(std::move(device)), model_hash_(std::move(model_hash))
{
}
void AttendanceCoordinator::refresh_gallery(GalleryInstaller const &install)
{
    auto snapshot = gallery_.load();
    if (!snapshot.version.empty() && snapshot.model_sha256 != model_hash_)
        throw std::runtime_error("Gallery model differs from the active ArcFace model");
    std::vector<vision_runtime::pipeline::GalleryTemplate> templates;
    templates.reserve(snapshot.templates.size());
    for (auto &item : snapshot.templates)
        templates.push_back({std::move(item.identity_id), item.embedding});
    install(std::move(templates));
    version_ = std::move(snapshot.version);
    template_count_ = snapshot.templates.size();
}
AttendanceRecord AttendanceCoordinator::record(vision_runtime::pipeline::AttendanceEvent const &event)
{
    AttendanceObservation observation;
    observation.occurred_at_utc_ms = event.occurred_at_utc_ms;
    observation.camera_frame_id = event.camera_frame_id;
    observation.track_id = event.track_id;
    observation.identity_id = event.identity_id;
    observation.similarity = event.similarity;
    observation.similarity_margin = event.similarity_margin;
    observation.pad_median_p_real = event.pad_median_p_real;
    auto record = envelope(observation, device_, version_);
    outbox_.enqueue(record);
    return record;
}
} // namespace edge_app
