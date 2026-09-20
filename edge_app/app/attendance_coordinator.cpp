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
    if (!pending_.empty())
        throw std::logic_error("Persist pending attendance before reloading gallery");
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
    stage({event});
    return *commit_next();
}
void AttendanceCoordinator::stage(std::vector<vision_runtime::pipeline::AttendanceEvent> const &events)
{
    if (!pending_.empty())
        throw std::logic_error("Persist pending attendance before accepting another frame");
    std::deque<AttendanceRecord> next;
    for (auto const &event : events)
    {
        AttendanceObservation observation;
        observation.occurred_at_utc_ms = event.occurred_at_utc_ms;
        observation.camera_frame_id = event.camera_frame_id;
        observation.track_id = event.track_id;
        observation.identity_id = event.identity_id;
        observation.similarity = event.similarity;
        observation.similarity_margin = event.similarity_margin;
        observation.pad_median_p_real = event.pad_median_p_real;
        next.push_back(envelope(observation, device_, version_));
    }
    pending_.swap(next);
}
std::optional<AttendanceRecord> AttendanceCoordinator::commit_next()
{
    if (pending_.empty())
        return std::nullopt;
    auto record = pending_.front();
    outbox_.ensure_enqueued(record);
    pending_.pop_front();
    return record;
}
} // namespace edge_app
