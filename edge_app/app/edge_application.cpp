#include "app/edge_application.hpp"
#include "app/attendance_coordinator.hpp"
#include "sync/sync_service.hpp"
#include "sync/sync_worker.hpp"
#include "sync/wire_format.hpp"
#include <ostream>
#include <stdexcept>

namespace edge_app
{
struct EdgeApplication::Impl
{
    ApplicationConfig config;
    std::string model_hash;
    // Declaration order is lifetime order: borrowers are destroyed before Database.
    Database database;
    OutboxRepository outbox;
    GalleryRepository gallery;
    AttendanceCoordinator coordinator;
    explicit Impl(ApplicationConfig c)
        : config(std::move(c)), model_hash(verify_model_files(config)), database(config.database), outbox(database),
          gallery(database), coordinator(outbox, gallery, config.device_id, model_hash)
    {
    }
    std::unique_ptr<Transport> make_transport() const
    {
        return config.base_url.empty() ? nullptr : http_transport(config.base_url, config.allow_loopback_http);
    }
    void describe(std::ostream &output)
    {
        auto snapshot = gallery.load();
        if (!snapshot.version.empty() && snapshot.model_sha256 != model_hash)
            throw std::runtime_error("Persisted gallery uses a different ArcFace model; import a compatible snapshot");
        output << "SQLite " << database.sqlite_version() << "; pending=" << outbox.pending_count()
               << "; templates=" << snapshot.templates.size() << '\n';
        if (config.vision.allow_uncalibrated_thresholds)
            output << "DEVELOPMENT: recognition thresholds are not validated for deployment.\n";
    }
};
EdgeApplication::EdgeApplication(ApplicationConfig config) : impl_(std::make_unique<Impl>(std::move(config)))
{
}
EdgeApplication::~EdgeApplication() = default;
void EdgeApplication::validate(std::ostream &output)
{
    auto transport = impl_->make_transport(); // Validate URL/filter setup without sending a request.
    impl_->describe(output);
    output << "Config, model files/hash, database and gallery metadata: PASS (camera/inference not started).\n";
}
void EdgeApplication::import_gallery(std::filesystem::path const &document)
{
    impl_->gallery.replace(parse_gallery_file(read_document(document), impl_->model_hash));
}
void EdgeApplication::run(std::atomic_bool const &stopping, std::ostream &output)
{
    auto &app = *impl_;
    auto transport = app.make_transport();
    app.describe(output);
    vision_runtime::pipeline::BiometricAttendanceRuntime runtime(app.config.vision);
    auto install = [&](auto templates) { runtime.set_gallery(std::move(templates)); };
    app.coordinator.refresh_gallery(install);
    // Worker must be joined before service/transport/repos are destroyed.
    std::unique_ptr<SyncService> sync;
    std::unique_ptr<SyncWorker> worker;
    if (transport)
    {
        sync = std::make_unique<SyncService>(app.outbox, app.gallery, *transport, app.config.device_id, app.model_hash);
        worker = std::make_unique<SyncWorker>(*sync, app.config.credential_file);
    }
    runtime.start();
    output << "Capture started. Empty gallery waits for import/sync. Ctrl+C to stop.\n";
    auto status_due = std::chrono::steady_clock::now();
    while (!stopping)
    {
        if (sync && sync->take_gallery_dirty())
        {
            app.coordinator.refresh_gallery(install);
            output << "Gallery applied; templates=" << app.coordinator.gallery_size() << '\n';
        }
        auto frame = runtime.wait_for_frame(std::chrono::milliseconds(1000));
        if (frame)
            for (auto const &event : frame->attendance_events)
            {
                auto record = app.coordinator.record(event);
                output << "COMMITTED event_id=" << record.id << " frame=" << event.camera_frame_id << '\n';
                if (worker)
                    worker->wake();
            }
        auto now = std::chrono::steady_clock::now();
        if (now >= status_due)
        {
            output << "sync=" << (worker ? worker->status() : "offline") << '\n';
            status_due = now + std::chrono::seconds(10);
        }
    }
    runtime.stop();
    worker.reset();
}
} // namespace edge_app
