#include "app/edge_application.hpp"
#include "app/attendance_coordinator.hpp"
#include "security/local_data.hpp"
#include "storage/storage_error.hpp"
#include "sync/sync_service.hpp"
#include "sync/sync_worker.hpp"
#include "sync/device_channel.hpp"
#include "sync/wire_format.hpp"
#include <ostream>
#include <stdexcept>
#include <thread>

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
        if (std::filesystem::exists(config.credential_file))
            prepare_private_file(config.credential_file);
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
        if (config.deployment)
            output
                << "DEPLOYMENT: report integrity/model binding checked; calibration validity requires human review.\n";
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
    std::unique_ptr<DeviceChannel> channel;
    if (transport)
    {
        sync = std::make_unique<SyncService>(app.outbox, app.gallery, *transport, app.config.device_id, app.model_hash);
        worker = std::make_unique<SyncWorker>(*sync, app.config.credential_file);
        channel = std::make_unique<DeviceChannel>(app.config.base_url, app.config.allow_loopback_http,
            app.config.credential_file, app.model_hash, app.outbox, [&] { sync->request_gallery_refresh(true); worker->wake(); });
        channel->gallery_installed(app.coordinator.gallery_version(), app.coordinator.gallery_size());
    }
    runtime.start();
    if (channel) channel->runtime_state("running");
    output << "Capture started. Empty gallery waits for import/sync. Ctrl+C to stop.\n";
    auto status_due = std::chrono::steady_clock::now();
    auto persist_due = status_due;
    bool persistence_blocked = false;
    bool gallery_reload = false;
    auto gallery_retry_due = status_due;
    while (!stopping)
    {
        if (app.coordinator.pending_count())
        {
            if (std::chrono::steady_clock::now() < persist_due)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            try
            {
                while (auto record = app.coordinator.commit_next())
                {
                    output << "COMMITTED event_id=" << record->id << '\n' << std::flush;
                    if (worker)
                        worker->wake();
                }
                if (persistence_blocked)
                    output << "PERSISTENCE_RECOVERED; biometric processing resumed.\n" << std::flush;
                persistence_blocked = false;
                if (channel) channel->runtime_state(gallery_reload ? "gallery_error" : "running");
            }
            catch (StorageError const &error)
            {
                persistence_blocked = true;
                if (channel) channel->runtime_state("persistence_blocked");
                output << "PERSISTENCE_BLOCKED sqlite_code=" << error.code()
                       << " uncommitted=" << app.coordinator.pending_count()
                       << "; retry in 1s; no new biometric decisions.\n"
                       << std::flush;
                persist_due = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            }
            continue;
        }
        if (sync && sync->take_gallery_dirty()) gallery_reload = true;
        if (gallery_reload && std::chrono::steady_clock::now() >= gallery_retry_due)
        {
            try
            {
                app.coordinator.refresh_gallery(install);
                gallery_reload = false;
                if (channel)
                {
                    channel->gallery_installed(app.coordinator.gallery_version(), app.coordinator.gallery_size());
                    channel->runtime_state("running");
                }
                output << "Gallery applied; templates=" << app.coordinator.gallery_size() << '\n';
            }
            catch (...)
            {
                // GalleryMatcher installs atomically; preserve the last installed version and retry.
                if (channel) channel->runtime_state("gallery_error");
                gallery_retry_due = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                output << "GALLERY_INSTALL_FAILED; retaining installed gallery; retry in 5s.\n";
            }
        }
        auto frame = runtime.wait_for_frame(std::chrono::milliseconds(1000));
        if (frame)
        {
            if (channel) channel->frame_processed();
            app.coordinator.stage(frame->attendance_events);
        }
        auto now = std::chrono::steady_clock::now();
        if (now >= status_due)
        {
            output << "sync=" << (worker ? worker->status() : "offline")
                   << " channel=" << (channel ? channel->status() : "offline") << '\n';
            status_due = now + std::chrono::seconds(10);
        }
    }
    runtime.stop();
    channel.reset();
    worker.reset();
    if (app.coordinator.pending_count())
        throw std::runtime_error("PERSISTENCE_INCOMPLETE: stopping with uncommitted attendance in memory; "
                                 "these events are NOT recorded and require recapture after recovery");
}
} // namespace edge_app
