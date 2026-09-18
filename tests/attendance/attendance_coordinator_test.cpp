#include "app/attendance_coordinator.hpp"
#include "support/edge_fixtures.hpp"
#include <thread>

using namespace edge_test;
int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        TestDatabase database(root / "coordinator.db");
        AttendanceCoordinator coordinator(database.outbox, database.gallery, "device-test", "test-hash");
        vision_runtime::pipeline::AttendanceEvent event;
        event.occurred_at_utc_ms = utc_ms();
        event.identity_id = "S01";
        event.track_id = 9007199254740993ULL;
        event.camera_frame_id = 42;
        event.similarity = .81F;
        event.similarity_margin = .1F;
        event.pad_median_p_real = .99F;
        rejects([&] { (void)coordinator.record(event); });
        check(database.outbox.pending_count() == 0, "Event before gallery installation was persisted");

        const auto owner = std::this_thread::get_id();
        database.gallery.replace(gallery());
        coordinator.refresh_gallery([&](auto templates) {
            check(std::this_thread::get_id() == owner, "Gallery installed on another thread");
            check(templates.size() == 1 && templates[0].identity_id == "S01" && templates[0].embedding[0] == 1,
                  "Domain-to-runtime gallery conversion changed the template");
        });
        check(coordinator.gallery_version() == "v1" && coordinator.gallery_size() == 1,
              "Gallery installation not published");
        auto saved = coordinator.record(event);
        auto persisted = database.outbox.due(utc_ms()).at(0);
        check(saved.id == persisted.id && saved.payload == persisted.payload &&
                  saved.occurred_ms == event.occurred_at_utc_ms,
              "Coordinator returned before saving original event");
        auto json = JsonObject::Parse(winrt::to_hstring(persisted.payload));
        check(json.GetNamedString(L"gallery_version") == L"v1" && json.GetNamedString(L"device_id") == L"device-test" &&
                  json.GetNamedString(L"track_id") == L"9007199254740993" &&
                  json.GetNamedString(L"camera_frame_id") == L"42",
              "Application envelope lost source identity or precision");

        auto next = gallery();
        next.version = "v2";
        next.etag = "\"v2\"";
        database.gallery.replace(next);
        rejects([&] {
            coordinator.refresh_gallery(
                [](auto) { throw std::runtime_error("Simulated runtime installation failure"); });
        });
        check(coordinator.gallery_version() == "v1", "Failed installation published new version");
        auto still_old = coordinator.record(event);
        check(JsonObject::Parse(winrt::to_hstring(still_old.payload)).GetNamedString(L"gallery_version") == L"v1",
              "Event used persisted version before runtime installed it");
        coordinator.refresh_gallery([](auto) {});
        check(coordinator.gallery_version() == "v2", "Successful replacement not published");

        // Database remains compatible when reopened; no migration is needed for this refactor.
        {
            Database reopened(root / "coordinator.db");
            OutboxRepository outbox(reopened);
            GalleryRepository templates(reopened);
            check(outbox.pending_count() == 2 && templates.load().version == "v2",
                  "Schema or payload changed during refactor");
        }
        std::cout << "Coordinator owner thread, gallery install failure/version and commit boundary: PASS\n";
        return 0;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "Windows error " << std::hex << e.code().value << '\n';
        return 1;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
