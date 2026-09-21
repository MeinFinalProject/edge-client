#include "support/edge_fixtures.hpp"
#include "sync/device_channel.hpp"
#include <chrono>
#include <thread>
using namespace edge_test;
int main()
{
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        std::filesystem::create_directories(root);
        TestDatabase db(root / "channel.db");
        auto row = record();
        db.outbox.enqueue(row);
        protect_token(root / "token.dpapi", std::string(64, 'a'));
        auto before = std::chrono::steady_clock::now();
        {
            DeviceChannel channel("https://127.0.0.1:1/api/v1", false, root / "token.dpapi",
                                  std::string(64, 'a'), db.outbox, [] {});
            channel.runtime_state("running");
            channel.frame_processed();
            channel.gallery_installed("installed-v1", 1);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            check(db.outbox.pending_count() == 1, "Disconnected telemetry modified durable outbox");
        }
        check(std::chrono::steady_clock::now() - before < std::chrono::seconds(3), "Channel shutdown blocked");
        check(db.outbox.due(utc_ms()).front().payload == row.payload, "Channel changed attendance payload");
        // Notification requests shorten healthy polling, but preserve endpoint backoff.
        struct GalleryReply : Mock
        {
            int gallery_code = 304;
            HttpResponse request(std::string const &, std::string const &, std::string const &, std::string const &) override
            { ++get_calls; return {gallery_code, "", "", "120"}; }
        } http;
        db.gallery.replace(gallery());
        // Separate empty outbox avoids a POST in this gallery-specific test.
        TestDatabase empty(root / "empty.db");
        empty.gallery.replace(gallery());
        SyncService polling(empty.outbox, empty.gallery, http, "device-test", "test-hash");
        polling.set_token("test-token");
        auto now = utc_ms();
        polling.step(now);
        polling.step(now + 10);
        check(http.get_calls == 1, "Normal gallery poll was not delayed");
        polling.request_gallery_refresh();
        polling.step(now + 20);
        check(http.get_calls == 2, "Notification did not wake gallery poll");
        http.gallery_code = 429;
        polling.request_gallery_refresh();
        polling.step(now + 30);
        polling.request_gallery_refresh();
        polling.step(now + 1000);
        check(http.get_calls == 3, "Notification bypassed Retry-After");
        http.gallery_code = 401;
        polling.step(now + 130000);
        check(polling.auth_blocked(), "401 must block HTTP sync");
        http.gallery_code = 304;
        polling.request_gallery_refresh(true);
        polling.step(now + 130001);
        check(!polling.auth_blocked(), "Authenticated reconnection did not recover a re-enabled device");
        std::cout << "Device channel cancellation/offline isolation/gallery backoff PASS\n";
        return 0;
    }
    catch (std::exception const &e) { std::cerr << e.what() << '\n'; return 1; }
    catch (...) { return 1; }
}
