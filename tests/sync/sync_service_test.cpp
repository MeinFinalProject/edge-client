#include "support/database_access.hpp"
#include "support/edge_fixtures.hpp"
using namespace edge_test;
int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        std::filesystem::create_directories(root);
        TestDatabase s(root / "sync.db");
        s.gallery.replace(gallery());
        Mock http;
        SyncService sync(s.outbox, s.gallery, http, "device-test", "test-hash");
        sync.set_token("token-one");
        auto now = utc_ms();
        s.outbox.enqueue(record());
        http.lose_ack = true;
        sync.step(now);
        check(s.outbox.pending_count() == 1 && http.accepted.size() == 1, "Lost ACK must retain event");
        sync.step(now + 500);
        check(http.post_calls == 1, "Retry happened before deadline");
        sync.step(now + 10000);
        check(s.outbox.pending_count() == 0 && http.accepted.size() == 1,
              "Retry duplicate was not acknowledged safely");
        s.outbox.enqueue(record());
        s.outbox.enqueue(record());
        http.partial = true;
        sync.step(now + 20000);
        check(s.outbox.pending_count() == 1, "Partial ACK removed unacknowledged event");
        http.partial = false;
        sync.step(now + 30000);
        check(s.outbox.pending_count() == 0, "Remaining partial event not retried");
        s.outbox.enqueue(record());
        http.invalid = true;
        sync.step(now + 40000);
        check(s.outbox.pending_count() == 1, "Unknown ACK removed event");
        http.invalid = false;
        http.code = 401;
        sync.step(now + 50000);
        check(sync.auth_blocked() && s.outbox.pending_count() == 1, "401 did not preserve/block");
        int calls = http.post_calls;
        sync.step(now + 60000);
        check(http.post_calls == calls, "401 retried without credentials changing");
        sync.set_token("token-two");
        http.code = 429;
        http.retry_after = "120";
        sync.step(now + 70000);
        check(s.outbox.due(now + 180000).empty(), "Retry-After ignored");
        calls = http.post_calls;
        s.outbox.enqueue(record());
        sync.step(now + 80000);
        check(http.post_calls == calls && s.outbox.pending_count() == 2,
              "New event bypassed the endpoint Retry-After deadline");
        http.code = 400;
        sync.step(now + 200000);
        check(s.outbox.pending_count() == 0 && s.outbox.dead_count() == 2,
              "Schema rejection not retained in dead letter");
        http.gallery_body = empty_gallery("v2", "wrong");
        sync.step(now + 300000);
        check(s.gallery.load().version == "v1", "Incompatible gallery replaced valid snapshot");
        http.gallery_body = empty_gallery();
        sync.step(now + 400000);
        check(sync.take_gallery_dirty() && s.gallery.load().templates.empty() && s.gallery.load().version == "v2",
              "Empty gallery revocation failed");
        check(retry_at(now, 0, "120", 0) == now + 120000, "Numeric Retry-After incorrect");
        rejects([] { auto t = http_transport("http://example.com", true); });
        TestDatabase mixed(root / "mixed-attempts.db");
        auto old = record();
        mixed.outbox.enqueue(old);
        for (int i = 0; i < 8; ++i)
            mixed.outbox.retry({old}, 0, "previous_failure");
        auto fresh = record();
        fresh.occurred_ms = old.occurred_ms - 1; // First in the batch, with fewer attempts.
        mixed.outbox.enqueue(fresh);
        Mock unavailable;
        unavailable.code = 503;
        AttendanceSync retry_mixed(mixed.outbox, unavailable, "device-test");
        retry_mixed.step(now);
        check(mixed.outbox.due(now + 200000).empty(), "Fresh row reset an older row's exponential backoff");
        check(mixed.outbox.due(now + 400000).size() == 2, "Mixed retry batch never became due");
        for (int code : {403, 408, 429, 500, 502, 503, 504})
        {
            TestDatabase errors(root / ("http-" + std::to_string(code) + ".db"));
            auto event = record();
            errors.outbox.enqueue(event);
            Mock failing;
            failing.code = code;
            SyncService service(errors.outbox, errors.gallery, failing, "device-test", "test-hash");
            service.set_token("test-token");
            service.step(now);
            check(errors.outbox.pending_count() == 1 && errors.outbox.dead_count() == 0,
                  "Retryable/auth HTTP error discarded an event");
            check(service.auth_blocked() == (code == 403), "Incorrect auth-blocked state");
        }
        struct Reply : Mock
        {
            std::string body;
            HttpResponse request(std::string const &method, std::string const &path, std::string const &payload,
                                 std::string const &etag) override
            {
                if (method == "POST")
                {
                    ++post_calls;
                    return {200, body};
                }
                return Mock::request(method, path, payload, etag);
            }
        } malformed;
        TestDatabase receipts(root / "malformed-receipts.db");
        auto first = record(), second = record();
        receipts.outbox.enqueue(first);
        receipts.outbox.enqueue(second);
        SyncService receipt_sync(receipts.outbox, receipts.gallery, malformed, "device-test", "test-hash");
        receipt_sync.set_token("test-token");
        auto ack = "{\"event_id\":\"" + first.id + "\",\"status\":\"accepted\"}";
        std::vector<std::string> invalid_bodies{"", "{", "{}",
                                                "{\"schema_version\":1,\"results\":[" + ack + "," + ack + "]}"};
        for (std::size_t i = 0; i < invalid_bodies.size(); ++i)
        {
            malformed.body = invalid_bodies[i];
            receipt_sync.step(now + static_cast<std::int64_t>(i) * 400000);
            check(receipts.outbox.pending_count() == 2 && receipts.outbox.dead_count() == 0,
                  "Malformed/duplicate ACK silently removed a pending event");
        }
        malformed.body = "{\"schema_version\":1,\"results\":[" + ack + "]}";
        receipt_sync.step(now + 2000000);
        check(receipts.outbox.pending_count() == 1, "Partial receipt did not retain the missing event");
        auto remaining = receipts.outbox.due(now + 2400000);
        check(remaining.size() == 1 && remaining[0].id == second.id && remaining[0].payload == second.payload,
              "Retry changed the unacknowledged payload");

        TestDatabase local_failure(root / "ack-commit.db");
        local_failure.outbox.enqueue(first);
        Mock accepted;
        SyncService accepted_sync(local_failure.outbox, local_failure.gallery, accepted, "device-test", "test-hash");
        accepted_sync.set_token("test-token");
        DatabaseTestAccess::run(local_failure.database,
                                [](sqlite3 *d) { sqlite3_commit_hook(d, [](void *) { return 1; }, nullptr); });
        rejects([&] { accepted_sync.step(now); });
        check(accepted.accepted.size() == 1 && local_failure.outbox.pending_count() == 1,
              "Failed local ACK COMMIT lost the original event");
        DatabaseTestAccess::run(local_failure.database, [](sqlite3 *d) { sqlite3_commit_hook(d, nullptr, nullptr); });
        accepted_sync.step(now + 10000);
        check(accepted.accepted.size() == 1 && local_failure.outbox.pending_count() == 0,
              "Retry after local ACK failure did not acknowledge the duplicate safely");
        std::cout << "Sync service receipts/retry/gallery PASS\n";
        return 0;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "WinRT error " << std::hex << e.code().value << '\n';
        return 1;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
