#include "app/attendance_coordinator.hpp"
#include "support/database_access.hpp"
#include "support/edge_fixtures.hpp"
#include <algorithm>
using namespace edge_test;
namespace
{
template <class F> void storage_failure(F f, int expected)
{
    try
    {
        f();
    }
    catch (StorageError const &e)
    {
        check((e.code() & 255) == expected, "Unexpected SQLite failure code");
        return;
    }
    throw std::runtime_error("Persistence incorrectly reported success");
}
} // namespace
int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        auto file = root / "recovery.db";
        std::size_t committed = 0;
        {
            TestDatabase db(file);
            db.gallery.replace(gallery());
            AttendanceCoordinator c(db.outbox, db.gallery, "device-test", "test-hash");
            c.refresh_gallery([](auto) {});
            vision_runtime::pipeline::AttendanceEvent event;
            event.occurred_at_utc_ms = utc_ms();
            event.identity_id = "S01";
            auto previous = c.record(event);
            c.stage({event, event});
            auto id = c.pending_event_id();
            DatabaseTestAccess::run(db.database, [](sqlite3 *d) {
                sqlite3_set_authorizer(
                    d,
                    [](void *, int action, char const *table, char const *, char const *, char const *) {
                        return action == SQLITE_INSERT && table && std::string(table) == "attendance_outbox"
                                   ? SQLITE_DENY
                                   : SQLITE_OK;
                    },
                    nullptr);
            });
            storage_failure([&] { (void)c.commit_next(); }, SQLITE_AUTH);
            check(c.pending_count() == 2 && c.pending_event_id() == id && db.outbox.pending_count() == 1,
                  "Write error lost staged/committed events or changed UUID");
            rejects([&] { c.stage({event}); });
            rejects([&] { c.refresh_gallery([](auto) {}); });
            DatabaseTestAccess::run(db.database, [](sqlite3 *d) {
                sqlite3_set_authorizer(d, nullptr, nullptr);
                sqlite3_commit_hook(d, [](void *) { return 1; }, nullptr);
            });
            storage_failure([&] { (void)c.commit_next(); }, SQLITE_CONSTRAINT);
            check(c.pending_count() == 2 && c.pending_event_id() == id && db.outbox.pending_count() == 1,
                  "Failed COMMIT was reported as recorded or damaged old data");
            DatabaseTestAccess::run(db.database, [](sqlite3 *d) {
                check(sqlite3_get_autocommit(d) != 0, "Failed COMMIT left a transaction open");
                sqlite3_commit_hook(d, nullptr, nullptr);
            });
            auto recovered = c.commit_next();
            check(recovered && recovered->id == id, "Persistence retry regenerated the UUID");
            DatabaseTestAccess::run(db.database,
                                    [](sqlite3 *d) { sqlite3_commit_hook(d, [](void *) { return 1; }, nullptr); });
            storage_failure([&] { (void)c.commit_next(); }, SQLITE_CONSTRAINT);
            check(c.pending_count() == 1 && db.outbox.pending_count() == 2,
                  "Failure later in a frame batch lost or retracted its earlier COMMIT");
            DatabaseTestAccess::run(db.database, [](sqlite3 *d) { sqlite3_commit_hook(d, nullptr, nullptr); });
            (void)c.commit_next();
            check(c.pending_count() == 0 && db.outbox.pending_count() == 3, "Batch recovery lost events");
            db.outbox.ensure_enqueued(*recovered);
            check(db.outbox.pending_count() == 3, "Uncertain COMMIT retry duplicated local record");
            auto conflict = *recovered;
            conflict.payload += " ";
            rejects([&] { db.outbox.ensure_enqueued(conflict); });

            // Real SQLITE_FULL from a per-connection page cap, without filling the host disk.
            DatabaseTestAccess::run(db.database, [](sqlite3 *d) {
                edge_app::detail::Statement size(d, "PRAGMA page_count");
                size.step();
                auto pages = sqlite3_column_int(size.s, 0);
                edge_app::detail::sql(d, ("PRAGMA max_page_count=" + std::to_string(pages)).c_str());
            });
            bool full = false;
            event.identity_id = std::string(240, 'T');
            for (int i = 0; i < 1000 && !full; ++i)
            {
                c.stage({event});
                try
                {
                    (void)c.commit_next();
                }
                catch (StorageError const &e)
                {
                    check((e.code() & 255) == SQLITE_FULL, "Expected page-limit SQLITE_FULL");
                    full = true;
                }
            }
            check(full && c.pending_count() == 1, "Disk-full simulation did not retain the uncommitted event");
            id = c.pending_event_id();
            committed = db.outbox.pending_count();
            storage_failure([&] { (void)c.commit_next(); }, SQLITE_FULL);
            check(db.outbox.pending_count() == committed, "Repeated FULL failure changed committed rows");
            DatabaseTestAccess::run(db.database,
                                    [](sqlite3 *d) { edge_app::detail::sql(d, "PRAGMA max_page_count=1000000"); });
            check(c.commit_next()->id == id && db.outbox.pending_count() == committed + 1,
                  "Capacity recovery lost or changed the pending event");
            ++committed;
            auto rows = db.outbox.due(utc_ms(), 100);
            check(std::any_of(rows.begin(), rows.end(),
                              [&](auto const &r) { return r.id == previous.id && r.payload == previous.payload; }),
                  "Persistence errors changed a previously committed event");
        }
        TestDatabase reopened(file);
        check(reopened.outbox.pending_count() == committed, "Restart lost recovered committed events");
        auto retained = record();
        reopened.outbox.enqueue(retained);
        reopened.outbox.dead_letter({retained}, "test_rejected");
        reopened.outbox.ensure_enqueued(retained);
        check(reopened.outbox.pending_count() == committed && reopened.outbox.dead_count() == 1,
              "Uncertain COMMIT retry resurrected a permanently rejected record");
        auto conflict = retained;
        conflict.payload += " ";
        rejects([&] { reopened.outbox.ensure_enqueued(conflict); });
        std::cout << "Write denial, failed COMMIT, SQLITE_FULL, stable retries and reopen: PASS\n";
        return 0;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "Windows error " << e.code().value << '\n';
        return 1;
    }
}
