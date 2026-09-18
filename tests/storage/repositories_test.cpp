#include "support/edge_fixtures.hpp"
#include <algorithm>
using namespace edge_test;
int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        std::filesystem::create_directories(root);
        TestDatabase database(root / "repositories.db");
        check(database.gallery.load_metadata().version.empty(), "Missing gallery must have empty metadata");
        auto event = record();
        database.outbox.enqueue(event);
        database.gallery.replace(gallery());
        check(database.outbox.pending_count() == 1 && database.gallery.load().templates.size() == 1,
              "Repositories did not share the database");
        auto large = gallery();
        large.version = "large";
        large.etag = "\"large\"";
        large.template_ids.clear();
        large.templates.clear();
        for (int i = 0; i < 10000; ++i)
        {
            large.template_ids.push_back("template-" + std::to_string(i));
            GalleryTemplate item;
            item.identity_id = "identity-" + std::to_string(i);
            item.embedding[i % embedding_dimension] = 2;
            large.templates.push_back(item);
        }
        database.gallery.replace(large);
        auto metadata = database.gallery.load_metadata();
        check(metadata.version == large.version && metadata.etag == large.etag && metadata.model_id == large.model_id &&
                  metadata.model_sha256 == large.model_sha256,
              "Gallery metadata did not match the committed snapshot");
        auto restored = database.gallery.load();
        check(restored.templates.size() == 10000, "Large snapshot lost templates");
        for (std::size_t i = 0; i < restored.templates.size(); ++i)
        {
            auto number = std::stoi(restored.template_ids[i].substr(9));
            check(restored.templates[i].identity_id == "identity-" + std::to_string(number) &&
                      restored.templates[i].embedding[number % embedding_dimension] == 1,
                  "Reused insert retained a previous template binding");
        }
        large.template_ids.back() = large.template_ids.front();
        large.version = "invalid";
        rejects([&] { database.gallery.replace(large); });
        check(database.gallery.load_metadata().version == "large" && database.gallery.load().templates.size() == 10000,
              "Failed large snapshot replacement did not roll back atomically");

        auto legacy = root / "schema-v1.db";
        const auto now = utc_ms();
        auto retained = record();
        {
            TestDatabase initial(legacy);
            initial.outbox.enqueue(retained);
            initial.outbox.retry({retained}, now + 120000, "http_429");
            initial.gallery.replace(gallery());
        }
        // Recreate the previously released schema: v1 had no delivery_state table.
        sqlite3 *raw = nullptr;
        auto path = legacy.u8string();
        check(sqlite3_open(reinterpret_cast<char const *>(path.c_str()), &raw) == SQLITE_OK, "Open v1 fixture failed");
        auto rc = sqlite3_exec(raw, "DROP TABLE delivery_state; PRAGMA user_version=1;", nullptr, nullptr, nullptr);
        sqlite3_close(raw);
        check(rc == SQLITE_OK, "Create v1 fixture failed");
        {
            TestDatabase migrated(legacy);
            check(migrated.outbox.pending_count() == 1 && migrated.gallery.load().version == "v1",
                  "Schema migration lost existing data");
            migrated.outbox.enqueue(record());
            check(migrated.outbox.due(now + 1000).empty(), "Migration lost the outstanding retry deadline");
            auto due = migrated.outbox.due(now + 120001);
            check(due.size() == 2, "Migrated records did not become due");
            check(std::any_of(due.begin(), due.end(),
                              [&](auto const &row) {
                                  return row.id == retained.id && row.payload == retained.payload && row.attempts == 1;
                              }),
                  "Migration changed the original event payload/UUID/retry count");
        }
        {
            TestDatabase reopened(legacy);
            check(reopened.outbox.due(now + 1000).empty(), "Endpoint retry deadline did not survive restart");
        }
        std::cout << "Shared database repositories PASS\n";
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
