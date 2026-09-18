#include "support/edge_fixtures.hpp"
using namespace edge_test;
namespace
{
void crash_child(std::filesystem::path const &db)
{
    TestDatabase s(db);
    for (int i = 0; i < 100; ++i)
        s.outbox.enqueue(record());
    TerminateProcess(GetCurrentProcess(), 37);
}
void run_crash(std::filesystem::path const &db, bool before_commit = false)
{
    wchar_t exe[32768];
    check(GetModuleFileNameW(nullptr, exe, 32768) > 0, "Cannot find test executable");
    std::wstring command = L"\"" + std::wstring(exe) +
                           (before_commit ? L"\" --rollback-and-crash \"" : L"\" --commit-and-crash \"") +
                           db.wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    check(CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                         &process),
          "Cannot launch crash subprocess");
    auto wait = WaitForSingleObject(process.hProcess, 20000);
    if (wait != WAIT_OBJECT_0)
        TerminateProcess(process.hProcess, 99);
    DWORD code = 0;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    check(wait == WAIT_OBJECT_0 && code == 37, "Crash subprocess failed before commit");
}
} // namespace
int main(int argc, char **argv)
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        if (argc == 3 && std::string(argv[1]) == "--commit-and-crash")
        {
            crash_child(std::filesystem::u8path(argv[2]));
            return 2;
        }
        if (argc == 3 && std::string(argv[1]) == "--rollback-and-crash")
        {
            sqlite3 *db = nullptr;
            check(sqlite3_open(argv[2], &db) == SQLITE_OK, "Crash test open failed");
            check(sqlite3_exec(
                      db,
                      "BEGIN IMMEDIATE; DELETE FROM gallery_template; UPDATE gallery_state SET version='uncommitted';",
                      nullptr, nullptr, nullptr) == SQLITE_OK,
                  "Crash test transaction failed");
            TerminateProcess(GetCurrentProcess(), 37);
            return 2;
        }
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        std::filesystem::create_directories(root);
        auto db = root / "events.db";
        std::unordered_set<std::string> ids;
        {
            TestDatabase s(db);
            std::cout << "OS SQLite=" << s.database.sqlite_version() << '\n';
            for (int i = 0; i < 100; ++i)
            {
                auto r = record();
                ids.insert(r.id);
                s.outbox.enqueue(r);
            }
            check(ids.size() == 100 && s.outbox.pending_count() == 100, "100 offline commits failed");
            auto r = s.outbox.due(utc_ms(), 100).front();
            rejects([&] { s.outbox.enqueue(r); });
            check(s.outbox.pending_count() == 100, "Duplicate insert damaged database");
            auto json = JsonObject::Parse(winrt::to_hstring(r.payload));
            check(json.GetNamedString(L"track_id") == L"9007199254740993", "JSON lost uint64 precision");
            check(r.id[14] == '7' && std::string("89ab").find(r.id[19]) != std::string::npos,
                  "Invalid UUID version/variant");
            auto g = gallery();
            s.gallery.replace(g);
            g.version = "bad";
            g.template_ids.push_back("t1");
            g.templates.push_back(g.templates[0]);
            rejects([&] { s.gallery.replace(g); });
            check(s.gallery.load().version == "v1", "GallerySnapshot transaction failed rollback");
        }
        run_crash(db, true);
        {
            TestDatabase s(db);
            check(s.outbox.pending_count() == 100, "Restart lost committed events");
            for (auto &r : s.outbox.due(utc_ms(), 100))
                check(ids.contains(r.id), "Restart regenerated UUID");
            check(s.gallery.load().version == "v1" && s.gallery.load().templates.size() == 1,
                  "Crash before commit lost old gallery");
            Mock drain;
            SyncService sync(s.outbox, s.gallery, drain, "device-test", "test-hash");
            sync.set_token("local-test");
            for (int i = 0; i < 4; ++i)
                sync.step(utc_ms());
            check(s.outbox.pending_count() == 0 && drain.accepted.size() == 100,
                  "100 offline records did not drain exactly once at mock");
        }
        run_crash(root / "crash.db");
        {
            TestDatabase s(root / "crash.db");
            check(s.outbox.pending_count() == 100, "Forced process termination lost commits");
        }
        std::cout << "Offline/restart/crash recovery PASS\n";
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
