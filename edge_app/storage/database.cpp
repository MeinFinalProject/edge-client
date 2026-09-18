#include "storage/database_internal.hpp"
#include "storage/schema.hpp"
namespace edge_app
{
using namespace detail;
Database::Database(std::filesystem::path const &file) : impl_(std::make_unique<Impl>())
{
    if (file.has_parent_path())
        std::filesystem::create_directories(file.parent_path());
    auto u = file.u8string();
    check(sqlite3_open_v2(reinterpret_cast<char const *>(u.c_str()), &impl_->db,
                          SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr));
    auto d = impl_->db;
    check(sqlite3_busy_timeout(d, 1500));
    // Conservative journal mode for OS-managed SQLite versions, including older
    // versions affected by upstream WAL-reset races. One process owns this DB.
    sql(d, "PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;");
    int v;
    {
        Statement version(d, "PRAGMA user_version");
        version.step();
        v = sqlite3_column_int(version.s, 0);
    }
    if (v < 0 || v > schema_version)
        throw std::runtime_error("Unsupported database schema version");
    if (v < schema_version)
    {
        Transaction t(d);
        if (v == 0)
            sql(d, create_schema);
        sql(d, migrate_v1_to_v2);
        t.commit();
    }
}
Database::~Database() = default;
std::string Database::sqlite_version() const
{
    return sqlite3_libversion();
}
} // namespace edge_app
