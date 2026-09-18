#pragma once
// Internal SQLite RAII helpers; include only from storage .cpp files.
#include "storage/database.hpp"
#include <mutex>
#include <stdexcept>
#include <winsqlite/winsqlite3.h>
namespace edge_app
{
struct Database::Impl
{
    sqlite3 *db = nullptr;
    std::mutex mutex;
    ~Impl()
    {
        if (db)
            sqlite3_close(db);
    }
};
namespace detail
{
inline void check(int rc)
{
    if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW)
        throw std::runtime_error("SQLite operation failed: " + std::to_string(rc));
}
inline void sql(sqlite3 *d, char const *q)
{
    check(sqlite3_exec(d, q, nullptr, nullptr, nullptr));
}
struct Statement
{
    sqlite3_stmt *s = nullptr;
    Statement(sqlite3 *d, char const *q)
    {
        check(sqlite3_prepare_v2(d, q, -1, &s, nullptr));
    }
    Statement(Statement const &) = delete;
    Statement &operator=(Statement const &) = delete;
    ~Statement()
    {
        sqlite3_finalize(s);
    }
    void text(int i, std::string const &v)
    {
        check(sqlite3_bind_text(s, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT));
    }
    void integer(int i, std::int64_t v)
    {
        check(sqlite3_bind_int64(s, i, v));
    }
    void blob(int i, void const *v, int size)
    {
        check(sqlite3_bind_blob(s, i, v, size, SQLITE_TRANSIENT));
    }
    bool step()
    {
        auto r = sqlite3_step(s);
        check(r);
        return r == SQLITE_ROW;
    }
    void reset()
    {
        check(sqlite3_reset(s));
        check(sqlite3_clear_bindings(s));
    }
    std::string string(int i)
    {
        auto p = sqlite3_column_text(s, i);
        return p ? std::string(reinterpret_cast<char const *>(p), sqlite3_column_bytes(s, i)) : std::string{};
    }
};
struct Transaction
{
    sqlite3 *d;
    bool done = false;
    explicit Transaction(sqlite3 *p) : d(p)
    {
        sql(d, "BEGIN IMMEDIATE");
    }
    Transaction(Transaction const &) = delete;
    Transaction &operator=(Transaction const &) = delete;
    ~Transaction()
    {
        if (!done)
            sqlite3_exec(d, "ROLLBACK", nullptr, nullptr, nullptr);
    }
    void commit()
    {
        sql(d, "COMMIT");
        done = true;
    }
};
} // namespace detail
} // namespace edge_app
