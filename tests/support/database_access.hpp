#pragma once
#include "storage/database_internal.hpp"
namespace edge_app
{
// Test-only friend: fault injection uses the real SQLite connection/transaction.
// No injection switches or raw database handles are exposed to the application.
struct DatabaseTestAccess
{
    template <class F> static auto run(Database &database, F f)
    {
        std::lock_guard lock(database.impl_->mutex);
        return f(database.impl_->db);
    }
};
} // namespace edge_app
