#include "storage/outbox_repository.hpp"
#include "storage/database_internal.hpp"
#include <algorithm>
namespace edge_app
{
using namespace detail;
namespace
{
void dead(sqlite3 *d, std::string const &id, std::string const &reason)
{
    Statement s(d, "INSERT INTO dead_letter(event_id,payload,reason) SELECT event_id,payload,? FROM attendance_outbox "
                   "WHERE event_id=?");
    s.text(1, reason);
    s.text(2, id);
    s.step();
    Statement remove(d, "DELETE FROM attendance_outbox WHERE event_id=?");
    remove.text(1, id);
    remove.step();
}
} // namespace
void OutboxRepository::enqueue(AttendanceRecord const &r)
{
    std::lock_guard lock(database_.impl_->mutex);
    Transaction t(database_.impl_->db);
    Statement s(database_.impl_->db, "INSERT INTO attendance_outbox(event_id,payload,occurred_ms) VALUES(?,?,?)");
    s.text(1, r.id);
    s.text(2, r.payload);
    s.integer(3, r.occurred_ms);
    s.step();
    t.commit();
}
std::vector<AttendanceRecord> OutboxRepository::due(std::int64_t now, std::size_t limit)
{
    std::lock_guard lock(database_.impl_->mutex);
    Statement s(database_.impl_->db,
                "SELECT event_id,payload,occurred_ms,attempts,next_ms FROM attendance_outbox WHERE "
                "next_ms<=?1 AND ?1 >= (SELECT retry_not_before_ms FROM delivery_state WHERE singleton=1) "
                "ORDER BY occurred_ms,event_id LIMIT ?2");
    s.integer(1, now);
    s.integer(2, static_cast<std::int64_t>(std::min<std::size_t>(limit, 100)));
    std::vector<AttendanceRecord> rows;
    while (s.step())
        rows.push_back({s.string(0), s.string(1), sqlite3_column_int64(s.s, 2), sqlite3_column_int64(s.s, 4),
                        sqlite3_column_int(s.s, 3)});
    return rows;
}
void OutboxRepository::retry(std::vector<AttendanceRecord> const &rows, std::int64_t next, std::string const &reason)
{
    if (rows.empty())
        return;
    std::lock_guard lock(database_.impl_->mutex);
    Transaction t(database_.impl_->db);
    // Endpoint backoff also applies to new events and survives process restart.
    Statement pause(database_.impl_->db,
                    "UPDATE delivery_state SET retry_not_before_ms=MAX(retry_not_before_ms,?) WHERE singleton=1");
    pause.integer(1, next);
    pause.step();
    Statement s(database_.impl_->db,
                "UPDATE attendance_outbox SET attempts=MIN(attempts+1,30),next_ms=?,last_error=? WHERE event_id=?");
    for (auto const &r : rows)
    {
        s.integer(1, next);
        s.text(2, reason);
        s.text(3, r.id);
        s.step();
        s.reset();
    }
    t.commit();
}
void OutboxRepository::acknowledge(std::vector<DeliveryReceipt> const &receipts)
{
    for (auto const &r : receipts)
        if (r.disposition != "accepted" && r.disposition != "duplicate" && r.disposition != "rejected")
            throw std::invalid_argument("Invalid receipt disposition");
    std::lock_guard lock(database_.impl_->mutex);
    Transaction t(database_.impl_->db);
    for (auto const &r : receipts)
    {
        if (r.disposition == "rejected")
            dead(database_.impl_->db, r.event_id, "business_rejected");
        else
        {
            Statement s(database_.impl_->db, "DELETE FROM attendance_outbox WHERE event_id=?");
            s.text(1, r.event_id);
            s.step();
        }
    }
    t.commit();
}
void OutboxRepository::dead_letter(std::vector<AttendanceRecord> const &rows, std::string const &reason)
{
    std::lock_guard lock(database_.impl_->mutex);
    Transaction t(database_.impl_->db);
    for (auto const &r : rows)
        dead(database_.impl_->db, r.id, reason);
    t.commit();
}
std::size_t OutboxRepository::pending_count()
{
    std::lock_guard lock(database_.impl_->mutex);
    Statement s(database_.impl_->db, "SELECT COUNT(*) FROM attendance_outbox");
    s.step();
    return static_cast<std::size_t>(sqlite3_column_int64(s.s, 0));
}
std::size_t OutboxRepository::dead_count()
{
    std::lock_guard lock(database_.impl_->mutex);
    Statement s(database_.impl_->db, "SELECT COUNT(*) FROM dead_letter");
    s.step();
    return static_cast<std::size_t>(sqlite3_column_int64(s.s, 0));
}

} // namespace edge_app
