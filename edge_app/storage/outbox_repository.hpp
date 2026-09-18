#pragma once
#include "domain/attendance_record.hpp"
#include "storage/database.hpp"
#include <vector>
namespace edge_app
{
class OutboxRepository
{
  public:
    explicit OutboxRepository(Database &database) : database_(database)
    {
    }
    void enqueue(AttendanceRecord const &record);
    std::vector<AttendanceRecord> due(std::int64_t now_ms, std::size_t limit = 32);
    void retry(std::vector<AttendanceRecord> const &records, std::int64_t next_ms, std::string const &reason);
    void acknowledge(std::vector<DeliveryReceipt> const &receipts);
    void dead_letter(std::vector<AttendanceRecord> const &records, std::string const &reason);
    std::size_t pending_count();
    std::size_t dead_count();

  private:
    Database &database_;
};
} // namespace edge_app
