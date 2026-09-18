#pragma once
#include "storage/outbox_repository.hpp"
#include "sync/http_client.hpp"
namespace edge_app
{
class AttendanceSync
{
  public:
    AttendanceSync(OutboxRepository &outbox, Transport &http, std::string device);
    SyncResult step(std::int64_t now);

  private:
    OutboxRepository &outbox_;
    Transport &http_;
    std::string device_;
};
} // namespace edge_app
