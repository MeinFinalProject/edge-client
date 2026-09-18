#pragma once
#include "sync/sync_service.hpp"
#include <filesystem>
#include <memory>
namespace edge_app
{
// Callback only wakes a worker. The HTTP result determines server reachability.
class SyncWorker
{
  public:
    SyncWorker(SyncService &sync, std::filesystem::path credential_file);
    ~SyncWorker();
    void wake();
    std::string status() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edge_app
