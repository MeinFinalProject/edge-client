#pragma once
#include "storage/outbox_repository.hpp"
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace edge_app
{
// Independent, optional operational channel. Never delivers attendance or installs a gallery.
class DeviceChannel
{
  public:
    DeviceChannel(std::string base_url, bool allow_loopback_http, std::filesystem::path credential,
                  std::string model_hash, OutboxRepository &outbox, std::function<void()> reconcile);
    ~DeviceChannel();
    void runtime_state(std::string state);
    void frame_processed();
    void gallery_installed(std::string version, std::size_t templates);
    std::string status() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace edge_app
