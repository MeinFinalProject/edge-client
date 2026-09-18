#pragma once
#include <functional>
#include <memory>
namespace edge_app
{
// Optional network hints. Failure to subscribe does not disable timed retries.
// The callback may run on another thread and must not access the vision runtime.
class NetworkWatcher
{
  public:
    explicit NetworkWatcher(std::function<void()> wake);
    ~NetworkWatcher();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace edge_app
