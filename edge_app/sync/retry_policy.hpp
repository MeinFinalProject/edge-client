#pragma once
#include <chrono>
#include <cstdint>
#include <string>
namespace edge_app
{
unsigned retry_jitter();
std::int64_t retry_at(std::int64_t now, int attempts, std::string const &retry_after, unsigned jitter_ms);
// Pair an injected UTC start with monotonic elapsed time for post-request backoff.
class RetryClock
{
    std::int64_t start_ms_;
    std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();

  public:
    explicit RetryClock(std::int64_t start_ms) : start_ms_(start_ms)
    {
    }
    std::int64_t now() const
    {
        return start_ms_ +
               std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin_).count();
    }
};
} // namespace edge_app
