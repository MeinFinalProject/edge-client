#pragma once
#include <stdexcept>
#include <string>
namespace edge_app
{
// Error codes only: never include SQL, payloads, identities or file contents.
class StorageError : public std::runtime_error
{
  public:
    explicit StorageError(int code)
        : std::runtime_error("SQLite operation failed: " + std::to_string(code)), code_(code)
    {
    }
    int code() const noexcept
    {
        return code_;
    }

  private:
    int code_;
};
} // namespace edge_app
