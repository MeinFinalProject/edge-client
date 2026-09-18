#pragma once
#include <filesystem>
#include <memory>
#include <string>
namespace edge_app
{
class OutboxRepository;
class GalleryRepository;
// Owns one SQLite connection and its mutex. Repositories borrow it and cannot
// outlive it. Native handles are private to storage implementation files.
class Database
{
  public:
    explicit Database(std::filesystem::path const &file);
    ~Database();
    Database(Database const &) = delete;
    Database &operator=(Database const &) = delete;
    std::string sqlite_version() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class OutboxRepository;
    friend class GalleryRepository;
};
} // namespace edge_app
