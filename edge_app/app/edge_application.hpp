#pragma once
#include "app/application_config.hpp"
#include <atomic>
#include <iosfwd>
#include <memory>

namespace edge_app
{
// Composition root: owns database/repositories and creates runtime, transport,
// sync service and worker for the lifetime of run(). CLI/console controls stay outside.
class EdgeApplication
{
  public:
    explicit EdgeApplication(ApplicationConfig config);
    ~EdgeApplication();
    void validate(std::ostream &output);
    void import_gallery(std::filesystem::path const &document);
    void run(std::atomic_bool const &stopping, std::ostream &output);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace edge_app
