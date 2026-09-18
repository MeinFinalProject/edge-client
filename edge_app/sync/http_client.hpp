#pragma once
#include <chrono>
#include <memory>
#include <string>
namespace edge_app
{
struct HttpResponse
{
    int status = 0;
    std::string body, etag, retry_after;
};
class Transport
{
  public:
    virtual ~Transport() = default;
    virtual HttpResponse request(std::string const &method, std::string const &path, std::string const &body,
                                 std::string const &etag) = 0;
    virtual void set_token(std::string const &token) = 0;
};
std::unique_ptr<Transport> http_transport(std::string const &base_url, bool allow_loopback_http = false,
                                          std::chrono::milliseconds timeout = std::chrono::seconds(10));

enum class SyncResult
{
    Continue,
    AuthenticationBlocked
};

} // namespace edge_app
