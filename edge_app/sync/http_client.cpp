#include "sync/http_client.hpp"
#include "security/device_credential.hpp"
#include <stdexcept>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/Windows.Web.Http.Filters.h>
#include <winrt/Windows.Web.Http.Headers.h>
#include <winrt/Windows.Web.Http.h>

namespace edge_app
{
namespace
{
using namespace winrt::Windows::Web::Http;
using namespace winrt::Windows::Web::Http::Filters;
using namespace winrt::Windows::Storage::Streams;
using winrt::Windows::Foundation::Uri;
template <class T> auto finish(T const &op, std::chrono::steady_clock::time_point deadline)
{
    auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= remaining.zero() || op.wait_for(std::chrono::duration_cast<winrt::Windows::Foundation::TimeSpan>(
                                             remaining)) == winrt::Windows::Foundation::AsyncStatus::Started)
    {
        op.Cancel();
        throw std::runtime_error("HTTP deadline exceeded");
    }
    return op.GetResults();
}
class Http final : public Transport
{
    std::string base_, token_;
    std::chrono::milliseconds timeout_;
    HttpBaseProtocolFilter filter_;
    HttpClient client_{nullptr};

  public:
    Http(std::string base, bool loopback, std::chrono::milliseconds timeout) : base_(std::move(base)), timeout_(timeout)
    {
        Uri uri(winrt::to_hstring(base_));
        const auto scheme = winrt::to_string(uri.SchemeName()), host = winrt::to_string(uri.Host());
        if (scheme != "https" &&
            !(loopback && scheme == "http" && (host == "127.0.0.1" || host == "localhost" || host == "[::1]")))
            throw std::invalid_argument("HTTPS required; plain HTTP allowed only for explicit loopback tests");
        if (!uri.UserName().empty() || !uri.Password().empty() || !uri.Query().empty() || !uri.Fragment().empty() ||
            timeout_.count() <= 0)
            throw std::invalid_argument("Invalid HTTP base URL/deadline");
        while (!base_.empty() && base_.back() == '/')
            base_.pop_back();
        filter_.AllowAutoRedirect(false);
        filter_.AllowUI(false);
        filter_.CookieUsageBehavior(HttpCookieUsageBehavior::NoCookies);
        filter_.CacheControl().ReadBehavior(HttpCacheReadBehavior::NoCache);
        filter_.CacheControl().WriteBehavior(HttpCacheWriteBehavior::NoCache);
        client_ = HttpClient(filter_);
    }
    void set_token(std::string const &t) override
    {
        validate_token(t);
        token_ = t;
    }
    HttpResponse request(std::string const &method, std::string const &path, std::string const &body,
                         std::string const &etag) override
    {
        if (token_.empty())
            throw std::runtime_error("Device credential unavailable");
        if (path != "/attendance-events/batch" && path != "/gallery")
            throw std::invalid_argument("Unknown API path");
        auto deadline = std::chrono::steady_clock::now() + timeout_;
        HttpRequestMessage req(method == "POST" ? HttpMethod::Post() : HttpMethod::Get(),
                               Uri(winrt::to_hstring(base_ + path)));
        req.Headers().Authorization(
            winrt::Windows::Web::Http::Headers::HttpCredentialsHeaderValue(L"Bearer", winrt::to_hstring(token_)));
        if (!etag.empty())
            req.Headers().TryAppendWithoutValidation(L"If-None-Match", winrt::to_hstring(etag));
        if (method == "POST")
            req.Content(HttpStringContent(winrt::to_hstring(body), UnicodeEncoding::Utf8, L"application/json"));
        auto response = finish(client_.SendRequestAsync(req, HttpCompletionOption::ResponseHeadersRead), deadline);
        HttpResponse out;
        out.status = static_cast<int>(response.StatusCode());
        if (response.Headers().HasKey(L"ETag"))
            out.etag = winrt::to_string(response.Headers().Lookup(L"ETag"));
        if (response.Headers().HasKey(L"Retry-After"))
            out.retry_after = winrt::to_string(response.Headers().Lookup(L"Retry-After"));
        // Error handling needs status/headers only. A truncated error page must
        // not turn 401 or 429 into a generic transport failure that loses policy.
        if (out.status < 200 || out.status >= 300)
        {
            response.Close();
            return out;
        }
        if (response.Content())
        {
            auto stream = finish(response.Content().ReadAsInputStreamAsync(), deadline);
            const std::size_t limit = path == "/gallery" ? 32 * 1024 * 1024 : 1024 * 1024;
            for (;;)
            {
                auto buffer = finish(stream.ReadAsync(Buffer(65536), 65536, InputStreamOptions::Partial), deadline);
                if (!buffer.Length())
                    break;
                if (out.body.size() + buffer.Length() > limit)
                    throw std::runtime_error("HTTP response exceeds limit");
                auto reader = DataReader::FromBuffer(buffer);
                winrt::com_array<std::uint8_t> bytes(buffer.Length());
                reader.ReadBytes(bytes);
                out.body.append(reinterpret_cast<char const *>(bytes.data()), bytes.size());
            }
        }
        return out;
    }
};
} // namespace
std::unique_ptr<Transport> http_transport(std::string const &base, bool allow, std::chrono::milliseconds timeout)
{
    return std::make_unique<Http>(base, allow, timeout);
}
} // namespace edge_app
