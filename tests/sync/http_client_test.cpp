// Real native HTTP client against a loopback socket; no backend/cloud required.
#include "sync/http_client.hpp"
#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <winrt/base.h>
#include <winsock2.h>
#include <ws2tcpip.h>

namespace
{
using namespace std::chrono_literals;
void check(bool ok, char const *why)
{
    if (!ok)
        throw std::runtime_error(why);
}
struct Socket
{
    SOCKET value = INVALID_SOCKET;
    ~Socket()
    {
        if (value != INVALID_SOCKET)
            closesocket(value);
    }
};
class Server
{
    Socket listener;
    std::jthread thread;
    std::mutex mutex;
    std::string observed;

  public:
    unsigned short port = 0;
    std::atomic_int requests = 0;
    explicit Server(std::function<std::string(std::string const &)> answer)
    {
        listener.value = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        check(listener.value != INVALID_SOCKET, "socket failed");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        check(bind(listener.value, reinterpret_cast<sockaddr *>(&address), sizeof address) == 0, "bind failed");
        int size = sizeof address;
        check(getsockname(listener.value, reinterpret_cast<sockaddr *>(&address), &size) == 0, "getsockname failed");
        port = ntohs(address.sin_port);
        check(listen(listener.value, 4) == 0, "listen failed");
        thread = std::jthread([this, answer](std::stop_token stop) {
            while (!stop.stop_requested())
            {
                fd_set ready;
                FD_ZERO(&ready);
                FD_SET(listener.value, &ready);
                timeval timeout{0, 100000};
                if (select(0, &ready, nullptr, nullptr, &timeout) <= 0)
                    continue;
                Socket peer{accept(listener.value, nullptr, nullptr)};
                if (peer.value == INVALID_SOCKET)
                    continue;
                DWORD receive_timeout = 2000;
                setsockopt(peer.value, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char *>(&receive_timeout),
                           sizeof receive_timeout);
                std::string request;
                char buffer[4096];
                while (request.size() < 1024 * 1024)
                {
                    int n = recv(peer.value, buffer, sizeof buffer, 0);
                    if (n <= 0)
                        break;
                    request.append(buffer, n);
                    auto end = request.find("\r\n\r\n");
                    if (end != std::string::npos)
                    {
                        auto key = request.find("Content-Length:");
                        if (key == std::string::npos)
                            key = request.find("content-length:");
                        std::size_t length = key == std::string::npos ? 0 : std::stoul(request.substr(key + 15));
                        if (request.size() >= end + 4 + length)
                            break;
                    }
                }
                {
                    std::lock_guard lock(mutex);
                    observed = request;
                }
                ++requests;
                auto response = answer(request);
                std::size_t offset = 0;
                while (offset < response.size())
                {
                    int n = send(peer.value, response.data() + offset,
                                 static_cast<int>(std::min<std::size_t>(65536, response.size() - offset)), 0);
                    if (n <= 0)
                        break;
                    offset += n;
                }
            }
        });
    }
    ~Server()
    {
        thread.request_stop();
        if (thread.joinable())
            thread.join();
    }
    std::string url() const
    {
        return "http://127.0.0.1:" + std::to_string(port);
    }
    std::string request()
    {
        std::lock_guard lock(mutex);
        return observed;
    }
};
std::string response(int status, std::string body, std::string headers = "")
{
    return "HTTP/1.1 " + std::to_string(status) +
           " Test\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: " +
           std::to_string(body.size()) + "\r\n" + headers + "\r\n" + body;
}
} // namespace
int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data))
        return 1;
    struct Cleanup
    {
        ~Cleanup()
        {
            WSACleanup();
        }
    } cleanup;
    try
    {
        Server ok([](auto const &) {
            return response(200, "{\"schema_version\":1,\"results\":[]}", "ETag: \"v1\"\r\nRetry-After: 120\r\n");
        });
        auto client = edge_app::http_transport(ok.url(), true, 2s);
        client->set_token("local-test-token");
        auto r = client->request("POST", "/attendance-events/batch", "{\"events\":[]}", "");
        check(r.status == 200 && r.etag == "\"v1\"" && r.retry_after == "120", "HTTP headers/status failed");
        check(ok.request().find("Bearer local-test-token") != std::string::npos &&
                  ok.request().find("{\"events\":[]}") != std::string::npos,
              "POST auth/body failed");
        r = client->request("GET", "/gallery", "", "\"v1\"");
        check(ok.request().find("If-None-Match: \"v1\"") != std::string::npos, "Conditional gallery header missing");
        Server redirect([&](auto const &) { return response(307, "", "Location: " + ok.url() + "/gallery\r\n"); });
        auto redirect_client = edge_app::http_transport(redirect.url(), true, 2s);
        redirect_client->set_token("local-test-token");
        auto before = ok.requests.load();
        r = redirect_client->request("GET", "/gallery", "", "");
        check(r.status == 307 && ok.requests == before, "Redirect followed with device credential");
        Server slow([](auto const &) {
            std::this_thread::sleep_for(500ms);
            return response(200, "{}");
        });
        auto timed = edge_app::http_transport(slow.url(), true, 100ms);
        timed->set_token("local-test-token");
        bool caught = false;
        auto start = std::chrono::steady_clock::now();
        try
        {
            (void)timed->request("GET", "/gallery", "", "");
        }
        catch (...)
        {
            caught = true;
        }
        check(caught && std::chrono::steady_clock::now() - start < 2s, "Request deadline not enforced");
        Server large([](auto const &) { return response(200, std::string(1024 * 1024 + 1, 'x')); });
        auto limited = edge_app::http_transport(large.url(), true, 2s);
        limited->set_token("local-test-token");
        caught = false;
        try
        {
            (void)limited->request("POST", "/attendance-events/batch", "{}", "");
        }
        catch (...)
        {
            caught = true;
        }
        check(caught, "Response size limit not enforced");
        Server auth_error([](auto const &) { return response(401, std::string(1024 * 1024 + 1, 'x')); });
        auto auth_client = edge_app::http_transport(auth_error.url(), true, 2s);
        auth_client->set_token("local-test-token");
        r = auth_client->request("POST", "/attendance-events/batch", "{}", "");
        check(r.status == 401 && r.body.empty(), "Oversized error page hid the authentication status");
        Server rate_error([](auto const &) {
            return std::string("HTTP/1.1 429 Test\r\nConnection: close\r\nContent-Length: 10000\r\n"
                               "Retry-After: 120\r\n\r\n"); // Deliberately missing the promised error body.
        });
        auto rate_client = edge_app::http_transport(rate_error.url(), true, 2s);
        rate_client->set_token("local-test-token");
        r = rate_client->request("POST", "/attendance-events/batch", "{}", "");
        check(r.status == 429 && r.retry_after == "120", "Truncated error page lost Retry-After");
        Server lost([](auto const &) { return std::string{}; });
        auto dropped = edge_app::http_transport(lost.url(), true, 2s);
        dropped->set_token("local-test-token");
        caught = false;
        try
        {
            (void)dropped->request("POST", "/attendance-events/batch", "{}", "");
        }
        catch (...)
        {
            caught = true;
        }
        check(caught, "Dropped connection treated as success");
        std::cout << "Native HTTP loopback: body/auth, ETag, redirect, timeout, size limit, dropped connection: PASS\n";
        return 0;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "WinRT " << std::hex << e.code().value << '\n';
        return 1;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
