#include "sync/device_channel.hpp"
#include "sync/http_client.hpp"
#include "security/device_credential.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Networking.Sockets.h>
#include <winrt/Windows.Storage.Streams.h>

namespace edge_app
{
namespace
{
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
using namespace winrt::Windows::Data::Json;
using namespace winrt::Windows::Networking::Sockets;
using namespace winrt::Windows::Storage::Streams;
template <typename T> auto finish(T const &operation, std::stop_token stop, std::chrono::seconds timeout)
{
    auto deadline = Clock::now() + timeout;
    while (operation.wait_for(100ms) == winrt::Windows::Foundation::AsyncStatus::Started)
        if (stop.stop_requested() || Clock::now() >= deadline)
        {
            operation.Cancel();
            throw std::runtime_error("Device channel operation cancelled or timed out");
        }
    return operation.GetResults();
}
struct State
{
    std::mutex mutex;
    std::condition_variable cv;
    std::string runtime = "starting", version, channel = "connecting";
    std::size_t templates = 0;
    std::optional<Clock::time_point> frame;
};
// Socket callbacks own only this small state, never application/repository references.
struct Connection
{
    std::atomic_bool closed{false}, reconcile{false};
    std::atomic<Clock::duration::rep> last_server{Clock::now().time_since_epoch().count()};
};
struct Apartment
{
    Apartment() { winrt::init_apartment(winrt::apartment_type::multi_threaded); }
    ~Apartment() { winrt::uninit_apartment(); }
};
}
struct DeviceChannel::Impl
{
    std::shared_ptr<State> state = std::make_shared<State>();
    std::jthread worker;
    Impl(std::string base, bool loopback, std::filesystem::path credential, std::string hash,
         OutboxRepository &outbox, std::function<void()> reconcile)
    {
        worker = std::jthread([=, &outbox, shared = state](std::stop_token stop) {
            auto status = [&](std::string value) { std::lock_guard lock(shared->mutex); shared->channel = std::move(value); };
            try
            {
                Apartment apartment;
                // Reuse the strict HTTP URL policy, including explicit loopback-only test support.
                auto validated = http_transport(base, loopback);
                auto url = base;
                while (!url.empty() && url.back() == '/') url.pop_back();
                url.replace(0, url.find(':'), url.starts_with("https:") ? "wss" : "ws");
                url += "/device-channel";
                unsigned failures = 0;
                std::minstd_rand random(std::random_device{}());
                while (!stop.stop_requested())
                {
                    MessageWebSocket socket{nullptr};
                    auto connected_at = Clock::now();
                    try
                    {
                        auto token = unprotect_token(credential);
                        validate_token(token);
                        socket = MessageWebSocket();
                        socket.Control().MessageType(SocketMessageType::Utf8);
                        socket.Control().MaxMessageSize(4096);
                        socket.SetRequestHeader(L"Authorization", winrt::to_hstring("Bearer " + token));
                        auto connection = std::make_shared<Connection>();
                        socket.MessageReceived([connection](auto const &, MessageWebSocketMessageReceivedEventArgs const &args) {
                            try
                            {
                                if (args.MessageType() != SocketMessageType::Utf8) throw std::runtime_error("Invalid channel message");
                                auto reader = args.GetDataReader();
                                auto json = JsonObject::Parse(reader.ReadString(reader.UnconsumedBufferLength()));
                                auto type = json.GetNamedString(L"type");
                                if (json.Size() != 2 || json.GetNamedNumber(L"schema_version") != 1 ||
                                    (type != L"reconcile" && type != L"status_ack")) throw std::runtime_error("Invalid channel message");
                                connection->last_server = Clock::now().time_since_epoch().count();
                                if (type == L"reconcile") connection->reconcile = true;
                            }
                            catch (...) { connection->closed = true; }
                        });
                        socket.Closed([connection](auto const &, auto const &) { connection->closed = true; });
                        status("connecting");
                        finish(socket.ConnectAsync(winrt::Windows::Foundation::Uri(winrt::to_hstring(url))), stop, 10s);
                        connected_at = Clock::now();
                        status("connected");
                        reconcile(); // Recover notifications missed while offline, even before server hello.
                        auto due = Clock::now();
                        while (!stop.stop_requested() && !connection->closed)
                        {
                            if (connection->reconcile.exchange(false)) reconcile();
                            auto now = Clock::now();
                            if (now - Clock::time_point(Clock::duration(connection->last_server.load())) > 35s)
                                throw std::runtime_error("Device channel acknowledgement expired");
                            if (now >= due)
                            {
                                if (unprotect_token(credential) != token) break; // DPAPI credential rotation.
                                JsonObject report;
                                report.Insert(L"schema_version", JsonValue::CreateNumberValue(1));
                                report.Insert(L"type", JsonValue::CreateStringValue(L"status"));
                                report.Insert(L"model_sha256", JsonValue::CreateStringValue(winrt::to_hstring(hash)));
                                {
                                    std::lock_guard lock(shared->mutex);
                                    report.Insert(L"runtime_state", JsonValue::CreateStringValue(winrt::to_hstring(shared->runtime)));
                                    report.Insert(L"installed_gallery_version", JsonValue::CreateStringValue(winrt::to_hstring(shared->version)));
                                    report.Insert(L"installed_template_count", JsonValue::CreateNumberValue(static_cast<double>(shared->templates)));
                                    report.Insert(L"frame_age_ms", shared->frame ? JsonValue::CreateNumberValue(static_cast<double>(std::clamp<std::int64_t>(
                                        std::chrono::duration_cast<std::chrono::milliseconds>(now - *shared->frame).count(), 0, 86400000))) : JsonValue::CreateNullValue());
                                }
                                // If SQLite cannot even be read, disconnect rather than inventing zero queue counts.
                                report.Insert(L"outbox_pending_count", JsonValue::CreateNumberValue(static_cast<double>(outbox.pending_count())));
                                report.Insert(L"outbox_dead_count", JsonValue::CreateNumberValue(static_cast<double>(outbox.dead_count())));
                                DataWriter writer(socket.OutputStream());
                                writer.WriteString(report.Stringify());
                                finish(writer.StoreAsync(), stop, 5s);
                                writer.DetachStream();
                                due = now + 10s;
                            }
                            std::unique_lock lock(shared->mutex);
                            shared->cv.wait_for(lock, 100ms, [&] { return stop.stop_requested(); });
                        }
                        if (Clock::now() - connected_at > 30s) failures = 0;
                    }
                    catch (...) { status("disconnected_or_credential_unavailable"); }
                    if (socket) { try { socket.Close(1001, L""); } catch (...) {} }
                    if (!stop.stop_requested())
                    {
                        status("reconnecting");
                        auto delay = std::chrono::milliseconds(1000 * (1u << std::min(failures++, 5u)) + random() % 1000);
                        std::unique_lock lock(shared->mutex);
                        shared->cv.wait_for(lock, delay, [&] { return stop.stop_requested(); });
                    }
                }
                status("stopped");
            }
            catch (...) { status("channel_initialization_failed"); }
        });
    }
    ~Impl()
    {
        worker.request_stop();
        state->cv.notify_all();
        if (worker.joinable()) worker.join();
    }
};
DeviceChannel::DeviceChannel(std::string base, bool loopback, std::filesystem::path credential, std::string hash,
    OutboxRepository &outbox, std::function<void()> reconcile)
    : impl_(std::make_unique<Impl>(std::move(base), loopback, std::move(credential), std::move(hash), outbox, std::move(reconcile))) {}
DeviceChannel::~DeviceChannel() = default;
void DeviceChannel::runtime_state(std::string value) { std::lock_guard lock(impl_->state->mutex); impl_->state->runtime = std::move(value); }
void DeviceChannel::frame_processed() { std::lock_guard lock(impl_->state->mutex); impl_->state->frame = Clock::now(); }
void DeviceChannel::gallery_installed(std::string version, std::size_t templates)
{
    std::lock_guard lock(impl_->state->mutex);
    impl_->state->version = std::move(version);
    impl_->state->templates = templates;
}
std::string DeviceChannel::status() const { std::lock_guard lock(impl_->state->mutex); return impl_->state->channel; }
} // namespace edge_app
