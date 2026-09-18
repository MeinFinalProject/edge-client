#include "sync/network_watcher.hpp"
#include <winrt/Windows.Networking.Connectivity.h>
namespace edge_app
{
struct NetworkWatcher::Impl
{
    winrt::event_token token{};
    bool subscribed = false;
    explicit Impl(std::function<void()> wake)
    {
        try
        {
            token = winrt::Windows::Networking::Connectivity::NetworkInformation::NetworkStatusChanged(
                [wake = std::move(wake)](auto const &) { wake(); });
            subscribed = true;
        }
        catch (...)
        { /* Polling remains available. */
        }
    }
    ~Impl()
    {
        try
        {
            if (subscribed)
                winrt::Windows::Networking::Connectivity::NetworkInformation::NetworkStatusChanged(token);
        }
        catch (...)
        {
        }
    }
};
NetworkWatcher::NetworkWatcher(std::function<void()> wake) : impl_(std::make_unique<Impl>(std::move(wake)))
{
}
NetworkWatcher::~NetworkWatcher() = default;
} // namespace edge_app
