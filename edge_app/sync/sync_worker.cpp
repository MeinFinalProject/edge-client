#include "sync/sync_worker.hpp"
#include "domain/uuid_v7.hpp"
#include "security/device_credential.hpp"
#include "sync/network_watcher.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <winrt/base.h>
namespace edge_app
{
struct SyncWorker::Impl
{
    struct Signal
    {
        std::mutex mutex;
        std::condition_variable cv;
        bool wake = false;
    };
    std::shared_ptr<Signal> signal = std::make_shared<Signal>();
    SyncService &sync;
    std::filesystem::path credential;
    mutable std::mutex status_mutex;
    std::string state = "starting";
    std::unique_ptr<NetworkWatcher> network;
    std::jthread worker;
    void status(std::string s)
    {
        std::lock_guard l(status_mutex);
        state = std::move(s);
    }
    Impl(SyncService &s, std::filesystem::path file) : sync(s), credential(std::move(file))
    {
        std::weak_ptr<Signal> weak = signal;
        network = std::make_unique<NetworkWatcher>([weak] {
            if (auto p = weak.lock())
            {
                std::lock_guard lock(p->mutex);
                p->wake = true;
                p->cv.notify_one();
            }
        });
        worker = std::jthread([this](std::stop_token stop) {
            try
            {
                winrt::init_apartment(winrt::apartment_type::multi_threaded);
            }
            catch (...)
            {
                status("worker_initialization_failed");
                return;
            }
            while (!stop.stop_requested())
            {
                try
                {
                    sync.set_token(unprotect_token(credential));
                    sync.step(utc_ms());
                    status(sync.auth_blocked() ? "authentication_blocked" : "running");
                }
                catch (...)
                {
                    status("sync_error_or_credential_unavailable");
                }
                std::unique_lock l(signal->mutex);
                signal->cv.wait_for(l, std::chrono::seconds(1), [&] { return signal->wake || stop.stop_requested(); });
                signal->wake = false;
            }
            winrt::uninit_apartment();
        });
    }
    ~Impl()
    {
        network.reset();
        worker.request_stop();
        signal->cv.notify_all();
        if (worker.joinable())
            worker.join();
    }
};
SyncWorker::SyncWorker(SyncService &s, std::filesystem::path p) : impl_(std::make_unique<Impl>(s, std::move(p)))
{
}
SyncWorker::~SyncWorker() = default;
void SyncWorker::wake()
{
    std::lock_guard l(impl_->signal->mutex);
    impl_->signal->wake = true;
    impl_->signal->cv.notify_one();
}
std::string SyncWorker::status() const
{
    std::lock_guard l(impl_->status_mutex);
    return impl_->state;
}

} // namespace edge_app
