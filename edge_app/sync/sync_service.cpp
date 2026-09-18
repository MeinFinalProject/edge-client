#include "sync/sync_service.hpp"
#include "sync/retry_policy.hpp"
namespace edge_app
{
SyncService::SyncService(OutboxRepository &outbox, GalleryRepository &gallery, Transport &http, std::string device,
                         std::string hash)
    : http_(http), attendance_(outbox, http, std::move(device)), gallery_(gallery, http, std::move(hash))
{
}
void SyncService::set_token(std::string const &token)
{
    if (token != token_)
    {
        http_.set_token(token);
        token_ = token;
        blocked_ = false;
        gallery_.reset_poll();
    }
}
void SyncService::step(std::int64_t now)
{
    if (blocked_ || token_.empty())
        return;
    RetryClock clock(now);
    if (attendance_.step(now) == SyncResult::AuthenticationBlocked)
    {
        blocked_ = true;
        return;
    }
    if (gallery_.step(clock.now()) == SyncResult::AuthenticationBlocked)
        blocked_ = true;
}
} // namespace edge_app
