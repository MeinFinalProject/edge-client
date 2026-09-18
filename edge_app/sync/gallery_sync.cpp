#include "sync/gallery_sync.hpp"
#include "sync/retry_policy.hpp"
#include "sync/wire_format.hpp"
#include <algorithm>
namespace edge_app
{
GallerySync::GallerySync(GalleryRepository &gallery, Transport &http, std::string hash)
    : gallery_(gallery), http_(http), hash_(std::move(hash))
{
}
SyncResult GallerySync::step(std::int64_t now)
{
    RetryClock clock(now);
    if (now < gallery_due_)
        return SyncResult::Continue;
    auto old = gallery_.load_metadata();
    HttpResponse r;
    try
    {
        r = http_.request("GET", "/gallery", "", old.etag);
    }
    catch (...)
    {
        gallery_due_ = retry_at(clock.now(), gallery_attempts_, "", retry_jitter());
        gallery_attempts_ = std::min(gallery_attempts_ + 1, 30);
        return SyncResult::Continue;
    }
    if (r.status == 401 || r.status == 403)
    {
        return SyncResult::AuthenticationBlocked;
    }
    if (r.status == 304 && !old.version.empty())
    {
        gallery_due_ = clock.now() + 60000;
        gallery_attempts_ = 0;
        return SyncResult::Continue;
    }
    if (r.status == 200)
    {
        GallerySnapshot next;
        bool valid = false;
        try
        {
            next = parse_gallery(r.body, r.etag, hash_);
            valid = true;
        }
        catch (...)
        {
        }
        if (valid)
        {
            gallery_.replace(next);
            dirty_ = true;
            gallery_due_ = clock.now() + 60000;
            gallery_attempts_ = 0;
            return SyncResult::Continue;
        }
    }
    gallery_due_ = retry_at(clock.now(), gallery_attempts_, r.retry_after, retry_jitter());
    gallery_attempts_ = std::min(gallery_attempts_ + 1, 30);

    return SyncResult::Continue;
}

} // namespace edge_app
