#include "sync/attendance_sync.hpp"
#include "sync/retry_policy.hpp"
#include "sync/wire_format.hpp"
#include <algorithm>
#include <unordered_set>
namespace edge_app
{
AttendanceSync::AttendanceSync(OutboxRepository &outbox, Transport &http, std::string device)
    : outbox_(outbox), http_(http), device_(std::move(device))
{
}
SyncResult AttendanceSync::step(std::int64_t now)
{
    RetryClock clock(now);
    auto rows = outbox_.due(now);
    if (!rows.empty())
    {
        // A newly queued event must not reset the backoff of older failed rows.
        auto attempts = [&] {
            int maximum = 0;
            for (auto const &row : rows)
                maximum = std::max(maximum, row.attempts);
            return maximum;
        };
        HttpResponse r;
        bool received = false;
        try
        {
            r = http_.request("POST", "/attendance-events/batch", batch_payload(device_, rows), "");
            received = true;
        }
        catch (...)
        {
            outbox_.retry(rows, retry_at(clock.now(), attempts(), "", retry_jitter()), "transport_failure");
        }
        if (received)
        {
            if (r.status == 401 || r.status == 403)
            {
                return SyncResult::AuthenticationBlocked;
            }
            if (r.status == 400 || r.status == 422)
                outbox_.dead_letter(rows, "schema_rejected_" + std::to_string(r.status));
            else if (r.status >= 200 && r.status < 300)
            {
                std::vector<DeliveryReceipt> receipts;
                bool valid = false;
                try
                {
                    receipts = parse_receipts(r.body, rows);
                    valid = true;
                }
                catch (...)
                {
                }
                if (valid)
                {
                    outbox_.acknowledge(receipts);
                    std::unordered_set<std::string> done;
                    for (auto const &a : receipts)
                        done.insert(a.event_id);
                    std::erase_if(rows, [&](auto const &e) { return done.contains(e.id); });
                }
                if (!rows.empty())
                    outbox_.retry(rows, retry_at(clock.now(), attempts(), r.retry_after, retry_jitter()),
                                  valid ? "missing_ack" : "invalid_ack");
            }
            else
                outbox_.retry(rows, retry_at(clock.now(), attempts(), r.retry_after, retry_jitter()),
                              "http_" + std::to_string(r.status));
        }
    }

    return SyncResult::Continue;
}

} // namespace edge_app
