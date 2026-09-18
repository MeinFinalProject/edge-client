#pragma once
#include <cstdint>
#include <string>
namespace edge_app
{
// Immutable wire payload after enqueue; retries only change delivery metadata.
struct AttendanceRecord
{
    std::string id, payload;
    std::int64_t occurred_ms = 0, next_attempt_ms = 0;
    int attempts = 0;
};
// Application data at the boundary from the vision runtime. No Windows API types.
struct AttendanceObservation
{
    std::int64_t occurred_at_utc_ms = 0;
    std::uint64_t camera_frame_id = 0, track_id = 0;
    std::string identity_id;
    float similarity = 0, similarity_margin = 0, pad_median_p_real = 0;
};
struct DeliveryReceipt
{
    std::string event_id, disposition;
};
} // namespace edge_app
