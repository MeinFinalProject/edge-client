#pragma once
#include <cstdint>
#include <string>
namespace edge_app
{
std::int64_t utc_ms();
std::string uuid_v7(std::int64_t timestamp_ms);
} // namespace edge_app
