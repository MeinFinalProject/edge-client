#pragma once

#include <string>

namespace vision_runtime::detail {

struct LegacyLowLightState {
    bool device_found = false;
    bool camera_control_available = false;
    bool supported = false;

    long value = 0;
    long flags = 0;
    long minimum = 0;
    long maximum = 0;
    long step = 0;
    long default_value = 0;
    long caps = 0;

    std::string error;
};

LegacyLowLightState query_legacy_low_light_priority(
    std::wstring const& friendly_name
);

bool set_legacy_low_light_priority(
    std::wstring const& friendly_name,
    long value,
    std::string& error
);

} // namespace vision_runtime::detail
