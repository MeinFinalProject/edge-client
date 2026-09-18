#include "domain/uuid_v7.hpp"
#include <Windows.h>
#include <array>
#include <bcrypt.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace edge_app
{
std::int64_t utc_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
std::string uuid_v7(std::int64_t ms)
{
    if (ms < 0 || ms > 0xffffffffffffLL)
        throw std::invalid_argument("UUID timestamp out of range");
    std::array<unsigned char, 16> b{};
    if (BCryptGenRandom(nullptr, b.data(), 16, BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0)
        throw std::runtime_error("CNG random failed");
    for (int i = 5; i >= 0; --i)
    {
        b[i] = static_cast<unsigned char>(ms);
        ms >>= 8;
    }
    b[6] = (b[6] & 15) | 0x70;
    b[8] = (b[8] & 63) | 0x80;
    std::ostringstream o;
    o << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i)
    {
        if (i == 4 || i == 6 || i == 8 || i == 10)
            o << '-';
        o << std::setw(2) << unsigned(b[i]);
    }
    return o.str();
}

} // namespace edge_app
