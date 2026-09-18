#include "sync/retry_policy.hpp"
#include <Windows.h>
#include <algorithm>
#include <bcrypt.h>
#include <ctime>
#include <iomanip>
#include <sstream>
namespace edge_app
{
unsigned retry_jitter()
{
    unsigned n = 0;
    if (BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(&n), sizeof n, BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0)
        return 250;
    return n % 501;
}
std::int64_t retry_at(std::int64_t now, int attempts, std::string const &header, unsigned jitter_ms)
{
    auto delay = std::min<std::int64_t>(300000, 1000LL << std::clamp(attempts, 0, 9));
    std::int64_t server_delay = 0;
    if (!header.empty())
    {
        if (header.size() <= 10 && header.find_first_not_of("0123456789") == std::string::npos)
            server_delay = std::stoll(header) * 1000;
        else
        {
            std::tm tm{};
            std::istringstream s(header);
            s.imbue(std::locale::classic());
            s >> std::get_time(&tm, "%a, %d %b %Y %H:%M:%S GMT");
            if (!s.fail())
            {
                auto sec = _mkgmtime64(&tm);
                if (sec > 0)
                    server_delay = std::max<std::int64_t>(0, sec * 1000 - now);
            }
        }
    }
    return now + std::max(delay, server_delay) + std::min(jitter_ms, 500U);
}

} // namespace edge_app
