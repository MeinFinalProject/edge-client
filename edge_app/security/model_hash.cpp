#include "security/model_hash.hpp"
#include <Windows.h>
#include <array>
#include <bcrypt.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace edge_app
{
std::string sha256_file(std::filesystem::path const &p)
{
    struct Handles
    {
        BCRYPT_ALG_HANDLE a = nullptr;
        BCRYPT_HASH_HANDLE h = nullptr;
        ~Handles()
        {
            if (h)
                BCryptDestroyHash(h);
            if (a)
                BCryptCloseAlgorithmProvider(a, 0);
        }
    } s;
    auto check = [](NTSTATUS n) {
        if (n < 0)
            throw std::runtime_error("SHA-256 failure");
    };
    check(BCryptOpenAlgorithmProvider(&s.a, BCRYPT_SHA256_ALGORITHM, nullptr, 0));
    check(BCryptCreateHash(s.a, &s.h, nullptr, 0, nullptr, 0, 0));
    std::ifstream in(p, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot read model for hash");
    std::array<unsigned char, 65536> b{};
    while (in.read(reinterpret_cast<char *>(b.data()), b.size()) || in.gcount())
        check(BCryptHashData(s.h, b.data(), static_cast<ULONG>(in.gcount()), 0));
    if (!in.eof())
        throw std::runtime_error("Model read failure");
    std::array<unsigned char, 32> d{};
    check(BCryptFinishHash(s.h, d.data(), 32, 0));
    std::ostringstream o;
    o << std::hex << std::setfill('0');
    for (auto v : d)
        o << std::setw(2) << unsigned(v);
    return o.str();
}

} // namespace edge_app
