#include "security/device_credential.hpp"
#include <Windows.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include <wincrypt.h>

namespace edge_app
{
void validate_token(std::string const &t)
{
    if (t.empty() || t.size() > 4096)
        throw std::invalid_argument("Invalid credential length");
    for (unsigned char c : t)
        if (c <= 32 || c >= 127)
            throw std::invalid_argument("Credential contains unsupported characters");
}
namespace
{
struct Blob
{
    DATA_BLOB b{};
    ~Blob()
    {
        if (b.pbData)
        {
            SecureZeroMemory(b.pbData, b.cbData);
            LocalFree(b.pbData);
        }
    }
};
} // namespace
void protect_token(std::filesystem::path const &file, std::string const &token)
{
    validate_token(token);
    DATA_BLOB input{static_cast<DWORD>(token.size()), reinterpret_cast<BYTE *>(const_cast<char *>(token.data()))};
    Blob out;
    if (!CryptProtectData(&input, L"Edge client device token", nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN,
                          &out.b))
        throw std::runtime_error("DPAPI protection failed");
    if (file.has_parent_path())
        std::filesystem::create_directories(file.parent_path());
    auto temporary = file;
    temporary += L".new";
    {
        std::ofstream f(temporary, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<char *>(out.b.pbData), out.b.cbData);
        f.flush();
        if (!f)
            throw std::runtime_error("Credential write failed");
    }
    if (!MoveFileExW(temporary.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Credential replacement failed");
}
std::string unprotect_token(std::filesystem::path const &file)
{
    const auto size = std::filesystem::file_size(file);
    if (size == 0 || size > 16384)
        throw std::runtime_error("Invalid protected credential size");
    std::ifstream f(file, std::ios::binary);
    std::vector<BYTE> bytes((std::istreambuf_iterator<char>(f)), {});
    if (bytes.size() != size)
        throw std::runtime_error("Credential read failed");
    DATA_BLOB in{static_cast<DWORD>(bytes.size()), bytes.data()};
    Blob out;
    if (!CryptUnprotectData(&in, nullptr, nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out.b))
        throw std::runtime_error("DPAPI unprotect failed for current Windows user");
    std::string token(reinterpret_cast<char *>(out.b.pbData), out.b.cbData);
    validate_token(token);
    return token;
}
} // namespace edge_app
