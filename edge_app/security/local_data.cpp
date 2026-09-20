#include "security/local_data.hpp"
#include <Windows.h>
#include <aclapi.h>
#include <sddl.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace edge_app
{
namespace
{
struct Handle
{
    HANDLE value = INVALID_HANDLE_VALUE;
    ~Handle()
    {
        if (value != INVALID_HANDLE_VALUE && value)
            CloseHandle(value);
    }
};
struct Local
{
    void *value = nullptr;
    ~Local()
    {
        if (value)
            LocalFree(value);
    }
};
void require(bool ok)
{
    if (!ok)
        throw std::runtime_error("Local data permission hardening failed (Win32 " + std::to_string(GetLastError()) +
                                 ")");
}
void reject_reparse(std::filesystem::path const &path)
{
    for (auto p = path; !p.empty(); p = p.parent_path())
    {
        auto attributes = GetFileAttributesW(p.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_REPARSE_POINT))
            throw std::invalid_argument("Local data paths must not use reparse points");
        if (p == p.root_path())
            break;
    }
}
void harden(std::filesystem::path const &path, PSECURITY_DESCRIPTOR descriptor, bool directory)
{
    Handle object{CreateFileW(path.c_str(), READ_CONTROL | WRITE_DAC,
                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                              FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr)};
    require(object.value != INVALID_HANDLE_VALUE);
    BY_HANDLE_FILE_INFORMATION info{};
    require(GetFileInformationByHandle(object.value, &info));
    if ((info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) ||
        bool(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != directory || (!directory && info.nNumberOfLinks != 1))
        throw std::invalid_argument("Local data must use ordinary directories and single-link files");
    PACL acl = nullptr;
    BOOL present = FALSE, defaulted = FALSE;
    require(GetSecurityDescriptorDacl(descriptor, &present, &acl, &defaulted) && present && acl);
    auto error =
        SetSecurityInfo(object.value, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                        nullptr, nullptr, acl, nullptr);
    if (error != ERROR_SUCCESS)
        throw std::runtime_error("Cannot restrict local data permissions (Win32 " + std::to_string(error) + ")");
}
} // namespace
void prepare_private_file(std::filesystem::path const &file)
{
    auto path = std::filesystem::absolute(file).lexically_normal();
    auto parent = path.parent_path();
    if (path.filename().empty() || path.native().find(L':', 2) != std::wstring::npos ||
        path.native().starts_with(L"\\\\") || parent == parent.root_path() ||
        _wcsicmp(parent.c_str(), std::filesystem::current_path().c_str()) == 0)
        throw std::invalid_argument("Use a dedicated local data directory, not a root or working directory");
    for (auto name : {L"USERPROFILE", L"LOCALAPPDATA", L"APPDATA", L"PROGRAMDATA", L"WINDIR"})
    {
        wchar_t value[32768];
        auto size = GetEnvironmentVariableW(name, value, 32768);
        if (size && size < 32768 && _wcsicmp(parent.c_str(), value) == 0)
            throw std::invalid_argument("Use an application subdirectory for private local data");
    }
    reject_reparse(path);
    // Validate an existing target before changing its parent's inheritable ACL.
    // Otherwise a rejected hard link could already change another file's ACL.
    {
        Handle existing{CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                    FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr)};
        if (existing.value != INVALID_HANDLE_VALUE)
        {
            BY_HANDLE_FILE_INFORMATION info{};
            require(GetFileInformationByHandle(existing.value, &info));
            if ((info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) ||
                info.nNumberOfLinks != 1)
                throw std::invalid_argument("Local data must use an ordinary single-link file");
        }
        else if (GetLastError() != ERROR_FILE_NOT_FOUND && GetLastError() != ERROR_PATH_NOT_FOUND)
            require(false);
    }
    Handle token;
    require(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token.value));
    DWORD length = 0;
    GetTokenInformation(token.value, TokenUser, nullptr, 0, &length);
    std::vector<BYTE> user(length);
    require(GetTokenInformation(token.value, TokenUser, user.data(), length, &length));
    Local sid;
    require(ConvertSidToStringSidW(reinterpret_cast<TOKEN_USER *>(user.data())->User.Sid,
                                   reinterpret_cast<LPWSTR *>(&sid.value)));
    auto sddl = std::wstring(L"D:P(A;OICI;FA;;;") + static_cast<wchar_t *>(sid.value) + L")(A;OICI;FA;;;SY)";
    Local descriptor;
    require(ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &descriptor.value,
                                                                 nullptr));
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), descriptor.value, FALSE};
    std::filesystem::create_directories(parent.parent_path());
    if (!CreateDirectoryW(parent.c_str(), &attributes) && GetLastError() != ERROR_ALREADY_EXISTS)
        require(false);
    harden(parent, descriptor.value, true);
    // Create with the restrictive ACL before any secret/database bytes are written.
    {
        Handle created{CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &attributes,
                                   CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr)};
        if (created.value == INVALID_HANDLE_VALUE && GetLastError() != ERROR_FILE_EXISTS)
            require(false);
    }
    harden(path, descriptor.value, false);
}
} // namespace edge_app
