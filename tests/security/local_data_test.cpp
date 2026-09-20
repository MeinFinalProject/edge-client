#include "security/local_data.hpp"
#include "support/database_access.hpp"
#include "support/edge_fixtures.hpp"
#include <aclapi.h>
#include <sddl.h>
using namespace edge_test;
namespace
{
std::wstring acl_text(std::filesystem::path const &path)
{
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    check(GetNamedSecurityInfoW(path.c_str(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, nullptr,
                                nullptr, &descriptor) == ERROR_SUCCESS,
          "Cannot snapshot ACL");
    LPWSTR text = nullptr;
    auto ok = ConvertSecurityDescriptorToStringSecurityDescriptorW(descriptor, SDDL_REVISION_1,
                                                                   DACL_SECURITY_INFORMATION, &text, nullptr);
    LocalFree(descriptor);
    check(ok, "Cannot stringify ACL");
    std::wstring result(text);
    LocalFree(text);
    return result;
}
void restricted(std::filesystem::path const &path, bool inherited = false)
{
    PACL acl = nullptr;
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    auto error = GetNamedSecurityInfoW(path.c_str(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &acl,
                                       nullptr, &descriptor);
    check(error == ERROR_SUCCESS, "Cannot read data ACL");
    struct Free
    {
        void *p;
        ~Free()
        {
            LocalFree(p);
        }
    } free{descriptor};
    SECURITY_DESCRIPTOR_CONTROL control{};
    DWORD revision = 0;
    check(GetSecurityDescriptorControl(descriptor, &control, &revision) && (inherited || (control & SE_DACL_PROTECTED)),
          "Data ACL still inherits broad parent permissions");
    check(acl && acl->AceCount == 2, "Expected current user and SYSTEM only");
    HANDLE token = nullptr;
    check(OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token), "Open token failed");
    DWORD length = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &length);
    std::vector<BYTE> buffer(length);
    auto ok = GetTokenInformation(token, TokenUser, buffer.data(), length, &length);
    CloseHandle(token);
    check(ok, "Read token user failed");
    bool user = false, system = false;
    for (DWORD i = 0; i < acl->AceCount; ++i)
    {
        void *raw = nullptr;
        check(GetAce(acl, i, &raw), "Read ACE failed");
        auto ace = static_cast<ACCESS_ALLOWED_ACE *>(raw);
        check(ace->Header.AceType == ACCESS_ALLOWED_ACE_TYPE && (inherited || !(ace->Header.AceFlags & INHERITED_ACE)),
              "Unexpected or inherited permission");
        if (EqualSid(&ace->SidStart, reinterpret_cast<TOKEN_USER *>(buffer.data())->User.Sid))
            user = true;
        else if (IsWellKnownSid(&ace->SidStart, WinLocalSystemSid))
            system = true;
        else
            throw std::runtime_error("Data access granted to an unrelated principal");
    }
    check(user && system, "Missing private data principal");
}
} // namespace
int main()
{
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        auto file = root / "edge.db";
        Database db(file);
        restricted(root);
        restricted(file);
        DatabaseTestAccess::run(db, [&](sqlite3 *d) {
            detail::sql(d, "BEGIN IMMEDIATE; INSERT INTO attendance_outbox(event_id,payload,occurred_ms) "
                           "VALUES('acl-test','{}',1)");
            auto journal = file;
            journal += L"-journal";
            check(std::filesystem::exists(journal), "Expected a live rollback journal");
            restricted(journal, true);
            detail::sql(d, "ROLLBACK");
        });
        protect_token(root / "device.dpapi", "local-test-token");
        restricted(root / "device.dpapi");
        protect_token(root / "device.dpapi", "rotated-test-token");
        restricted(root / "device.dpapi");
        check(unprotect_token(root / "device.dpapi") == "rotated-test-token", "Credential rotation failed");
        check(!std::filesystem::exists(root / "device.dpapi.new"), "Temporary credential retained after success");
        rejects([&] { prepare_private_file(std::filesystem::current_path() / "unsafe.db"); });
        auto source = root / "original.bin", link = root / "linked.bin";
        {
            std::ofstream f(source);
            f << "unchanged";
        }
        check(CreateHardLinkW(link.c_str(), source.c_str(), nullptr), "Create hard link fixture failed");
        rejects([&] { prepare_private_file(link); });
        auto untouched = root / "reject-before-acl";
        std::filesystem::create_directory(untouched);
        auto second_link = untouched / "linked.bin";
        check(CreateHardLinkW(second_link.c_str(), source.c_str(), nullptr), "Create second hard link failed");
        auto parent_acl = acl_text(untouched), source_acl = acl_text(source);
        rejects([&] { prepare_private_file(second_link); });
        check(acl_text(untouched) == parent_acl && acl_text(source) == source_acl,
              "Rejected hard link changed parent or source permissions");
        std::cout << "Data directory/database/token ACLs, rotation and unsafe paths: PASS\n";
        return 0;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
