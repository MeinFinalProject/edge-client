#include "support/edge_fixtures.hpp"
using namespace edge_test;
int main()
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        auto root = std::filesystem::current_path() / "edge-test-data" / uuid_v7(utc_ms());
        std::filesystem::create_directories(root);
        protect_token(root / "token.dpapi", "local-test-token");
        check(unprotect_token(root / "token.dpapi") == "local-test-token", "DPAPI round trip failed");
        {
            std::ifstream f(root / "token.dpapi", std::ios::binary);
            std::string data{std::istreambuf_iterator<char>(f), {}};
            check(data.find("local-test-token") == std::string::npos, "Token plaintext persisted");
        }
        {
            std::ofstream f(root / "bad.dpapi");
            f << "invalid";
        }
        rejects([&] { (void)unprotect_token(root / "bad.dpapi"); });
        std::cout << "DPAPI credential PASS\n";
        return 0;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "WinRT error " << std::hex << e.code().value << '\n';
        return 1;
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
