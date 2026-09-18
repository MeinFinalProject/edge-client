#include "app/edge_application.hpp"
#include "security/device_credential.hpp"
#include <Windows.h>
#include <atomic>
#include <iostream>
#include <winrt/base.h>

namespace
{
std::atomic_bool stopping = false;
BOOL WINAPI stop_handler(DWORD signal)
{
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT)
    {
        stopping = true;
        return TRUE;
    }
    return FALSE;
}
std::string prompt_token()
{
    auto input = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode = 0;
    if (!GetConsoleMode(input, &mode))
        throw std::runtime_error("Token provisioning requires an interactive console");
    struct Restore
    {
        HANDLE h;
        DWORD mode;
        ~Restore()
        {
            SetConsoleMode(h, mode);
        }
    } restore{input, mode};
    if (!SetConsoleMode(input, mode & ~ENABLE_ECHO_INPUT))
        throw std::runtime_error("Cannot hide token input");
    std::cout << "Device token (hidden): " << std::flush;
    std::string token;
    std::getline(std::cin, token);
    std::cout << '\n';
    return token;
}
} // namespace
int main(int argc, char **argv)
{
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
    try
    {
        std::filesystem::path config_file = "edge-config.local.json", import_file;
        bool validate_only = false;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            auto value = [&]() {
                if (++i >= argc)
                    throw std::invalid_argument("Missing option value");
                return std::filesystem::path(argv[i]);
            };
            if (arg == "--config")
                config_file = value();
            else if (arg == "--import-gallery")
                import_file = value();
            else if (arg == "--validate-only")
                validate_only = true;
            else if (arg == "--provision-token")
            {
                auto file = value();
                auto token = prompt_token();
                edge_app::protect_token(file, token);
                SecureZeroMemory(token.data(), token.size());
                std::cout << "Credential protected for this Windows user.\n";
                return 0;
            }
            else if (arg == "--help")
            {
                std::cout << "edge_client [--config file] [--validate-only | --import-gallery file]\n"
                             "edge_client --provision-token data/device.dpapi\n"
                             "Paths in config are relative to its directory. Ctrl+C stops capture.\n";
                return 0;
            }
            else
                throw std::invalid_argument("Unknown option: " + arg);
        }
        edge_app::EdgeApplication application(edge_app::load_config(config_file));
        if (!import_file.empty())
        {
            application.import_gallery(import_file);
            std::cout << "Gallery committed.\n";
        }
        else if (validate_only)
        {
            application.validate(std::cout);
        }
        else
        {
            struct ConsoleHandler
            {
                ConsoleHandler()
                {
                    SetConsoleCtrlHandler(stop_handler, TRUE);
                }
                ~ConsoleHandler()
                {
                    SetConsoleCtrlHandler(stop_handler, FALSE);
                }
            } handler;
            application.run(stopping, std::cout);
        }
        return 0;
    }
    catch (winrt::hresult_error const &e)
    {
        std::cerr << "Windows operation failed (HRESULT " << std::hex << static_cast<unsigned>(e.code().value)
                  << ").\n";
        return 1;
    }
    catch (std::exception const &e)
    {
        std::cerr << "edge_client: " << e.what() << '\n';
        return 1;
    }
}
