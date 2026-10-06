#include "storage/database.hpp"
#include "storage/outbox_repository.hpp"

#include <Windows.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path default_database_path() {
    std::wstring local_app_data(32768, L'\0');
    const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", local_app_data.data(),
                                            static_cast<DWORD>(local_app_data.size()));
    if (n == 0 || n >= local_app_data.size()) {
        throw std::runtime_error("LOCALAPPDATA is unavailable");
    }
    local_app_data.resize(n);
    return std::filesystem::path{local_app_data} / L"TA" / L"ta_demo" / L"attendance.db";
}

std::filesystem::path parse_database_path(int argc, char** argv) {
    if (argc == 1) {
        return default_database_path();
    }
    if (argc == 3 && std::string{argv[1]} == "--db") {
        return std::filesystem::absolute(argv[2]).lexically_normal();
    }
    std::cerr << "Usage: check_ta_demo.exe [--db <path-to-attendance.db>]\n";
    std::exit(1);
}

void print_separator() {
    std::cout << std::string(72, '=') << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto path = parse_database_path(argc, argv);
        if (!std::filesystem::exists(path)) {
            std::cerr << "Database not found: " << path.string() << '\n';
            std::cerr << "Run ta_demo first so it can persist an attendance event.\n";
            return 1;
        }

        edge_app::Database database{path};
        edge_app::OutboxRepository outbox{database};
        const auto rows = outbox.due(std::numeric_limits<std::int64_t>::max(), 100);
        const auto pending = outbox.pending_count();

        print_separator();
        std::cout << "TA DEMO DATABASE\n";
        print_separator();
        std::cout << "Path           : " << path.string() << '\n';
        std::cout << "SQLite version : " << database.sqlite_version() << '\n';
        std::cout << "Pending outbox : " << pending << '\n';
        std::cout << "Dead letter    : " << outbox.dead_count() << '\n';
        std::cout << "Shown events   : " << rows.size();
        if (pending > rows.size()) {
            std::cout << " (first 100)";
        }
        std::cout << "\n\n";

        if (rows.empty()) {
            std::cout << "No pending attendance events.\n";
            return 0;
        }

        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            std::cout << "[" << i + 1 << "] event_id    : " << row.id << '\n';
            std::cout << "    occurred_ms : " << row.occurred_ms << '\n';
            std::cout << "    attempts    : " << row.attempts << '\n';
            std::cout << "    next_ms     : " << row.next_attempt_ms << '\n';
            std::cout << "    payload     : " << row.payload << "\n\n";
        }
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
