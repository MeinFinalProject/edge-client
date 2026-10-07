#include <Windows.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

std::string sibling_demo() {
    std::wstring buffer(32768, L'\0');
    const DWORD length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        throw std::runtime_error("GetModuleFileNameW failed");
    }
    buffer.resize(length);
    std::wstring path = buffer.substr(0, buffer.find_last_of(L"\\/") + 1);
    path += L"ta_demo.exe";
    const int size = WideCharToMultiByte(CP_UTF8, 0, path.data(),
                                         static_cast<int>(path.size()), nullptr, 0, nullptr,
                                         nullptr);
    if (size <= 0) throw std::runtime_error("WideCharToMultiByte failed");
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, path.data(), static_cast<int>(path.size()), result.data(),
                        size, nullptr, nullptr);
    return result;
}

struct BenchmarkResult {
    int exit_code = 1;
    double process_wall_s = 0.0;
    int frames = 0;
    double elapsed_s = 0.0;
    double avg_fps = 0.0;
};

BenchmarkResult run_mode(std::string const& demo, char const* mode, int seconds) {
    _putenv_s("EDGE_ZERO_COPY", mode);
    const std::string output_path =
        "zero_copy_benchmark_" +
        std::to_string(GetCurrentProcessId()) + "_" + (mode[0] == '0' ? "copy" : "zero") +
        ".log";
    const std::string command = "cmd.exe /S /C \"\"" + demo + "\" --benchmark-seconds " +
                                std::to_string(seconds) + " --no-pad > \"" + output_path +
                                "\" 2>&1\"";
    const auto begin = Clock::now();
    BenchmarkResult result;
    result.exit_code = std::system(command.c_str());
    result.process_wall_s =
        std::chrono::duration<double>(Clock::now() - begin).count();

    std::ifstream output(output_path);
    const std::string text((std::istreambuf_iterator<char>(output)),
                           std::istreambuf_iterator<char>());
    std::cout << text;
    std::smatch match;
    const std::regex benchmark(
        R"(\[BENCHMARK\]\s+frames=(\d+)\s+elapsed_s=([0-9.]+)\s+avg_fps=([0-9.]+))");
    if (std::regex_search(text, match, benchmark)) {
        result.frames = std::stoi(match[1].str());
        result.elapsed_s = std::stod(match[2].str());
        result.avg_fps = std::stod(match[3].str());
    }
    DeleteFileA(output_path.c_str());

    std::cout << std::fixed << std::setprecision(3)
              << "[RESULT] mode=" << (mode[0] == '0' ? "CopyResource" : "zero-copy")
              << " process_wall_s=" << result.process_wall_s
              << " exit_code=" << result.exit_code << "\n";
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int seconds = argc > 1 ? std::stoi(argv[1]) : 20;
        if (seconds < 1) {
            throw std::runtime_error("duration must be >= 1 second");
        }

        const std::string demo = sibling_demo();
        std::cout << "Zero-copy benchmark: each mode runs ta_demo for " << seconds
                  << " seconds after its normal warm-up.\n\n";

        const BenchmarkResult zero_copy_result = run_mode(demo, "", seconds);
        std::cout << "\n--- switching to CopyResource fallback ---\n\n";
        const BenchmarkResult copy_result = run_mode(demo, "0", seconds);

        std::cout << "\n[SUMMARY] zero-copy avg_fps=" << zero_copy_result.avg_fps
                  << " CopyResource avg_fps=" << copy_result.avg_fps
                  << " delta_fps=" << (zero_copy_result.avg_fps - copy_result.avg_fps)
                  << " (positive favors zero-copy)\n"
                  << "[SUMMARY] zero-copy process_wall_s=" << zero_copy_result.process_wall_s
                  << " CopyResource process_wall_s=" << copy_result.process_wall_s
                  << " delta_s=" << (zero_copy_result.process_wall_s - copy_result.process_wall_s)
                  << " (negative favors zero-copy)\n";
        return zero_copy_result.exit_code == 0 && copy_result.exit_code == 0 ? 0 : 1;
    } catch (std::exception const& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
