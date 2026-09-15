#pragma once

// Small, CPU-only labelled capture rules. No camera/model dependency: these rules can
// be checked with synthetic inputs before any labelled camera data is collected.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace capture {

inline void validate_label(std::string const& label, std::string const& scenario) {
    if (label != "real" && label != "spoof")
        throw std::invalid_argument("--label must be real or spoof (operator ground truth)");
    if (scenario != "live" && scenario != "photo" && scenario != "display" && scenario != "replay")
        throw std::invalid_argument("--scenario must be live, photo, display, or replay");
    if ((label == "real") != (scenario == "live"))
        throw std::invalid_argument("real requires live; spoof requires photo/display/replay");
}

inline bool identifier(std::string const& s) {
    if (s.empty() || s == "." || s == "..") return false;
    for (unsigned char c : s)
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-')) return false;
    return true;
}

template<class T> std::string number(T value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << value;
    return out.str();
}

inline std::string csv_line(std::vector<std::string> const& fields) {
    std::string out;
    for (auto const& field : fields) {
        if (!out.empty()) out += ',';
        // Quote every cell, including empty cells; escape quotes per CSV rules.
        out += '"';
        for (char c : field) { if (c == '"') out += '"'; out += c; }
        out += '"';
    }
    return out + "\r\n";
}

} // namespace capture
