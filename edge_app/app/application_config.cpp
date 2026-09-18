#include "app/application_config.hpp"
#include "pipeline/attendance_processor.hpp"
#include "security/model_hash.hpp"
#include <fstream>
#include <winrt/Windows.Data.Json.h>
namespace edge_app
{
std::string read_document(std::filesystem::path const &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read input file");
    if (std::filesystem::file_size(path) > 32 * 1024 * 1024)
        throw std::runtime_error("Input exceeds 32 MiB");
    return {std::istreambuf_iterator<char>(f), {}};
}
ApplicationConfig load_config(std::filesystem::path const &config_file)
{
    using winrt::Windows::Data::Json::JsonObject;
    auto json = JsonObject::Parse(winrt::to_hstring(read_document(config_file)));
    if (json.GetNamedNumber(L"schema_version") != 1)
        throw std::invalid_argument("Unsupported config schema");
    auto string = [&](wchar_t const *name) { return winrt::to_string(json.GetNamedString(name)); };
    auto path = [&](wchar_t const *name) {
        return std::filesystem::absolute(config_file).parent_path() / std::filesystem::u8path(string(name));
    };
    auto device = string(L"device_id"), url = string(L"base_url");
    if (device.empty() || device.size() > 256 || device.find_first_of("\r\n") != std::string::npos)
        throw std::invalid_argument("Invalid device ID");
    vision_runtime::pipeline::BiometricAttendanceConfig cfg;
    cfg.tracking.scrfd.model_path = path(L"scrfd_model");
    cfg.tracking.scrfd.score_threshold = .1F;
    cfg.arcface.model_path = path(L"arcface_model");
    cfg.pad.model_path = path(L"pad_model");
    cfg.recognition_similarity_threshold = static_cast<float>(json.GetNamedNumber(L"recognition_threshold"));
    cfg.recognition_min_margin = static_cast<float>(json.GetNamedNumber(L"recognition_min_margin", 0));
    cfg.calibration_id = winrt::to_string(json.GetNamedString(L"calibration_id", L""));
    cfg.allow_uncalibrated_thresholds = json.GetNamedBoolean(L"allow_uncalibrated_thresholds", false);
    vision_runtime::pipeline::validate_attendance_config(cfg);
    ApplicationConfig result;
    result.device_id = std::move(device);
    result.base_url = std::move(url);
    result.database = path(L"database");
    result.credential_file = path(L"credential_file");
    result.vision = std::move(cfg);
    result.allow_loopback_http = json.GetNamedBoolean(L"allow_loopback_http", false);
    return result;
}
std::string verify_model_files(ApplicationConfig const &config)
{
    auto const &cfg = config.vision;
    vision_runtime::pipeline::validate_attendance_config(cfg);
    for (auto const &file : {cfg.tracking.scrfd.model_path, cfg.arcface.model_path, cfg.pad.model_path})
        if (!std::filesystem::is_regular_file(file))
            throw std::runtime_error("Configured model does not exist");
    auto hash = edge_app::sha256_file(cfg.arcface.model_path);
    if (edge_app::sha256_file(cfg.pad.model_path) != vision_runtime::liveness::kModelSha256)
        throw std::runtime_error("PAD model does not match the selected preprocessing/model contract");
    return hash;
}
} // namespace edge_app
