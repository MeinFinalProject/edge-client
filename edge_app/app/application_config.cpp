#include "app/application_config.hpp"
#include "pipeline/attendance_processor.hpp"
#include "security/model_hash.hpp"
#include <algorithm>
#include <fstream>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Foundation.Collections.h>
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
    using winrt::Windows::Data::Json::JsonValueType;
    for (auto key : {L"mode", L"calibration_id"})
        if (json.HasKey(key) && json.GetNamedValue(key).ValueType() != JsonValueType::String)
            throw std::invalid_argument("mode and calibration_id must be JSON strings");
    for (auto key : {L"allow_uncalibrated_thresholds", L"allow_loopback_http"})
        if (json.HasKey(key) && json.GetNamedValue(key).ValueType() != JsonValueType::Boolean)
            throw std::invalid_argument("Configuration flags must be JSON booleans, not strings or numbers");
    auto mode = json.GetNamedString(L"mode", L"development");
    if (mode != L"development" && mode != L"deployment")
        throw std::invalid_argument("mode must be development or deployment");
    const bool deployment = mode == L"deployment";
    if (deployment)
    {
        if (json.GetNamedBoolean(L"allow_uncalibrated_thresholds", false) ||
            json.GetNamedBoolean(L"allow_loopback_http", false))
            throw std::invalid_argument("Deployment forbids uncalibrated thresholds and plain loopback HTTP");
        for (auto key : {L"recognition_threshold", L"recognition_min_margin"})
            if (!json.HasKey(key) ||
                json.GetNamedValue(key).ValueType() != winrt::Windows::Data::Json::JsonValueType::Number)
                throw std::invalid_argument(
                    "Deployment requires calibrated numeric recognition_threshold and recognition_min_margin");
    }
    auto string = [&](wchar_t const *name) { return winrt::to_string(json.GetNamedString(name)); };
    auto path = [&](wchar_t const *name) {
        return std::filesystem::absolute(config_file).parent_path() / std::filesystem::u8path(string(name));
    };
    auto device = string(L"device_id"), url = string(L"base_url");
    if (device.empty() || device.size() > 256 || device.find_first_of("\r\n\0", 0, 3) != std::string::npos)
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
    if (_wcsicmp(result.database.lexically_normal().c_str(), result.credential_file.lexically_normal().c_str()) == 0)
        throw std::invalid_argument("Database and credential must use different files");
    result.vision = std::move(cfg);
    result.allow_loopback_http = json.GetNamedBoolean(L"allow_loopback_http", false);
    result.deployment = deployment;
    if (deployment)
    {
        for (auto key : {L"calibration_report", L"calibration_report_sha256", L"calibration_model_sha256"})
            if (!json.HasKey(key) ||
                json.GetNamedValue(key).ValueType() != winrt::Windows::Data::Json::JsonValueType::String ||
                json.GetNamedString(key).empty())
                throw std::invalid_argument(
                    "Deployment requires a calibration report, report SHA-256 and calibrated model SHA-256");
        result.calibration_report = path(L"calibration_report");
        result.calibration_report_sha256 = string(L"calibration_report_sha256");
        result.calibration_model_sha256 = string(L"calibration_model_sha256");
    }
    return result;
}
void verify_deployment_evidence(ApplicationConfig const &config, std::string const &model_hash)
{
    if (!config.deployment)
        return;
    if (config.vision.allow_uncalibrated_thresholds || config.allow_loopback_http ||
        config.vision.calibration_id.empty())
        throw std::invalid_argument("Invalid deployment configuration");
    auto digest = [](std::string const &s) {
        return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
    };
    if (!digest(config.calibration_report_sha256) || !digest(config.calibration_model_sha256) ||
        config.calibration_model_sha256 != model_hash)
        throw std::invalid_argument("Calibration evidence requires valid hashes matching the active ArcFace model");
    if (!std::filesystem::is_regular_file(config.calibration_report) ||
        std::filesystem::file_size(config.calibration_report) == 0)
        throw std::invalid_argument("Calibration report is missing or empty");
    if (sha256_file(config.calibration_report) != config.calibration_report_sha256)
        throw std::invalid_argument("Calibration report SHA-256 mismatch");
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
    verify_deployment_evidence(config, hash);
    return hash;
}
} // namespace edge_app
