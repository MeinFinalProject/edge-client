#pragma once
#include "pipeline/biometric_attendance_runtime.hpp"
#include <filesystem>
#include <string>

namespace edge_app
{
struct ApplicationConfig
{
    std::string device_id, base_url;
    std::filesystem::path database, credential_file;
    vision_runtime::pipeline::BiometricAttendanceConfig vision;
    bool allow_loopback_http = false;
};
ApplicationConfig load_config(std::filesystem::path const &file);
std::string read_document(std::filesystem::path const &file);
// Verifies local files/selected PAD checksum; returns the actual ArcFace hash.
std::string verify_model_files(ApplicationConfig const &config);
} // namespace edge_app
