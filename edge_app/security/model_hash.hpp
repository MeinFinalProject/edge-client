#pragma once
#include <filesystem>
#include <string>
namespace edge_app
{
std::string sha256_file(std::filesystem::path const &file);
}
