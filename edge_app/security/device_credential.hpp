#pragma once
#include <filesystem>
#include <string>
namespace edge_app
{
void protect_token(std::filesystem::path const &file, std::string const &token);
std::string unprotect_token(std::filesystem::path const &file);
void validate_token(std::string const &token);
} // namespace edge_app
