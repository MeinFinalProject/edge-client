#pragma once
#include <filesystem>
namespace edge_app
{
// Dedicated local directory only. Restricts directory/file DACL to current user
// and SYSTEM. Rejects reparse paths, network paths and multiply linked files.
void prepare_private_file(std::filesystem::path const &file);
} // namespace edge_app
