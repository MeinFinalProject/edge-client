// ============================================================================
// model_inspect — Print ONNX model metadata via WinML.
//
// Replaces edge/tools/inspect_onnx.py. Loads a model with the WinML
// LearningModel API and prints input/output feature descriptors including
// tensor shapes, element types, and known provenance for the models used by
// this project.
//
// Usage:
//   model_inspect.exe <path-to-onnx-file>
//   model_inspect.exe --all
//
// The --all flag inspects the active project artifacts under models/
// (searching upward). Historical files remain inspectable by explicit path.
//
// Model provenance catalog:
//   SCRFD det_500m / det_10g  — InsightFace (github.com/deepinsight/insightface)
//   ArcFace w600k_r50         — InsightFace (github.com/deepinsight/insightface)
//   liveness addon           — InsightFace Model Addons
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "liveness/model_contract.hpp"

#include <Windows.h>

#include <winrt/base.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Storage.h>

namespace {

namespace fs = std::filesystem;

using winrt::Windows::AI::MachineLearning::LearningModel;
using winrt::Windows::AI::MachineLearning::TensorKind;
using winrt::Windows::AI::MachineLearning::ILearningModelFeatureDescriptor;

// ── Known model provenance ──────────────────────────────────────────────────

struct Provenance {
    char const* filename_pattern;
    char const* project_role;
    char const* upstream;
    char const* upstream_url;
    char const* license_note;
};

constexpr Provenance kKnownModels[] = {
    {
        "det_500m",
        "Production SCRFD face detector (500M variant)",
        "InsightFace buffalo_sc",
        "https://github.com/deepinsight/insightface",
        "Non-commercial research only (code MIT, model weights restricted)"
    },
    {
        "det_10g",
        "Alternate SCRFD face detector (10G variant)",
        "InsightFace buffalo_l",
        "https://github.com/deepinsight/insightface",
        "Non-commercial research only (code MIT, model weights restricted)"
    },
    {
        "w600k_r50",
        "ArcFace face recognition (512-dim embedding)",
        "InsightFace buffalo_l",
        "https://github.com/deepinsight/insightface",
        "Non-commercial research only (code MIT, model weights restricted)"
    },
    {
        "liveness",
        "InsightFace Liveness (presentation attack detection)",
        "InsightFace Model Addons",
        "https://github.com/deepinsight/insightface",
        "Non-commercial research only (code MIT, model weights restricted)"
    },
};

Provenance const* find_provenance(std::string const& stem) {
    for (auto const& p : kKnownModels) {
        if (stem.find(p.filename_pattern) != std::string::npos) {
            return &p;
        }
    }
    return nullptr;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

std::string to_utf8(std::wstring const& ws) {
    if (ws.empty()) return {};
    const int n = WideCharToMultiByte(
        CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()),
        nullptr, 0, nullptr, nullptr
    );
    if (n <= 0) return "(encoding error)";
    std::string utf8(static_cast<std::size_t>(n), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()),
        utf8.data(), n, nullptr, nullptr
    );
    return utf8;
}

std::string to_utf8(winrt::hstring const& hs) {
    return to_utf8(std::wstring{hs.c_str(), hs.size()});
}

char const* tensor_kind_name(TensorKind kind) {
    switch (kind) {
        case TensorKind::Undefined: return "undefined";
        case TensorKind::Float:     return "float32";
        case TensorKind::UInt8:     return "uint8";
        case TensorKind::Int8:      return "int8";
        case TensorKind::UInt16:    return "uint16";
        case TensorKind::Int16:     return "int16";
        case TensorKind::Int32:     return "int32";
        case TensorKind::Int64:     return "int64";
        case TensorKind::String:    return "string";
        case TensorKind::Boolean:   return "bool";
        case TensorKind::Float16:   return "float16";
        case TensorKind::Double:    return "float64";
        case TensorKind::UInt32:    return "uint32";
        case TensorKind::UInt64:    return "uint64";
        case TensorKind::Complex64: return "complex64";
        case TensorKind::Complex128:return "complex128";
        default:                    return "unknown";
    }
}

fs::path executable_directory() {
    std::wstring buf(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0 || n >= buf.size()) {
        throw std::runtime_error("GetModuleFileNameW failed");
    }
    buf.resize(n);
    return fs::path{buf}.parent_path();
}

fs::path find_upwards(fs::path root, fs::path const& rel) {
    for (int i = 0; i < 12; ++i) {
        const auto candidate = root / rel;
        if (fs::exists(candidate)) {
            return fs::absolute(candidate).lexically_normal();
        }
        const auto parent = root.parent_path();
        if (parent == root) break;
        root = parent;
    }
    return {};
}

void print_separator() {
    std::cout << std::string(72, '=') << "\n";
}

void print_dash() {
    std::cout << std::string(72, '-') << "\n";
}

// ── Inspect a single model ──────────────────────────────────────────────────

bool inspect_model(fs::path const& model_path) {
    std::cout << "\n";
    print_separator();
    std::cout << "MODEL INSPECTION\n";
    print_separator();

    if (!fs::exists(model_path)) {
        std::cerr << "ERROR: file not found: " << model_path.string() << "\n";
        return false;
    }

    const auto size_bytes = fs::file_size(model_path);
    const double size_mb = static_cast<double>(size_bytes) / (1024.0 * 1024.0);

    std::cout << "File       : " << model_path.string() << "\n";
    std::cout << "Size       : " << std::fixed << std::setprecision(2)
              << size_mb << " MiB (" << size_bytes << " bytes)\n";

    // ── Provenance ──────────────────────────────────────────────────────
    const auto stem = model_path.stem().string();
    const auto* prov = find_provenance(stem);
    if (prov) {
        std::cout << "\n";
        std::cout << "PROVENANCE\n";
        print_dash();
        std::cout << "Role       : " << prov->project_role << "\n";
        std::cout << "Upstream   : " << prov->upstream << "\n";
        std::cout << "URL        : " << prov->upstream_url << "\n";
        std::cout << "License    : " << prov->license_note << "\n";
    } else {
        std::cout << "Provenance : unknown model (not in catalog)\n";
    }

    // ── WinML load ──────────────────────────────────────────────────────
    try {
        const auto model = LearningModel::LoadFromFilePath(model_path.wstring());

        // Model-level metadata
        std::cout << "\n";
        std::cout << "MODEL METADATA\n";
        print_dash();

        const auto author = to_utf8(model.Author());
        const auto name   = to_utf8(model.Name());
        const auto domain = to_utf8(model.Domain());
        const auto desc   = to_utf8(model.Description());

        if (!author.empty()) std::cout << "Author     : " << author << "\n";
        if (!name.empty())   std::cout << "Name       : " << name << "\n";
        if (!domain.empty()) std::cout << "Domain     : " << domain << "\n";
        if (!desc.empty())   std::cout << "Description: " << desc << "\n";

        // Inputs
        const auto inputs = model.InputFeatures();
        std::cout << "\n";
        std::cout << "INPUTS (" << inputs.Size() << ")\n";
        print_dash();

        for (uint32_t i = 0; i < inputs.Size(); ++i) {
            const auto& feat = inputs.GetAt(i);
            std::cout << "  [" << i << "] " << to_utf8(feat.Name()) << "\n";

            auto kind = feat.Kind();
            char const* kind_str = "unknown";
            using K = winrt::Windows::AI::MachineLearning::LearningModelFeatureKind;
            if (kind == K::Tensor)   kind_str = "Tensor";
            else if (kind == K::Sequence) kind_str = "Sequence";
            else if (kind == K::Map)      kind_str = "Map";
            else if (kind == K::Image)    kind_str = "Image";
            std::cout << "      kind : " << kind_str << "\n";

            // Try to get tensor shape via description string
            // WinML ILearningModelFeatureDescriptor doesn't directly expose shape,
            // but we can get it from the TensorFeatureDescriptor if it's a tensor.
            auto tensor = feat.try_as<winrt::Windows::AI::MachineLearning::TensorFeatureDescriptor>();
            if (tensor) {
                std::cout << "      type : " << tensor_kind_name(tensor.TensorKind()) << "\n";
                auto shape = tensor.Shape();
                std::cout << "      shape: [";
                for (uint32_t j = 0; j < shape.Size(); ++j) {
                    if (j > 0) std::cout << ", ";
                    const auto dim = shape.GetAt(j);
                    if (dim < 0) {
                        std::cout << "?";
                    } else {
                        std::cout << dim;
                    }
                }
                std::cout << "]\n";
            }
        }

        // Outputs
        const auto outputs = model.OutputFeatures();
        std::cout << "\n";
        std::cout << "OUTPUTS (" << outputs.Size() << ")\n";
        print_dash();

        for (uint32_t i = 0; i < outputs.Size(); ++i) {
            const auto& feat = outputs.GetAt(i);
            std::cout << "  [" << i << "] " << to_utf8(feat.Name()) << "\n";

            auto kind = feat.Kind();
            char const* kind_str = "unknown";
            using K = winrt::Windows::AI::MachineLearning::LearningModelFeatureKind;
            if (kind == K::Tensor)   kind_str = "Tensor";
            else if (kind == K::Sequence) kind_str = "Sequence";
            else if (kind == K::Map)      kind_str = "Map";
            else if (kind == K::Image)    kind_str = "Image";
            std::cout << "      kind : " << kind_str << "\n";

            auto tensor = feat.try_as<winrt::Windows::AI::MachineLearning::TensorFeatureDescriptor>();
            if (tensor) {
                std::cout << "      type : " << tensor_kind_name(tensor.TensorKind()) << "\n";
                auto shape = tensor.Shape();
                std::cout << "      shape: [";
                for (uint32_t j = 0; j < shape.Size(); ++j) {
                    if (j > 0) std::cout << ", ";
                    const auto dim = shape.GetAt(j);
                    if (dim < 0) {
                        std::cout << "?";
                    } else {
                        std::cout << dim;
                    }
                }
                std::cout << "]\n";
            }
        }

        std::cout << "\nWinML load : PASS\n";

    } catch (winrt::hresult_error const& ex) {
        std::cerr << "WinML load FAILED: " << to_utf8(ex.message()) << "\n";
        return false;
    } catch (std::exception const& ex) {
        std::cerr << "Load FAILED: " << ex.what() << "\n";
        return false;
    }

    return true;
}

// ── Locate the active artifacts (same set as models/manifest.yaml) ───────────

std::vector<fs::path> discover_models() {
    const auto models_dir = find_upwards(executable_directory(), "models");
    if (models_dir.empty()) {
        std::cerr << "ERROR: could not find models/ directory.\n";
        return {};
    }

    std::vector<fs::path> result;
    for (auto const* relative : {
        "insightface/buffalo_sc/det_500m.onnx",
        "insightface/buffalo_l/det_10g.onnx",
        "insightface/buffalo_l/w600k_r50.onnx",
        vision_runtime::liveness::kModelRelativePath
    }) {
        // Include missing paths too: inspect_model must report absent artifacts.
        result.push_back(models_dir / relative);
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace

int main(int argc, char* argv[]) {
    winrt::init_apartment();

    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  model_inspect.exe <path-to-onnx-file>\n"
                  << "  model_inspect.exe --all\n";
        return 1;
    }

    const std::string arg1 = argv[1];
    bool all_ok = true;

    if (arg1 == "--all") {
        const auto models = discover_models();
        if (models.empty()) {
            std::cerr << "No .onnx files found.\n";
            return 1;
        }
        std::cout << "Inspecting " << models.size() << " active project model(s).\n";
        for (auto const& path : models) {
            if (!inspect_model(path)) {
                all_ok = false;
            }
        }
    } else {
        all_ok = inspect_model(fs::path{arg1});
    }

    std::cout << "\n";
    print_separator();
    std::cout << "DONE" << (all_ok ? "" : " (with errors)") << "\n";
    print_separator();

    return all_ok ? 0 : 1;
}
