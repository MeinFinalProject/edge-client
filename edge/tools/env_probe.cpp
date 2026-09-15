// ============================================================================
// env_probe — Native environment verification.
//
// Replaces edge/tools/verify_env.py.
// Checks that the DirectX/DXGI runtime can enumerate adapters, prints GPU
// memory available, and confirms that the WinRT MediaCapture camera subsystem
// is responsive. No model loading or camera capture occurs.
// ============================================================================

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Media.Capture.Frames.h>

namespace {

using Microsoft::WRL::ComPtr;
using winrt::Windows::Media::Capture::Frames::MediaFrameSourceGroup;

// ── DXGI adapter enumeration ────────────────────────────────────────────────

struct AdapterInfo {
    std::wstring description;
    std::uint64_t dedicated_video_mb = 0;
    std::uint64_t shared_system_mb  = 0;
    std::uint32_t vendor_id         = 0;
    std::uint32_t device_id         = 0;
    bool          is_software       = false;
};

std::vector<AdapterInfo> enumerate_adapters() {
    ComPtr<IDXGIFactory4> factory;
    const HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        std::cerr << "CreateDXGIFactory1 failed: 0x"
                  << std::hex << hr << std::dec << "\n";
        return {};
    }

    std::vector<AdapterInfo> result;
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc))) {
            AdapterInfo info;
            info.description       = desc.Description;
            info.dedicated_video_mb = desc.DedicatedVideoMemory / (1024 * 1024);
            info.shared_system_mb  = desc.SharedSystemMemory   / (1024 * 1024);
            info.vendor_id         = desc.VendorId;
            info.device_id         = desc.DeviceId;
            info.is_software       = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
            result.push_back(info);
        }
        adapter.Reset();
    }
    return result;
}

// ── Camera source group enumeration ─────────────────────────────────────────

struct CameraGroupInfo {
    std::wstring id;
    std::wstring display_name;
    std::uint32_t source_count = 0;
};

std::vector<CameraGroupInfo> enumerate_cameras() {
    auto groups = MediaFrameSourceGroup::FindAllAsync().get();
    std::vector<CameraGroupInfo> result;
    for (auto const& g : groups) {
        CameraGroupInfo info;
        info.id           = g.Id().c_str();
        info.display_name = g.DisplayName().c_str();
        info.source_count = g.SourceInfos().Size();
        result.push_back(info);
    }
    return result;
}

// ── Print helpers ───────────────────────────────────────────────────────────

void print_wide(std::wstring const& ws) {
    const int needed = WideCharToMultiByte(
        CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()),
        nullptr, 0, nullptr, nullptr
    );
    if (needed <= 0) {
        std::cout << "(encoding error)";
        return;
    }
    std::string utf8(static_cast<std::size_t>(needed), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, ws.c_str(), static_cast<int>(ws.size()),
        utf8.data(), needed, nullptr, nullptr
    );
    std::cout << utf8;
}

void print_separator() {
    std::cout << std::string(72, '=') << "\n";
}

} // namespace

int main() {
    winrt::init_apartment();

    print_separator();
    std::cout << "TA ENVIRONMENT PROBE (NATIVE)\n";
    print_separator();

    // ── 1. DXGI adapters ────────────────────────────────────────────────────

    std::cout << "\n";
    std::cout << "DXGI ADAPTERS\n";
    std::cout << std::string(72, '-') << "\n";

    const auto adapters = enumerate_adapters();
    if (adapters.empty()) {
        std::cout << "WARNING: No DXGI adapters found.\n";
    }

    bool found_hardware_gpu = false;
    for (std::size_t i = 0; i < adapters.size(); ++i) {
        const auto& a = adapters[i];
        std::cout << "[" << i << "] ";
        print_wide(a.description);
        std::cout << "\n";
        std::cout << "    Vendor/Device  : 0x" << std::hex << a.vendor_id
                  << " / 0x" << a.device_id << std::dec << "\n";
        std::cout << "    Dedicated VRAM : " << a.dedicated_video_mb << " MB\n";
        std::cout << "    Shared Memory  : " << a.shared_system_mb  << " MB\n";
        std::cout << "    Type           : " << (a.is_software ? "SOFTWARE" : "HARDWARE") << "\n";
        if (!a.is_software) {
            found_hardware_gpu = true;
        }
    }

    std::cout << "\nHardware GPU  : " << (found_hardware_gpu ? "PASS" : "WARNING — no hardware adapter") << "\n";

    // ── 2. Camera subsystem ─────────────────────────────────────────────────

    std::cout << "\n";
    std::cout << "CAMERA SOURCE GROUPS\n";
    std::cout << std::string(72, '-') << "\n";

    const auto cameras = enumerate_cameras();
    if (cameras.empty()) {
        std::cout << "WARNING: No camera source groups found.\n";
    }

    for (std::size_t i = 0; i < cameras.size(); ++i) {
        const auto& c = cameras[i];
        std::cout << "[" << i << "] ";
        print_wide(c.display_name);
        std::cout << " (" << c.source_count << " sources)\n";
    }

    std::cout << "\nCamera groups : " << cameras.size() << "\n";

    // ── 3. OS version ───────────────────────────────────────────────────────

    std::cout << "\n";
    std::cout << "SYSTEM\n";
    std::cout << std::string(72, '-') << "\n";

    OSVERSIONINFOEXW osvi{};
    osvi.dwOSVersionInfoSize = sizeof(osvi);
    // RtlGetVersion is the accurate way (not affected by app manifests).
    using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
    auto* ntdll = GetModuleHandleW(L"ntdll.dll");
    if (ntdll) {
        auto rtl_get_version = reinterpret_cast<RtlGetVersionFn>(
            GetProcAddress(ntdll, "RtlGetVersion")
        );
        if (rtl_get_version) {
            rtl_get_version(reinterpret_cast<PRTL_OSVERSIONINFOW>(&osvi));
            std::cout << "Windows       : "
                      << osvi.dwMajorVersion << "."
                      << osvi.dwMinorVersion << "."
                      << osvi.dwBuildNumber << "\n";
        }
    }

    // ── Summary ─────────────────────────────────────────────────────────────

    std::cout << "\n";
    print_separator();

    const bool ok = found_hardware_gpu && !cameras.empty();
    std::cout << "Environment   : " << (ok ? "OK" : "WARNING — see above") << "\n";

    print_separator();

    return ok ? 0 : 1;
}
