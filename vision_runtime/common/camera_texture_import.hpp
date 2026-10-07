#pragma once

// Zero-copy import texture kamera (NV12) ke device D3D11On12 ("interop").
//
// Perbedaan dari versi lama:
//   * Tidak lagi memakai ID3D11Device1::OpenSharedResource1 di device 11on12
//     (itu yang gagal E_INVALIDARG). Sekarang: NT handle -> ID3D12Device::OpenSharedHandle
//     -> ID3D11On12Device::CreateWrappedResource -> SRV plane Y/UV.
//   * Jalur legacy GetSharedHandle dihapus (selalu gagal untuk texture NTHANDLE).
//   * Views.wrapped diakses melalui WrappedAccess agar Acquire/ReleaseWrappedResources
//     tidak mudah terlewat di caller.
//   * EDGE_ZERO_COPY=0 mematikan fitur ini.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include <d3d11_4.h>
#include <d3d11on12.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <winrt/base.h>

#include "common/gpu_frame.hpp"

namespace vision_runtime {

class CameraTextureImporter {
  public:
    struct Views {
        winrt::com_ptr<ID3D11Resource> wrapped;      // backing resource untuk WrappedAccess
        winrt::com_ptr<ID3D11ShaderResourceView> y;  // plane 0, R8_UNORM
        winrt::com_ptr<ID3D11ShaderResourceView> uv; // plane 1, R8G8_UNORM
    };

    class WrappedAccess {
      public:
        WrappedAccess() noexcept = default;
        WrappedAccess(WrappedAccess const&) = delete;
        WrappedAccess& operator=(WrappedAccess const&) = delete;

        WrappedAccess(WrappedAccess&& other) noexcept
            : on12_(std::move(other.on12_)), resource_(other.resource_), active_(other.active_) {
            other.resource_ = nullptr;
            other.active_ = false;
        }

        WrappedAccess& operator=(WrappedAccess&& other) noexcept {
            if (this != &other) {
                release();
                on12_ = std::move(other.on12_);
                resource_ = other.resource_;
                active_ = other.active_;
                other.resource_ = nullptr;
                other.active_ = false;
            }
            return *this;
        }

        ~WrappedAccess() {
            release();
        }

        void release() noexcept {
            if (active_ && on12_ && resource_) {
                on12_->ReleaseWrappedResources(&resource_, 1);
            }
            resource_ = nullptr;
            active_ = false;
        }

      private:
        friend class CameraTextureImporter;

        WrappedAccess(winrt::com_ptr<ID3D11On12Device> on12, ID3D11Resource* resource) noexcept
            : on12_(std::move(on12)), resource_(resource), active_(on12_ && resource_ != nullptr) {
            if (active_) {
                on12_->AcquireWrappedResources(&resource_, 1);
            }
        }

        winrt::com_ptr<ID3D11On12Device> on12_;
        ID3D11Resource* resource_ = nullptr;
        bool active_ = false;
    };

    struct Stats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
    };

    CameraTextureImporter(winrt::com_ptr<ID3D11Device3> device,
                          winrt::com_ptr<ID3D11On12Device> on12, std::string label,
                          std::size_t max_entries = 8)
        : device_(std::move(device)), on12_(std::move(on12)), label_(std::move(label)),
          max_entries_(max_entries == 0 ? 1 : max_entries), enabled_(env_enabled()) {
        if (!enabled_) {
            log("dimatikan lewat EDGE_ZERO_COPY=0; memakai CopyResource, tidak ada zero-copy");
            return;
        }
        if (on12_) {
            if (auto on12_1 = on12_.try_as<ID3D11On12Device1>()) {
                on12_1->GetD3D12Device(__uuidof(ID3D12Device), d3d12_.put_void());
            }
        }
        if (!device_ || !on12_ || !d3d12_) {
            enabled_ = false;
            log("tidak bisa mengambil ID3D12Device dari device 11on12; memakai CopyResource");
        }
    }

    CameraTextureImporter(CameraTextureImporter const&) = delete;
    CameraTextureImporter& operator=(CameraTextureImporter const&) = delete;

    ~CameraTextureImporter() {
        if (stats_.hits + stats_.misses > 0) {
            std::cerr << "[zero-copy:" << label_ << "] ringkasan: hits=" << stats_.hits
                      << " misses(texture berbeda)=" << stats_.misses
                      << " status=" << (enabled_ ? "aktif" : "fallback-copy") << std::endl;
        }
    }

    [[nodiscard]] bool enabled() const noexcept {
        return enabled_;
    }
    [[nodiscard]] Stats const& stats() const noexcept {
        return stats_;
    }

    // Acquires the imported camera resource for D3D11On12 use. The returned
    // guard must be released before flushing the interop context.
    [[nodiscard]] WrappedAccess access(Views const* views) const noexcept {
        return WrappedAccess{on12_, views != nullptr ? views->wrapped.get() : nullptr};
    }

    // nullopt   -> pakai jalur copy lama.
    // has_value -> SRV Y/UV menunjuk texture kamera; caller wajib memakai access().
    [[nodiscard]] std::optional<Views> acquire(GpuFrame const& frame) noexcept {
        if (!enabled_ || !frame.valid()) {
            return std::nullopt;
        }
        try {
            ID3D11Texture2D* camera_texture = frame.texture.get();

            std::uint64_t id = 0;
            HRESULT tag_hr = tag_of(camera_texture, id);
            if (FAILED(tag_hr)) {
                disable(frame, ImportFailure{tag_hr, "ID3D11Texture2D::SetPrivateData(tag)"});
                return std::nullopt;
            }

            if (auto it = index_.find(id); it != index_.end()) {
                lru_.splice(lru_.begin(), lru_, it->second);
                ++stats_.hits;
                return it->second->views;
            }

            Views views;
            ImportFailure failure;
            if (!import_new(frame, views, failure)) {
                disable(frame, failure);
                return std::nullopt;
            }

            ++stats_.misses;
            lru_.push_front(Entry{id, views});
            index_[id] = lru_.begin();
            while (lru_.size() > max_entries_) {
                index_.erase(lru_.back().id);
                lru_.pop_back();
            }
            if (stats_.misses <= 8) {
                log("texture kamera baru diimpor (distinct #" + std::to_string(stats_.misses) +
                    ")");
            }
            return views;
        } catch (...) {
            disable(frame, ImportFailure{E_FAIL, "exception tak terduga"});
            return std::nullopt;
        }
    }

  private:
    struct Entry {
        std::uint64_t id;
        Views views;
    };

    struct ImportFailure {
        HRESULT hr = S_OK;
        char const* stage = "";
    };

    struct HandleCloser {
        HANDLE handle = nullptr;
        ~HandleCloser() {
            if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
                CloseHandle(handle);
            }
        }
    };

    static bool env_enabled() {
        char* value = nullptr;
        std::size_t length = 0;
        const bool have = (_dupenv_s(&value, &length, "EDGE_ZERO_COPY") == 0) && value != nullptr;
        const bool off = have && value[0] == '0';
        std::free(value);
        return !off;
    }

    static std::atomic<std::uint64_t>& next_id() {
        static std::atomic<std::uint64_t> counter{0};
        return counter;
    }

    static HRESULT tag_of(ID3D11Texture2D* texture, std::uint64_t& out_id) {
        static const GUID kTag = {
            0x6e5c8f2a, 0x3c1d, 0x4b7e, {0x9a, 0x41, 0x7d, 0x20, 0xb5, 0x13, 0xc8, 0x6f}};
        std::uint64_t id = 0;
        UINT size = sizeof(id);
        if (SUCCEEDED(texture->GetPrivateData(kTag, &size, &id)) && size == sizeof(id) && id != 0) {
            out_id = id;
            return S_OK;
        }
        id = next_id().fetch_add(1, std::memory_order_relaxed) + 1;
        const HRESULT hr = texture->SetPrivateData(kTag, sizeof(id), &id);
        if (FAILED(hr)) {
            return hr;
        }
        out_id = id;
        return S_OK;
    }

    bool import_new(GpuFrame const& frame, Views& out, ImportFailure& fail) {
        D3D11_TEXTURE2D_DESC src{};
        frame.texture->GetDesc(&src);

        if (src.Format != DXGI_FORMAT_NV12) {
            fail = {E_INVALIDARG, "pemeriksaan awal: format bukan NV12"};
            return false;
        }
        if (src.ArraySize != 1 || src.MipLevels != 1) {
            fail = {E_NOTIMPL, "pemeriksaan awal: ArraySize/MipLevels bukan 1"};
            return false;
        }
        if ((src.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0) {
            fail = {E_INVALIDARG, "pemeriksaan awal: tidak ada BIND_SHADER_RESOURCE"};
            return false;
        }
        if ((src.MiscFlags & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX) != 0) {
            fail = {E_NOTIMPL, "pemeriksaan awal: texture memakai keyed mutex"};
            return false;
        }
        if ((src.MiscFlags & D3D11_RESOURCE_MISC_SHARED_NTHANDLE) == 0) {
            fail = {DXGI_ERROR_UNSUPPORTED, "pemeriksaan awal: texture tidak SHARED_NTHANDLE"};
            return false;
        }

        // 1) NT handle dari texture kamera
        auto dxgi = frame.texture.try_as<IDXGIResource1>();
        if (!dxgi) {
            fail = {E_NOINTERFACE, "QueryInterface(IDXGIResource1)"};
            return false;
        }
        HandleCloser closer;
        HRESULT hr = dxgi->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &closer.handle);
        if (FAILED(hr)) {
            fail = {hr, "IDXGIResource1::CreateSharedHandle"};
            return false;
        }

        // 2) Buka di D3D12 (device di bawah 11on12)
        winrt::com_ptr<ID3D12Resource> res12;
        hr = d3d12_->OpenSharedHandle(closer.handle, __uuidof(ID3D12Resource), res12.put_void());
        if (FAILED(hr)) {
            fail = {hr, "ID3D12Device::OpenSharedHandle"};
            return false;
        }

        // 3) Bungkus jadi texture D3D11 di device 11on12
        D3D11_RESOURCE_FLAGS flags{};
        flags.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        winrt::com_ptr<ID3D11Texture2D> wrapped;
        hr = on12_->CreateWrappedResource(res12.get(), &flags, D3D12_RESOURCE_STATE_COMMON,
                                          D3D12_RESOURCE_STATE_COMMON, __uuidof(ID3D11Texture2D),
                                          wrapped.put_void());
        if (FAILED(hr)) {
            fail = {hr, "ID3D11On12Device::CreateWrappedResource"};
            return false;
        }

        // 4) SRV plane Y dan UV
        D3D11_SHADER_RESOURCE_VIEW_DESC1 y_desc{};
        y_desc.Format = DXGI_FORMAT_R8_UNORM;
        y_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        y_desc.Texture2D.MostDetailedMip = 0;
        y_desc.Texture2D.MipLevels = 1;
        y_desc.Texture2D.PlaneSlice = 0;

        winrt::com_ptr<ID3D11ShaderResourceView1> y1;
        hr = device_->CreateShaderResourceView1(wrapped.get(), &y_desc, y1.put());
        if (FAILED(hr)) {
            fail = {hr, "CreateShaderResourceView1 (Y plane)"};
            return false;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC1 uv_desc{};
        uv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
        uv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        uv_desc.Texture2D.MostDetailedMip = 0;
        uv_desc.Texture2D.MipLevels = 1;
        uv_desc.Texture2D.PlaneSlice = 1;

        winrt::com_ptr<ID3D11ShaderResourceView1> uv1;
        hr = device_->CreateShaderResourceView1(wrapped.get(), &uv_desc, uv1.put());
        if (FAILED(hr)) {
            fail = {hr, "CreateShaderResourceView1 (UV plane)"};
            return false;
        }

        out.wrapped = wrapped.try_as<ID3D11Resource>();
        out.y = y1.try_as<ID3D11ShaderResourceView>();
        out.uv = uv1.try_as<ID3D11ShaderResourceView>();
        if (!out.wrapped || !out.y || !out.uv) {
            fail = {E_NOINTERFACE, "QueryInterface hasil wrap/SRV"};
            return false;
        }
        return true;
    }

    // ---- diagnostik (hanya saat gagal) ----
    static std::string hr_hex(HRESULT hr) {
        std::ostringstream out;
        out << "0x" << std::hex << std::uppercase << static_cast<std::uint32_t>(hr);
        return out.str();
    }

    void disable(GpuFrame const& frame, ImportFailure const& failure) {
        enabled_ = false;
        lru_.clear();
        index_.clear();

        std::cerr << "[zero-copy:" << label_ << "] GAGAL di '" << failure.stage << "' HRESULT "
                  << hr_hex(failure.hr) << " -> fallback ke CopyResource untuk seterusnya"
                  << std::endl;

        static std::atomic<bool> details_printed{false};
        if (details_printed.exchange(true)) {
            return;
        }
        if (frame.valid()) {
            D3D11_TEXTURE2D_DESC desc{};
            frame.texture->GetDesc(&desc);
            std::cerr << "[zero-copy:diag] texture kamera: " << desc.Width << "x" << desc.Height
                      << " Format=" << static_cast<int>(desc.Format) << " MiscFlags=0x" << std::hex
                      << desc.MiscFlags << " BindFlags=0x" << desc.BindFlags << std::dec
                      << std::endl;
        }
    }

    void log(std::string const& message) const {
        std::cerr << "[zero-copy:" << label_ << "] " << message << std::endl;
    }

    winrt::com_ptr<ID3D11Device3> device_;
    winrt::com_ptr<ID3D11On12Device> on12_;
    winrt::com_ptr<ID3D12Device> d3d12_;
    std::string label_;
    std::size_t max_entries_;
    bool enabled_;
    Stats stats_{};
    std::list<Entry> lru_;
    std::unordered_map<std::uint64_t, std::list<Entry>::iterator> index_;
};

} // namespace vision_runtime