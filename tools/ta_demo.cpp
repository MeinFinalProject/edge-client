#include "arcface/arcface.hpp"
#include "liveness/liveness.hpp"
#include "pipeline/face_tracking_runtime.hpp"
#include "pipeline/attendance_processor.hpp"

#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/base.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using vision_runtime::FaceDetection;
using vision_runtime::GpuFrame;
using vision_runtime::arcface::ArcFaceRecognizer;
using vision_runtime::arcface::ArcFaceResult;
using vision_runtime::arcface::kEmbeddingSize;
using vision_runtime::bytetrack::TrackResult;
using vision_runtime::liveness::LivenessDetector;
using vision_runtime::pipeline::FaceTrackingRuntime;

constexpr wchar_t kWindowClassName[] = L"TAWalkThroughDemoWindow";

std::filesystem::path executable_directory() {
    std::wstring buffer(32768, L'\0');
    const DWORD n = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (n == 0 || n >= buffer.size()) {
        throw std::runtime_error("GetModuleFileNameW failed");
    }
    buffer.resize(n);
    return std::filesystem::path{buffer}.parent_path();
}

std::filesystem::path find_upwards(
    std::filesystem::path root,
    std::filesystem::path const& rel
) {
    for (int i = 0; i < 12; ++i) {
        auto candidate = root / rel;
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate).lexically_normal();
        }
        auto parent = root.parent_path();
        if (parent.empty() || parent == root) break;
        root = parent;
    }
    return {};
}

std::wstring widen_ascii(std::string const& s) {
    return std::wstring(s.begin(), s.end());
}

std::wstring fixed(float v, int precision = 2) {
    std::wostringstream ss;
    ss << std::fixed << std::setprecision(precision) << v;
    return ss.str();
}

struct Args {
    std::filesystem::path scrfd_model;
    std::filesystem::path pad_model;
    std::filesystem::path arcface_model;
    std::string identity_name = "ENROLLED";
    float min_detection_score = 0.50F;
    float identity_threshold = 0.75F; // DEMO ONLY; not final calibrated threshold.
    int enroll_samples = 12;
    int arcface_every = 5;
    int pad_every = 10;
    bool pad_monitor = true; // display toggle; recognition always requires PAD
    bool pad_disabled = false;
};

Args parse_args(int argc, char** argv) {
    Args args{};
    for (int i = 1; i < argc; ++i) {
        const std::string token = argv[i];
        auto need = [&](char const* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string{"Missing value for "} + name);
            return argv[++i];
        };

        if (token == "--scrfd") {
            args.scrfd_model = std::filesystem::absolute(need("--scrfd"));
        } else if (token == "--pad") {
            args.pad_model = std::filesystem::absolute(need("--pad"));
        } else if (token == "--arcface") {
            args.arcface_model = std::filesystem::absolute(need("--arcface"));
        } else if (token == "--name") {
            args.identity_name = need("--name");
        } else if (token == "--identity-threshold") {
            args.identity_threshold = std::stof(need("--identity-threshold"));
        } else if (token == "--min-det-score") {
            args.min_detection_score = std::stof(need("--min-det-score"));
        } else if (token == "--enroll") {
            args.enroll_samples = std::stoi(need("--enroll"));
        } else if (token == "--arc-every") {
            args.arcface_every = std::stoi(need("--arc-every"));
        } else if (token == "--pad-every") {
            args.pad_every = std::stoi(need("--pad-every"));
        } else if (token == "--pad-monitor") {
            args.pad_monitor = true;
        } else if (token == "--no-pad") {
            args.pad_disabled = true;
            args.pad_monitor = false;
        } else if (token == "--help" || token == "-h") {
            std::cout
                << "ta_demo v4.8 options:\n"
                << "  --name <label>                 default ENROLLED\n"
                << "  --identity-threshold <float>  default 0.75 (DEMO ONLY)\n"
                << "  --enroll <samples>             default 12\n"
                << "  --arc-every <frames>           default 5\n"
                << "  --pad-every <frames>           default 10\n"
                << "  --pad-monitor                   start with PAD raw-score monitor ON\n"
                << "  --no-pad                        disable PAD module entirely\n"
                << "  --scrfd <det_500m.onnx>\n"
                << "  --pad <liveness.onnx>          selected InsightFace addon\n"
                << "  --arcface <w600k_r50.onnx>\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + token);
        }
    }

    const auto cwd = std::filesystem::current_path();
    const auto exe = executable_directory();

    if (args.scrfd_model.empty()) {
        const auto rel = std::filesystem::path{L"models\\insightface\\buffalo_sc\\det_500m.onnx"};
        args.scrfd_model = find_upwards(cwd, rel);
        if (args.scrfd_model.empty()) args.scrfd_model = find_upwards(exe, rel);
    }
    if (args.pad_model.empty()) {
        const auto rel = std::filesystem::path{L"models"} / vision_runtime::liveness::kModelRelativePath;
        args.pad_model = find_upwards(cwd, rel);
        if (args.pad_model.empty()) args.pad_model = find_upwards(exe, rel);
    }
    if (args.arcface_model.empty()) {
        const auto rel = std::filesystem::path{L"models\\insightface\\buffalo_l\\w600k_r50.onnx"};
        args.arcface_model = find_upwards(cwd, rel);
        if (args.arcface_model.empty()) args.arcface_model = find_upwards(exe, rel);
    }

    if (args.scrfd_model.empty() || !std::filesystem::exists(args.scrfd_model)) {
        throw std::runtime_error("det_500m.onnx not found; pass --scrfd <path>");
    }
    if (args.arcface_model.empty() || !std::filesystem::exists(args.arcface_model)) {
        throw std::runtime_error("w600k_r50.onnx not found; pass --arcface <path>");
    }
    if (!args.pad_disabled && (args.pad_model.empty() || !std::filesystem::exists(args.pad_model))) {
        throw std::runtime_error("Selected liveness.onnx not found; pass --pad <path> (or explicit --no-pad for diagnostics)");
    }
    if (!std::isfinite(args.identity_threshold) || args.identity_threshold < -1.0F || args.identity_threshold > 1.0F) {
        throw std::runtime_error("--identity-threshold must be in [-1,1]");
    }
    if (!std::isfinite(args.min_detection_score) || args.min_detection_score < 0.0F || args.min_detection_score > 1.0F) {
        throw std::runtime_error("--min-det-score must be 0..1");
    }
    if (args.enroll_samples < 1 || args.arcface_every < 1 || args.pad_every < 1) {
        throw std::runtime_error("enroll/every values must be >= 1");
    }
    return args;
}

struct OverlayTrack {
    std::uint64_t track_id = 0;
    float x1 = 0.0F;
    float y1 = 0.0F;
    float x2 = 0.0F;
    float y2 = 0.0F;
    float det_score = 0.0F;
    std::optional<float> pad_p_real;
    std::optional<float> cosine;
    bool identity_matched = false;
    std::wstring identity_name;
};

struct HudInfo {
    double fps = 0.0;
    double scrfd_ms = 0.0;
    double bytetrack_ms = 0.0;
    bool reference_ready = false;
    int enroll_count = 0;
    int enroll_target = 12;
    bool pad_monitor = false;
    float demo_identity_threshold = 0.75F;
    std::wstring message;
};

class PreviewWindow {
public:
    PreviewWindow(std::uint32_t source_width, std::uint32_t source_height)
        : source_width_(source_width), source_height_(source_height) {
        create_window();
    }

    ~PreviewWindow() {
        if (font_) DeleteObject(font_);
        if (font_small_) DeleteObject(font_small_);
        if (hwnd_ && IsWindow(hwnd_)) DestroyWindow(hwnd_);
    }

    PreviewWindow(PreviewWindow const&) = delete;
    PreviewWindow& operator=(PreviewWindow const&) = delete;

    [[nodiscard]] bool pump_messages() {
        MSG msg{};
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                closed_ = true;
                return false;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        return !closed_;
    }

    [[nodiscard]] bool consume_key(int vk) {
        const SHORT value = GetAsyncKeyState(vk);
        const bool down = (value & 0x8000) != 0;
        const bool was_down = key_down_[vk];
        key_down_[vk] = down;
        return down && !was_down;
    }

    void render(
        GpuFrame const& frame,
        std::vector<OverlayTrack> const& tracks,
        HudInfo const& hud
    ) {
        ensure_graphics(frame);

        winrt::com_ptr<ID3D11VideoProcessorInputView> input_view;
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC iv{};
        iv.FourCC = 0;
        iv.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        iv.Texture2D.MipSlice = 0;
        iv.Texture2D.ArraySlice = 0;
        winrt::check_hresult(video_device_->CreateVideoProcessorInputView(
            frame.texture.get(), enumerator_.get(), &iv, input_view.put()
        ));

        RECT src{0, 0, static_cast<LONG>(source_width_), static_cast<LONG>(source_height_)};
        RECT dst{0, 0, static_cast<LONG>(source_width_), static_cast<LONG>(source_height_)};
        video_context_->VideoProcessorSetStreamFrameFormat(
            processor_.get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE
        );
        video_context_->VideoProcessorSetStreamSourceRect(processor_.get(), 0, TRUE, &src);
        video_context_->VideoProcessorSetStreamDestRect(processor_.get(), 0, TRUE, &dst);
        video_context_->VideoProcessorSetOutputTargetRect(processor_.get(), TRUE, &dst);

        D3D11_VIDEO_PROCESSOR_STREAM stream{};
        stream.Enable = TRUE;
        stream.OutputIndex = 0;
        stream.InputFrameOrField = 0;
        stream.PastFrames = 0;
        stream.FutureFrames = 0;
        stream.pInputSurface = input_view.get();

        winrt::check_hresult(video_context_->VideoProcessorBlt(
            processor_.get(), output_view_.get(), 0, 1, &stream
        ));

        draw_overlay(tracks, hud);
        winrt::check_hresult(swap_chain_->Present(1, 0));
    }

private:
    static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        PreviewWindow* self = reinterpret_cast<PreviewWindow*>(
            GetWindowLongPtrW(hwnd, GWLP_USERDATA)
        );
        if (msg == WM_NCCREATE) {
            auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
            self = static_cast<PreviewWindow*>(cs->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        }
        if (self) {
            if (msg == WM_CLOSE) {
                self->closed_ = true;
                DestroyWindow(hwnd);
                return 0;
            }
            if (msg == WM_DESTROY) {
                self->closed_ = true;
                PostQuitMessage(0);
                return 0;
            }
        }
        return DefWindowProcW(hwnd, msg, wp, lp);
    }

    void create_window() {
        HINSTANCE instance = GetModuleHandleW(nullptr);
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = &PreviewWindow::wnd_proc;
        wc.hInstance = instance;
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        wc.lpszClassName = kWindowClassName;
        RegisterClassExW(&wc);

        constexpr int client_w = 1067; // 16:9 preview that fits comfortably on 1080p.
        constexpr int client_h = 600;
        RECT rect{0, 0, client_w, client_h};
        const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        AdjustWindowRect(&rect, style, FALSE);

        hwnd_ = CreateWindowExW(
            0,
            kWindowClassName,
            L"TA Walk-Through Prototype v4.8",
            style,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            nullptr,
            nullptr,
            instance,
            this
        );
        if (!hwnd_) throw std::runtime_error("CreateWindowExW failed");
        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);

        font_ = CreateFontW(
            25, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI"
        );
        font_small_ = CreateFontW(
            21, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI"
        );
    }

    void ensure_graphics(GpuFrame const& frame) {
        if (swap_chain_) return;
        if (!frame.valid()) throw std::runtime_error("Preview received invalid GPU frame");
        if (frame.width != source_width_ || frame.height != source_height_) {
            throw std::runtime_error("Preview source size changed unexpectedly");
        }
        if (frame.dxgi_format != DXGI_FORMAT_NV12) {
            throw std::runtime_error("Preview expects DXGI_FORMAT_NV12 camera frame");
        }

        frame.texture->GetDevice(device_.put());
        if (!device_) throw std::runtime_error("ID3D11Texture2D::GetDevice returned null");
        device_->GetImmediateContext(context_.put());

        video_device_ = device_.as<ID3D11VideoDevice>();
        video_context_ = context_.as<ID3D11VideoContext>();

        auto dxgi_device = device_.as<IDXGIDevice>();
        winrt::com_ptr<IDXGIAdapter> adapter;
        winrt::check_hresult(dxgi_device->GetAdapter(adapter.put()));
        winrt::com_ptr<IDXGIFactory2> factory;
        winrt::check_hresult(adapter->GetParent(
            __uuidof(IDXGIFactory2), factory.put_void()
        ));

        DXGI_SWAP_CHAIN_DESC1 desc{};
        desc.Width = source_width_;
        desc.Height = source_height_;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.Stereo = FALSE;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = 1;
        desc.Scaling = DXGI_SCALING_STRETCH;
        desc.SwapEffect = DXGI_SWAP_EFFECT_SEQUENTIAL;
        desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        desc.Flags = DXGI_SWAP_CHAIN_FLAG_GDI_COMPATIBLE;

        winrt::check_hresult(factory->CreateSwapChainForHwnd(
            device_.get(), hwnd_, &desc, nullptr, nullptr, swap_chain_.put()
        ));
        factory->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER);

        D3D11_VIDEO_PROCESSOR_CONTENT_DESC content{};
        content.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        content.InputFrameRate.Numerator = 30;
        content.InputFrameRate.Denominator = 1;
        content.InputWidth = source_width_;
        content.InputHeight = source_height_;
        content.OutputFrameRate.Numerator = 30;
        content.OutputFrameRate.Denominator = 1;
        content.OutputWidth = source_width_;
        content.OutputHeight = source_height_;
        content.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

        winrt::check_hresult(video_device_->CreateVideoProcessorEnumerator(
            &content, enumerator_.put()
        ));
        winrt::check_hresult(video_device_->CreateVideoProcessor(
            enumerator_.get(), 0, processor_.put()
        ));

        winrt::check_hresult(swap_chain_->GetBuffer(
            0, __uuidof(ID3D11Texture2D), back_buffer_.put_void()
        ));

        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC ov{};
        ov.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        ov.Texture2D.MipSlice = 0;
        winrt::check_hresult(video_device_->CreateVideoProcessorOutputView(
            back_buffer_.get(), enumerator_.get(), &ov, output_view_.put()
        ));

        gdi_surface_ = back_buffer_.as<IDXGISurface1>();
    }

    void text(HDC dc, int x, int y, std::wstring const& s, COLORREF color, HFONT font) {
        SelectObject(dc, font);
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, RGB(0, 0, 0));
        TextOutW(dc, x + 2, y + 2, s.c_str(), static_cast<int>(s.size()));
        SetTextColor(dc, color);
        TextOutW(dc, x, y, s.c_str(), static_cast<int>(s.size()));
    }

    void draw_overlay(std::vector<OverlayTrack> const& tracks, HudInfo const& hud) {
        HDC dc = nullptr;
        winrt::check_hresult(gdi_surface_->GetDC(FALSE, &dc));

        HBRUSH panel = CreateSolidBrush(RGB(15, 15, 15));
        RECT top{0, 0, static_cast<LONG>(source_width_), 76};
        FillRect(dc, &top, panel);
        DeleteObject(panel);

        std::wostringstream line1;
        line1 << L"FPS " << std::fixed << std::setprecision(1) << hud.fps
              << L"   SCRFD GPU " << std::setprecision(1) << hud.scrfd_ms << L" ms"
              << L"   ByteTrack CPU " << std::setprecision(3) << hud.bytetrack_ms << L" ms";
        text(dc, 18, 9, line1.str(), RGB(110, 235, 255), font_);

        std::wstring line2 = L"GPU image path: NV12 -> SCRFD / PAD / ArcFace   |   CPU: ByteTrack + decisions + overlay";
        if (hud.pad_monitor) line2 += L"   |   PAD ADDON: VOTE MEDIAN (AFTER ENROLLMENT)";
        text(dc, 18, 41, line2, RGB(235, 235, 235), font_small_);

        if (!hud.message.empty()) {
            text(dc, 18, 82, hud.message, RGB(255, 225, 80), font_);
        }

        for (auto const& t : tracks) {
            const int x1 = std::clamp(static_cast<int>(std::lround(t.x1)), 0, static_cast<int>(source_width_) - 1);
            const int y1 = std::clamp(static_cast<int>(std::lround(t.y1)), 0, static_cast<int>(source_height_) - 1);
            const int x2 = std::clamp(static_cast<int>(std::lround(t.x2)), x1 + 1, static_cast<int>(source_width_));
            const int y2 = std::clamp(static_cast<int>(std::lround(t.y2)), y1 + 1, static_cast<int>(source_height_));

            COLORREF color = RGB(0, 220, 255);
            if (t.identity_matched) color = RGB(50, 255, 90);
            else if (t.cosine.has_value()) color = RGB(255, 210, 50);

            HPEN pen = CreatePen(PS_SOLID, 4, color);
            HGDIOBJ old_pen = SelectObject(dc, pen);
            HGDIOBJ old_brush = SelectObject(dc, GetStockObject(NULL_BRUSH));
            Rectangle(dc, x1, y1, x2, y2);
            SelectObject(dc, old_brush);
            SelectObject(dc, old_pen);
            DeleteObject(pen);

            std::wostringstream row1;
            row1 << L"ID" << t.track_id << L"  det " << std::fixed << std::setprecision(2) << t.det_score;
            text(dc, x1 + 4, std::max(80, y1 - 50), row1.str(), color, font_);

            std::wostringstream row2;
            if (t.identity_matched) {
                row2 << t.identity_name;
                if (t.cosine) row2 << L"  cos " << std::fixed << std::setprecision(3) << *t.cosine;
            } else if (t.cosine) {
                row2 << L"ArcFace cos " << std::fixed << std::setprecision(3) << *t.cosine
                     << L"  (demo T=" << std::setprecision(2) << hud.demo_identity_threshold << L")";
            } else {
                row2 << L"ArcFace waiting";
            }
            text(dc, x1 + 4, std::max(104, y1 - 22), row2.str(), color, font_small_);

            if (hud.pad_monitor && t.pad_p_real) {
                std::wostringstream row3;
                row3 << L"PAD real " << std::fixed << std::setprecision(3) << *t.pad_p_real
                     << (*t.pad_p_real >= vision_runtime::liveness::kTaLiveThreshold ? L" LIVE" : L" SPOOF");
                text(dc, x1 + 4, std::min(static_cast<int>(source_height_) - 28, y2 + 5), row3.str(), RGB(255, 190, 80), font_small_);
            }
        }

        std::wstring keys = L"[E] re-enroll   [P] PAD monitor on/off   [R] reset tracker/state   [ESC] exit";
        text(dc, 18, static_cast<int>(source_height_) - 33, keys, RGB(240, 240, 240), font_small_);

        gdi_surface_->ReleaseDC(nullptr);
    }

    std::uint32_t source_width_ = 0;
    std::uint32_t source_height_ = 0;
    HWND hwnd_ = nullptr;
    bool closed_ = false;
    bool key_down_[256]{};
    HFONT font_ = nullptr;
    HFONT font_small_ = nullptr;

    winrt::com_ptr<ID3D11Device> device_;
    winrt::com_ptr<ID3D11DeviceContext> context_;
    winrt::com_ptr<ID3D11VideoDevice> video_device_;
    winrt::com_ptr<ID3D11VideoContext> video_context_;
    winrt::com_ptr<ID3D11VideoProcessorEnumerator> enumerator_;
    winrt::com_ptr<ID3D11VideoProcessor> processor_;
    winrt::com_ptr<IDXGISwapChain1> swap_chain_;
    winrt::com_ptr<ID3D11Texture2D> back_buffer_;
    winrt::com_ptr<ID3D11VideoProcessorOutputView> output_view_;
    winrt::com_ptr<IDXGISurface1> gdi_surface_;
};

struct TrackState {
    std::uint64_t last_seen_tracking_frame = 0;
    std::optional<float> pad_p_real;
    std::optional<float> cosine;
    bool identity_matched = false;
};

std::array<float, kEmbeddingSize> finalize_reference(
    std::array<double, kEmbeddingSize> const& sum,
    int count
) {
    if (count <= 0) throw std::runtime_error("No enrollment embeddings");
    std::array<float, kEmbeddingSize> ref{};
    for (std::size_t i = 0; i < ref.size(); ++i) {
        ref[i] = static_cast<float>(sum[i] / static_cast<double>(count));
    }
    return vision_runtime::arcface::normalize_embedding(std::move(ref));
}

void clear_identity_state(
    std::unordered_map<std::uint64_t, TrackState>& states,
    std::array<double, kEmbeddingSize>& enroll_sum,
    int& enrolled,
    bool& reference_ready,
    std::array<float, kEmbeddingSize>& reference
) {
    enroll_sum.fill(0.0);
    enrolled = 0;
    reference_ready = false;
    reference.fill(0.0F);
    for (auto& [id, state] : states) {
        (void)id;
        state.cosine.reset();
        state.identity_matched = false;
        state.pad_p_real.reset();
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        SetProcessDPIAware();
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        const Args args = parse_args(argc, argv);

        vision_runtime::pipeline::FaceTrackingConfig tracking_cfg{};
        tracking_cfg.camera.width = 1280;
        tracking_cfg.camera.height = 720;
        tracking_cfg.camera.fps = 30;
        tracking_cfg.camera.source_subtype = L"NV12";
        tracking_cfg.camera.output_subtype = L"NV12";

        tracking_cfg.scrfd.model_path = args.scrfd_model;
        tracking_cfg.scrfd.input_width = 640;
        tracking_cfg.scrfd.input_height = 640;
        tracking_cfg.scrfd.score_threshold = 0.10F;
        tracking_cfg.scrfd.nms_threshold = 0.40F;

        tracking_cfg.bytetrack.track_threshold = 0.50F;
        tracking_cfg.bytetrack.low_threshold = 0.10F;
        tracking_cfg.bytetrack.new_track_threshold = 0.60F;
        tracking_cfg.bytetrack.match_threshold = 0.80F;
        tracking_cfg.bytetrack.second_match_threshold = 0.50F;
        tracking_cfg.bytetrack.unconfirmed_match_threshold = 0.70F;
        tracking_cfg.bytetrack.track_buffer = 30;
        tracking_cfg.bytetrack.frame_rate = 30;
        tracking_cfg.bytetrack.mot20 = false;

        vision_runtime::arcface::ArcFaceConfig arc_cfg{};
        arc_cfg.model_path = args.arcface_model;
        arc_cfg.input_width = 112;
        arc_cfg.input_height = 112;

        std::optional<vision_runtime::liveness::LivenessConfig> pad_cfg;
        if (!args.pad_disabled) {
            vision_runtime::liveness::LivenessConfig cfg{};
            cfg.model_path = args.pad_model;
            pad_cfg = cfg;
        }

        std::cout
            << "================================================================================\n"
            << "TA WALK-THROUGH PROTOTYPE DEMO\n"
            << "================================================================================\n"
            << "Camera              : MediaCapture/MediaFrameReader 1280x720@30 NV12 GPU-backed\n"
            << "SCRFD               : GPU preprocessing + WinML/DirectML\n"
            << "ByteTrack           : native CPU (intentional winner)\n"
            << "PAD                 : InsightFace addon GPU; TA threshold 0.95; shared attendance gate\n"
            << "ArcFace             : GPU alignment/preprocessing + WinML/DirectML\n"
            << "Preview             : D3D11 video processor -> swapchain; no CPU image array\n"
            << "Overlay/control     : CPU/GDI + small model outputs\n"
            << "Enrollment          : " << args.enroll_samples << " ArcFace samples\n"
            << "Demo identity T     : " << args.identity_threshold << " (NOT final calibrated)\n\n";

        FaceTrackingRuntime runtime{tracking_cfg};
        ArcFaceRecognizer recognizer{arc_cfg};
        std::unique_ptr<LivenessDetector> pad;
        if (pad_cfg) pad = std::make_unique<LivenessDetector>(*pad_cfg);
        std::unique_ptr<vision_runtime::pipeline::AttendanceProcessor> decisions;
        if (pad) {
            vision_runtime::pipeline::BiometricAttendanceConfig config;
            config.recognition_similarity_threshold = args.identity_threshold;
            config.allow_uncalibrated_thresholds = true;
            config.min_detection_score = args.min_detection_score;
            config.pad_every_frames = static_cast<std::size_t>(args.pad_every);
            config.recognition_every_frames = static_cast<std::size_t>(args.arcface_every);
            decisions = std::make_unique<vision_runtime::pipeline::AttendanceProcessor>(config,
                vision_runtime::pipeline::BiometricEvaluators{
                    [&](auto const& frame, auto const& faces) { return pad->evaluate_many(frame, faces); },
                    [&](auto const& frame, auto const& faces) { return recognizer.evaluate_many(frame, faces); }
                });
        }


        PreviewWindow preview{1280, 720};
        runtime.start();

        constexpr int kWarmup = 35;
        for (int i = 0; i < kWarmup; ++i) {
            auto f = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!f) throw std::runtime_error("warm-up timeout");
            if (!preview.pump_messages()) return 0;
        }
        runtime.reset_tracker();

        std::unordered_map<std::uint64_t, TrackState> states;
        std::array<double, kEmbeddingSize> enroll_sum{};
        int enrolled = 0;
        bool reference_ready = false;
        std::array<float, kEmbeddingSize> reference{};
        bool pad_monitor_enabled = args.pad_monitor && pad != nullptr;

        std::uint64_t tracking_frame_index = 0;
        auto fps_window_begin = Clock::now();
        std::uint64_t fps_window_frames = 0;
        double fps = 0.0;
        bool running = true;

        while (running && preview.pump_messages()) {
            if (preview.consume_key(VK_ESCAPE)) break;
            if (preview.consume_key('E')) {
                clear_identity_state(states, enroll_sum, enrolled, reference_ready, reference);
                if (decisions) decisions->set_gallery({});
                std::cout << "[DEMO] Re-enrollment requested.\n";
            }
            if (preview.consume_key('P')) {
                if (pad) {
                    pad_monitor_enabled = !pad_monitor_enabled;
                    std::cout << "[DEMO] PAD raw monitor " << (pad_monitor_enabled ? "ON" : "OFF") << "\n";
                } else {
                    std::cout << "[DEMO] PAD module unavailable.\n";
                }
            }
            if (preview.consume_key('R')) {
                runtime.reset_tracker();
                states.clear();
                clear_identity_state(states, enroll_sum, enrolled, reference_ready, reference);
                if (decisions) decisions->set_gallery({});
                std::cout << "[DEMO] Tracker + identity state reset.\n";
            }

            auto frame = runtime.wait_for_frame(std::chrono::milliseconds{1500});
            if (!frame) continue;
            ++tracking_frame_index;
            ++fps_window_frames;

            const auto now = Clock::now();
            const double fps_elapsed = std::chrono::duration<double>(now - fps_window_begin).count();
            if (fps_elapsed >= 0.5) {
                fps = static_cast<double>(fps_window_frames) / fps_elapsed;
                fps_window_frames = 0;
                fps_window_begin = now;
            }

            for (auto const& track : frame->tracks) {
                auto& state = states[track.track_id];
                state.last_seen_tracking_frame = tracking_frame_index;
            }
            for (auto it = states.begin(); it != states.end();) {
                if (tracking_frame_index > it->second.last_seen_tracking_frame + 90) {
                    it = states.erase(it);
                } else {
                    ++it;
                }
            }

            // During manual enrollment, show scores only. Once enrolled, the shared
            // processor supplies PAD results without a second inference pass.
            if (!reference_ready && pad && pad_monitor_enabled && (tracking_frame_index % static_cast<std::uint64_t>(args.pad_every) == 0)) {
                std::vector<vision_runtime::FaceDetection> observations;
                std::vector<std::uint64_t> ids;
                for (auto const& track : frame->tracks) {
                    if (!track.has_observation || track.observation.score < args.min_detection_score) continue;
                    observations.push_back(track.observation);
                    ids.push_back(track.track_id);
                }
                if (!observations.empty()) {
                    auto results = pad->evaluate_many(frame->gpu_frame, observations);
                    for (std::size_t i = 0; i < results.size() && i < ids.size(); ++i) {
                        states[ids[i]].pad_p_real = results[i].live_score;
                    }
                }
            }

            // Enrollment: require exactly one current high-quality observed face.
            if (!reference_ready && (tracking_frame_index % static_cast<std::uint64_t>(args.arcface_every) == 2 % args.arcface_every)) {
                std::vector<TrackResult const*> candidates;
                for (auto const& track : frame->tracks) {
                    if (track.has_observation && track.observation.score >= args.min_detection_score) {
                        candidates.push_back(&track);
                    }
                }
                if (candidates.size() == 1) {
                    ArcFaceResult result = recognizer.evaluate(frame->gpu_frame, candidates[0]->observation);
                    for (std::size_t i = 0; i < kEmbeddingSize; ++i) enroll_sum[i] += result.embedding[i];
                    ++enrolled;
                    if (enrolled >= args.enroll_samples) {
                        reference = finalize_reference(enroll_sum, enrolled);
                        reference_ready = true;
                        if (decisions) decisions->set_gallery({{args.identity_name, reference}});
                        for (auto& [id, state] : states) {
                            (void)id;
                            state.cosine.reset();
                            state.identity_matched = false;
                        }
                        std::cout << "[DEMO] ArcFace reference locked from " << enrolled << " samples.\n";
                    }
                }
            }

            // Runtime and demo use the same PAD/quality/retry/cooldown rules.
            // Enrollment remains a supervised demo operation, not remote enrollment.
            if (reference_ready && decisions) {
                auto result = decisions->process(*frame);
                using Phase = vision_runtime::pipeline::BiometricTrackPhase;
                for (auto const& snapshot : result.tracks) {
                    auto& state = states[snapshot.track_id];
                    state.identity_matched = snapshot.phase == Phase::Recognized ||
                                             snapshot.phase == Phase::DuplicateIdentity;
                    state.cosine = snapshot.recognition_attempts
                        ? std::optional<float>{snapshot.identity_similarity} : std::nullopt;
                    state.pad_p_real = snapshot.pad_attempts
                        ? std::optional<float>{snapshot.pad_median_p_real} : std::nullopt;
                }
                for (auto const& event : result.attendance_events) {
                    std::cout << "[ATTENDANCE-DEMO] identity=" << event.identity_id
                              << " track=ID" << event.track_id
                              << " cosine=" << event.similarity
                              << " (shared decision engine; demo enrollment, not persisted)\n";
                }
            }

            std::vector<OverlayTrack> overlay_tracks;
            overlay_tracks.reserve(frame->tracks.size());
            for (auto const& track : frame->tracks) {
                OverlayTrack o{};
                o.track_id = track.track_id;
                o.x1 = track.x1;
                o.y1 = track.y1;
                o.x2 = track.x2;
                o.y2 = track.y2;
                o.det_score = track.score;
                auto it = states.find(track.track_id);
                if (it != states.end()) {
                    o.pad_p_real = it->second.pad_p_real;
                    o.cosine = it->second.cosine;
                    o.identity_matched = it->second.identity_matched;
                    if (o.identity_matched) o.identity_name = widen_ascii(args.identity_name) + L" | IDENTIFIED";
                }
                overlay_tracks.push_back(std::move(o));
            }

            HudInfo hud{};
            hud.fps = fps;
            hud.scrfd_ms = frame->timing.detector_total_ms;
            hud.bytetrack_ms = frame->timing.tracker_total_ms;
            hud.reference_ready = reference_ready;
            hud.enroll_count = enrolled;
            hud.enroll_target = args.enroll_samples;
            hud.pad_monitor = pad_monitor_enabled && pad != nullptr;
            hud.demo_identity_threshold = args.identity_threshold;

            if (!reference_ready) {
                std::wostringstream msg;
                msg << L"ARCFACE ENROLLMENT " << enrolled << L"/" << args.enroll_samples
                    << L" - keep exactly one face visible";
                hud.message = msg.str();
            } else {
                hud.message = decisions ? L"REFERENCE READY - shared PAD gate + recognition; DEMO threshold, no persistence"
                                        : L"PAD DISABLED - enrollment/preview only; attendance disabled";
            }

            preview.render(frame->gpu_frame, overlay_tracks, hud);
        }

        runtime.stop();
        return 0;
    } catch (winrt::hresult_error const& e) {
        std::cerr << "WinRT/DX error: 0x" << std::hex << static_cast<unsigned long>(e.code().value)
                  << std::dec << " - " << winrt::to_string(e.message()) << "\n";
        return 2;
    } catch (std::exception const& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
