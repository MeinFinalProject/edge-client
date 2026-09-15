#include "camera/camera.hpp"
#include "camera/directshow_low_light_control.hpp"
#include "camera/format_selector.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <windows.graphics.directx.direct3d11.interop.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Media.Capture.h>
#include <winrt/Windows.Media.Capture.Frames.h>
#include <winrt/Windows.Media.Devices.h>
#include <winrt/Windows.Media.MediaProperties.h>

namespace vision_runtime {
namespace {

using winrt::Windows::Media::Capture::MediaCaptureInitializationSettings;
using winrt::Windows::Media::Capture::MediaCaptureMemoryPreference;
using winrt::Windows::Media::Capture::MediaCaptureSharingMode;
using winrt::Windows::Media::Capture::MediaStreamType;
using winrt::Windows::Media::Capture::StreamingCaptureMode;

using winrt::Windows::Media::Capture::Frames::MediaFrameFormat;
using winrt::Windows::Media::Capture::Frames::MediaFrameReaderAcquisitionMode;
using winrt::Windows::Media::Capture::Frames::MediaFrameReaderStartStatus;
using winrt::Windows::Media::Capture::Frames::MediaFrameSource;
using winrt::Windows::Media::Capture::Frames::MediaFrameSourceGroup;
using winrt::Windows::Media::Capture::Frames::MediaFrameSourceKind;

constexpr std::uint32_t kMaxConsecutiveFailures = 30;

double frame_rate(MediaFrameFormat const& format) {
    auto ratio = format.FrameRate();
    auto denominator = ratio.Denominator();

    if (denominator == 0) {
        return 0.0;
    }

    return static_cast<double>(ratio.Numerator()) /
           static_cast<double>(denominator);
}

CameraMode to_camera_mode(MediaFrameFormat const& format) {
    CameraMode mode;

    auto video = format.VideoFormat();
    if (video) {
        mode.width = video.Width();
        mode.height = video.Height();
    }

    mode.fps = frame_rate(format);
    mode.subtype = std::wstring(format.Subtype().c_str());
    return mode;
}

CameraStreamPreference to_stream_preference(
    MediaStreamType stream_type
) noexcept {
    if (stream_type == MediaStreamType::VideoPreview) {
        return CameraStreamPreference::VideoPreview;
    }
    if (stream_type == MediaStreamType::VideoRecord) {
        return CameraStreamPreference::VideoRecord;
    }
    return CameraStreamPreference::Auto;
}

std::uint64_t steady_now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count()
    );
}

winrt::com_ptr<ID3D11Texture2D> texture_from_surface(
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DSurface const& surface
) {
    auto access = surface.as<
        ::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess
    >();

    winrt::com_ptr<ID3D11Texture2D> texture;

    winrt::check_hresult(
        access->GetInterface(
            __uuidof(ID3D11Texture2D),
            texture.put_void()
        )
    );

    return texture;
}

} // namespace

// WinRT, Media Foundation, threading, and callback state belong to one camera
// backend. Keeping the complete implementation here makes its lifecycle
// readable from top to bottom while camera.hpp remains a small public contract.
struct Camera::Impl {
    using MediaCapture = winrt::Windows::Media::Capture::MediaCapture;
    using MediaFrameReader =
        winrt::Windows::Media::Capture::Frames::MediaFrameReader;
    using MediaFrameSource =
        winrt::Windows::Media::Capture::Frames::MediaFrameSource;
    using MediaFrameSourceGroup =
        winrt::Windows::Media::Capture::Frames::MediaFrameSourceGroup;

    explicit Impl(CameraConfig requested_config);
    ~Impl();

    void start();
    void stop() noexcept;

    [[nodiscard]] std::optional<GpuFrame> wait_for_frame(
        std::uint64_t last_frame_id,
        std::chrono::milliseconds timeout
    );

    [[nodiscard]] std::uint64_t failed_reads() const noexcept;
    [[nodiscard]] std::uint64_t frame_id() const noexcept;
    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] CameraMode mode() const;
    [[nodiscard]] CameraControlStatus control_status() const;
    [[nodiscard]] CameraAcquisitionMode acquisition_mode() const noexcept;
    [[nodiscard]] CameraStreamPreference stream_preference() const noexcept;
    [[nodiscard]] CameraConfig config() const;

    void control_loop() noexcept;
    void initialize_capture();
    void cleanup_capture();

    void apply_legacy_constant_frame_rate() noexcept;
    void restore_legacy_camera_control() noexcept;
    void configure_constant_frame_rate() noexcept;

    void register_frame_handler();
    void on_frame_arrived(MediaFrameReader const& sender) noexcept;
    void record_failure(std::string const& message) noexcept;

    const CameraConfig config_data;
    const std::uint32_t index;
    const std::uint32_t width;
    const std::uint32_t height;
    const std::uint32_t fps;
    const CameraAcquisitionMode requested_acquisition_mode;
    const CameraStreamPreference requested_stream_preference;
    const std::wstring requested_source_id;
    const std::wstring requested_source_subtype;
    const std::wstring requested_output_subtype;

    std::atomic<bool> running_flag{false};
    std::atomic<bool> stop_requested{false};

    std::atomic<std::uint64_t> failed_reads_count{0};
    std::atomic<std::uint64_t> current_frame_id{0};
    std::atomic<std::uint32_t> consecutive_failures{0};

    mutable std::mutex state_mutex;
    std::condition_variable state_cv;
    GpuFrame latest;
    std::exception_ptr error;

    CameraMode active;
    CameraControlStatus controls;

    std::uint32_t texture_width = 0;
    std::uint32_t texture_height = 0;
    DXGI_FORMAT texture_format = DXGI_FORMAT_UNKNOWN;
    bool texture_metadata_ready = false;

    std::mutex lifecycle_mutex;
    std::condition_variable lifecycle_cv;
    bool start_finished = false;
    std::exception_ptr start_error;

    std::mutex control_mutex;
    std::condition_variable control_cv;
    std::thread control_thread;

    std::wstring selected_friendly_name;
    long legacy_original_value = 0;
    bool legacy_restore_pending = false;

    MediaCapture capture{nullptr};
    MediaFrameSourceGroup selected_group{nullptr};
    MediaFrameSource source{nullptr};
    MediaFrameReader reader{nullptr};

    winrt::event_token frame_token{};
    bool frame_handler_registered = false;
};

Camera::Camera(CameraConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

Camera::Camera(
    std::uint32_t index,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t fps,
    CameraAcquisitionMode acquisition_mode
)
    : Camera([&] {
        CameraConfig config;
        config.index = index;
        config.width = width;
        config.height = height;
        config.fps = fps;
        config.acquisition_mode = acquisition_mode;
        return config;
    }()) {}

Camera::~Camera() = default;

Camera& Camera::start() {
    impl_->start();
    return *this;
}

void Camera::stop() noexcept {
    impl_->stop();
}

std::optional<GpuFrame> Camera::wait_for_frame(
    std::uint64_t last_frame_id,
    std::chrono::milliseconds timeout
) {
    return impl_->wait_for_frame(last_frame_id, timeout);
}

std::uint64_t Camera::failed_reads() const noexcept {
    return impl_->failed_reads();
}

std::uint64_t Camera::frame_id() const noexcept {
    return impl_->frame_id();
}

bool Camera::running() const noexcept {
    return impl_->running();
}

CameraMode Camera::mode() const {
    return impl_->mode();
}

CameraControlStatus Camera::control_status() const {
    return impl_->control_status();
}

CameraAcquisitionMode Camera::acquisition_mode() const noexcept {
    return impl_->acquisition_mode();
}

CameraStreamPreference Camera::stream_preference() const noexcept {
    return impl_->stream_preference();
}

CameraConfig Camera::config() const {
    return impl_->config();
}

// Construction and public state ------------------------------------------------

Camera::Impl::Impl(CameraConfig requested_config)
    : config_data(std::move(requested_config)),
      index(config_data.index),
      width(config_data.width),
      height(config_data.height),
      fps(config_data.fps),
      requested_acquisition_mode(config_data.acquisition_mode),
      requested_stream_preference(config_data.stream_preference),
      requested_source_id(config_data.source_id),
      requested_source_subtype(
          camera_detail::normalize_subtype(config_data.source_subtype)
      ),
      requested_output_subtype(
          camera_detail::normalize_subtype(config_data.output_subtype)
      ) {}

Camera::Impl::~Impl() {
    stop();
}

void Camera::Impl::start() {
    if (control_thread.joinable()) {
        throw std::runtime_error("Camera sudah dijalankan");
    }

    {
        std::scoped_lock state_lock(state_mutex);
        latest = {};
        error = nullptr;
        active = {};
        controls = {};
        texture_width = 0;
        texture_height = 0;
        texture_format = DXGI_FORMAT_UNKNOWN;
        texture_metadata_ready = false;
    }

    failed_reads_count.store(0, std::memory_order_relaxed);
    consecutive_failures.store(0, std::memory_order_relaxed);
    current_frame_id.store(0, std::memory_order_relaxed);
    running_flag.store(false, std::memory_order_release);
    stop_requested.store(false, std::memory_order_release);

    {
        std::scoped_lock lifecycle_lock(lifecycle_mutex);
        start_finished = false;
        start_error = nullptr;
    }

    control_thread = std::thread([this] {
        control_loop();
    });

    std::unique_lock lifecycle_lock(lifecycle_mutex);
    lifecycle_cv.wait(
        lifecycle_lock,
        [this] { return start_finished; }
    );

    auto local_start_error = start_error;
    lifecycle_lock.unlock();

    if (local_start_error) {
        stop_requested.store(true, std::memory_order_release);
        control_cv.notify_all();

        if (control_thread.joinable()) {
            control_thread.join();
        }

        std::rethrow_exception(local_start_error);
    }
}

void Camera::Impl::stop() noexcept {
    stop_requested.store(true, std::memory_order_release);

    // Wake both the control thread and a consumer waiting for a frame.
    control_cv.notify_all();
    state_cv.notify_all();

    if (control_thread.joinable()) {
        try {
            control_thread.join();
        } catch (...) {
            // stop/destructor must remain noexcept.
        }
    }
}

std::optional<GpuFrame> Camera::Impl::wait_for_frame(
    std::uint64_t last_frame_id,
    std::chrono::milliseconds timeout
) {
    std::unique_lock lock(state_mutex);

    bool ready = state_cv.wait_for(
        lock,
        timeout,
        [this, last_frame_id] {
            return error != nullptr ||
                   !running_flag.load(std::memory_order_acquire) ||
                   (latest.valid() && latest.frame_id != last_frame_id);
        }
    );

    if (error) {
        std::rethrow_exception(error);
    }

    if (!ready || !running_flag.load(std::memory_order_acquire)) {
        return std::nullopt;
    }

    // One COM AddRef keeps the texture alive after releasing the mutex.
    return latest;
}

std::uint64_t Camera::Impl::failed_reads() const noexcept {
    return failed_reads_count.load(std::memory_order_relaxed);
}

std::uint64_t Camera::Impl::frame_id() const noexcept {
    return current_frame_id.load(std::memory_order_relaxed);
}

bool Camera::Impl::running() const noexcept {
    return running_flag.load(std::memory_order_acquire);
}

CameraMode Camera::Impl::mode() const {
    std::scoped_lock lock(state_mutex);
    return active;
}

CameraControlStatus Camera::Impl::control_status() const {
    std::scoped_lock lock(state_mutex);
    return controls;
}

CameraAcquisitionMode Camera::Impl::acquisition_mode() const noexcept {
    return requested_acquisition_mode;
}

CameraStreamPreference Camera::Impl::stream_preference() const noexcept {
    return requested_stream_preference;
}

CameraConfig Camera::Impl::config() const {
    return config_data;
}

// Lifecycle --------------------------------------------------------------------

void Camera::Impl::control_loop() noexcept {
    bool apartment_initialized = false;

    try {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        apartment_initialized = true;

        initialize_capture();

        register_frame_handler();

        auto status = reader.StartAsync().get();

        if (status != MediaFrameReaderStartStatus::Success) {
            std::ostringstream message;
            message << "MediaFrameReader gagal start. Status="
                    << static_cast<int>(status);

            switch (status) {
            case MediaFrameReaderStartStatus::UnknownFailure:
                message << " (UnknownFailure)";
                break;
            case MediaFrameReaderStartStatus::DeviceNotAvailable:
                message << " (DeviceNotAvailable)";
                break;
            case MediaFrameReaderStartStatus::OutputFormatNotSupported:
                message << " (OutputFormatNotSupported)";
                break;
            case MediaFrameReaderStartStatus::ExclusiveControlNotAvailable:
                message << " (ExclusiveControlNotAvailable)";
                break;
            default:
                break;
            }
            throw std::runtime_error(message.str());
        }

        // Some drivers expose control capability only while streaming.
        configure_constant_frame_rate();

        running_flag.store(true, std::memory_order_release);

        {
            std::scoped_lock lifecycle_lock(lifecycle_mutex);
            start_finished = true;
        }
        lifecycle_cv.notify_all();

        std::unique_lock control_lock(control_mutex);
        control_cv.wait(
            control_lock,
            [this] {
                return stop_requested.load(std::memory_order_acquire);
            }
        );

    } catch (...) {
        auto exception = std::current_exception();

        {
            std::scoped_lock lifecycle_lock(lifecycle_mutex);
            if (!start_finished) {
                start_error = exception;
                start_finished = true;
            }
        }
        lifecycle_cv.notify_all();

        {
            std::scoped_lock state_lock(state_mutex);
            if (!error) {
                error = exception;
            }
        }
        state_cv.notify_all();
    }

    running_flag.store(false, std::memory_order_release);

    try {
        cleanup_capture();
    } catch (...) {
        // Cleanup is best-effort.
    }

    {
        std::scoped_lock state_lock(state_mutex);
        latest = {};
    }
    state_cv.notify_all();

    if (apartment_initialized) {
        winrt::uninit_apartment();
    }
}

// Media Foundation setup --------------------------------------------------------

void Camera::Impl::initialize_capture() {
    auto groups = MediaFrameSourceGroup::FindAllAsync().get();

    std::vector<MediaFrameSourceGroup> eligible_groups;
    eligible_groups.reserve(groups.Size());

    for (auto const& group : groups) {
        bool has_color_video = false;

        for (auto const& info : group.SourceInfos()) {
            auto stream_type = info.MediaStreamType();

            bool video_stream =
                stream_type == MediaStreamType::VideoPreview ||
                stream_type == MediaStreamType::VideoRecord;

            if (
                info.SourceKind() == MediaFrameSourceKind::Color &&
                video_stream
            ) {
                has_color_video = true;
                break;
            }
        }

        if (has_color_video) {
            eligible_groups.push_back(group);
        }
    }

    if (index >= eligible_groups.size()) {
        std::ostringstream message;
        message << "Camera index " << index
                << " tidak tersedia. Color camera groups="
                << eligible_groups.size();
        throw std::runtime_error(message.str());
    }

    selected_group = eligible_groups[index];
    selected_friendly_name = std::wstring(
        selected_group.DisplayName().c_str()
    );

    // The DirectShow filter is released before MediaCapture opens. It is used
    // only to set legacy property #19, never as the production capture backend.
    apply_legacy_constant_frame_rate();

    MediaCaptureInitializationSettings settings;
    settings.SourceGroup(selected_group);
    settings.SharingMode(MediaCaptureSharingMode::ExclusiveControl);
    settings.MemoryPreference(MediaCaptureMemoryPreference::Auto);
    settings.StreamingCaptureMode(StreamingCaptureMode::Video);

    capture = MediaCapture();
    capture.InitializeAsync(settings).get();

    // Some UVC drivers reduce FPS in low light to extend exposure. Disable
    // exposure priority when supported, then inspect it again after reader start.
    configure_constant_frame_rate();

    // Read the map once; repeatedly calling FrameSources() is unnecessary.
    auto sources = capture.FrameSources();

    struct Candidate {
        MediaFrameSource source{nullptr};
        MediaFrameFormat format{nullptr};
        MediaStreamType stream_type = MediaStreamType::VideoPreview;
        double fps_error = 0.0;
        int stream_rank = 0;
        int format_rank = 0;
    };

    std::optional<Candidate> best;

    // Production searches only for the requested mode. Full capability
    // enumeration belongs to tools/camera_probe.cpp.
    for (auto const& info : selected_group.SourceInfos()) {
        auto stream_type = info.MediaStreamType();
        auto stream_preference = to_stream_preference(stream_type);

        bool video_stream =
            stream_type == MediaStreamType::VideoPreview ||
            stream_type == MediaStreamType::VideoRecord;

        if (
            info.SourceKind() != MediaFrameSourceKind::Color ||
            !video_stream ||
            !camera_detail::stream_matches(
                stream_preference,
                requested_stream_preference
            ) ||
            (!requested_source_id.empty() &&
             std::wstring(info.Id().c_str()) != requested_source_id) ||
            !sources.HasKey(info.Id())
        ) {
            continue;
        }

        auto candidate_source = sources.Lookup(info.Id());

        for (auto const& format : candidate_source.SupportedFormats()) {
            auto mode = to_camera_mode(format);

            if (mode.width != width || mode.height != height) {
                continue;
            }

            if (
                !requested_source_subtype.empty() &&
                camera_detail::normalize_subtype(mode.subtype) !=
                    requested_source_subtype
            ) {
                continue;
            }

            double fps_error = std::abs(
                mode.fps - static_cast<double>(fps)
            );

            int source_rank = camera_detail::stream_rank(stream_preference);
            int format_rank = camera_detail::subtype_rank(mode.subtype);

            camera_detail::FormatScore candidate_score{
                fps_error,
                source_rank,
                format_rank
            };

            if (
                !best ||
                camera_detail::is_better_candidate(
                    candidate_score,
                    camera_detail::FormatScore{
                        best->fps_error,
                        best->stream_rank,
                        best->format_rank
                    }
                )
            ) {
                best = Candidate{
                    candidate_source,
                    format,
                    stream_type,
                    fps_error,
                    source_rank,
                    format_rank
                };
            }
        }
    }

    if (!best || !camera_detail::is_supported_fps(best->fps_error)) {
        std::ostringstream message;
        message << "Mode kamera "
                << width << "x" << height
                << " @ " << fps
                << " FPS tidak tersedia untuk filter source yang diminta";
        if (!requested_source_subtype.empty()) {
            message << " (subtype="
                    << winrt::to_string(
                           winrt::hstring(requested_source_subtype.c_str())
                       )
                    << ")";
        }
        message << ". Jalankan tools/Release/camera_probe.exe dari folder build untuk melihat capability driver.";
        throw std::runtime_error(message.str());
    }

    source = best->source;
    source.SetFormatAsync(best->format).get();

    auto selected_mode = to_camera_mode(source.CurrentFormat());
    selected_mode.stream = to_stream_preference(best->stream_type);

    winrt::hstring reader_output_subtype = best->format.Subtype();
    if (!requested_output_subtype.empty()) {
        reader_output_subtype = winrt::hstring(requested_output_subtype.c_str());
    }
    selected_mode.output_subtype = std::wstring(reader_output_subtype.c_str());

    {
        std::scoped_lock state_lock(state_mutex);
        active = std::move(selected_mode);
    }

    // Requesting an explicit output subtype yields a frame copy that is safe to
    // retain as latest. A probe may transport MJPG while still outputting NV12.
    reader = capture.CreateFrameReaderAsync(
        source,
        reader_output_subtype
    ).get();

    reader.AcquisitionMode(
        requested_acquisition_mode == CameraAcquisitionMode::Buffered
            ? MediaFrameReaderAcquisitionMode::Buffered
            : MediaFrameReaderAcquisitionMode::Realtime
    );
}

// Optional camera controls ------------------------------------------------------

void Camera::Impl::apply_legacy_constant_frame_rate() noexcept {
    if (
        !config_data.force_constant_framerate ||
        selected_friendly_name.empty()
    ) {
        return;
    }

    try {
        auto snapshot = detail::query_legacy_low_light_priority(
            selected_friendly_name
        );

        CameraControlStatus observed;
        {
            std::scoped_lock state_lock(state_mutex);
            observed = controls;
        }

        observed.legacy_low_light_supported = snapshot.supported;

        if (!snapshot.supported) {
            std::scoped_lock state_lock(state_mutex);
            controls = observed;
            return;
        }

        observed.legacy_low_light_original_value = snapshot.value;
        observed.legacy_low_light_current_value = snapshot.value;

        // Value 0 means constant frame rate. If already zero, nothing needs to
        // be written or restored later.
        if (snapshot.value == 0) {
            observed.legacy_low_light_forced_off = true;
            observed.legacy_low_light_restore_pending = false;

            std::scoped_lock state_lock(state_mutex);
            controls = observed;
            return;
        }

        std::string control_error;
        const bool changed = detail::set_legacy_low_light_priority(
            selected_friendly_name,
            0,
            control_error
        );

        if (changed) {
            legacy_original_value = snapshot.value;
            legacy_restore_pending =
                config_data.restore_legacy_control_on_stop;

            observed.legacy_low_light_forced_off = true;
            observed.legacy_low_light_current_value = 0;
            observed.legacy_low_light_restore_pending =
                legacy_restore_pending;
        }

        std::scoped_lock state_lock(state_mutex);
        controls = observed;

    } catch (...) {
        // Constant-frame-rate control is an optimization. Unsupported legacy
        // controls must not prevent the main capture path from running.
    }
}

void Camera::Impl::restore_legacy_camera_control() noexcept {
    if (
        !legacy_restore_pending ||
        selected_friendly_name.empty()
    ) {
        return;
    }

    try {
        std::string restore_error;
        if (detail::set_legacy_low_light_priority(
                selected_friendly_name,
                legacy_original_value,
                restore_error
            )) {
            legacy_restore_pending = false;
        }
    } catch (...) {
        // stop()/destructor must remain noexcept.
    }
}

void Camera::Impl::configure_constant_frame_rate() noexcept {
    try {
        if (!capture) {
            return;
        }

        auto controller = capture.VideoDeviceController();
        if (!controller) {
            return;
        }

        CameraControlStatus observed;
        {
            std::scoped_lock state_lock(state_mutex);
            observed = controls;
        }

        // 1) Auto-exposure priority / low-light compensation. OFF prevents a
        // supporting driver from lowering FPS only to extend exposure.
        try {
            auto exposure_priority =
                controller.ExposurePriorityVideoControl();

            if (
                exposure_priority &&
                exposure_priority.Supported()
            ) {
                observed.exposure_priority_supported = true;

                exposure_priority.Enabled(false);

                observed.exposure_priority_enabled =
                    exposure_priority.Enabled();
                observed.exposure_priority_forced_off =
                    !observed.exposure_priority_enabled;
            }
        } catch (...) {
            // This control is optional; continue with ExposureControl.
        }

        // 2) Observe ExposureControl without silently changing auto/manual
        // exposure. Manual tuning belongs in a separate measured experiment.
        try {
            auto exposure = controller.ExposureControl();

            if (exposure && exposure.Supported()) {
                observed.exposure_control_supported = true;
                observed.exposure_auto = exposure.Auto();
                observed.exposure_value_100ns =
                    static_cast<std::int64_t>(exposure.Value().count());
                observed.exposure_min_100ns =
                    static_cast<std::int64_t>(exposure.Min().count());
                observed.exposure_max_100ns =
                    static_cast<std::int64_t>(exposure.Max().count());
                observed.exposure_step_100ns =
                    static_cast<std::int64_t>(exposure.Step().count());
            }
        } catch (...) {
            // Some drivers expose this only for particular active streams.
        }

        std::scoped_lock state_lock(state_mutex);
        controls = observed;

    } catch (...) {
        // Controls are optional telemetry/optimization and must not take down
        // the GPU capture path.
    }
}

// Frame callback ---------------------------------------------------------------

void Camera::Impl::register_frame_handler() {
    frame_token = reader.FrameArrived(
        [this](
            MediaFrameReader const& sender,
            winrt::Windows::Media::Capture::Frames::MediaFrameArrivedEventArgs const&
        ) {
            on_frame_arrived(sender);
        }
    );
    frame_handler_registered = true;
}

void Camera::Impl::on_frame_arrived(MediaFrameReader const& sender) noexcept {
    if (!running_flag.load(std::memory_order_acquire)) {
        return;
    }

    const auto callback_entry_ns = steady_now_ns();

    try {
        auto frame_reference = sender.TryAcquireLatestFrame();

        if (!frame_reference) {
            record_failure(
                "TryAcquireLatestFrame() tidak mengembalikan frame"
            );
            return;
        }

        std::uint64_t source_timestamp_ns = 0;
        if (auto system_time = frame_reference.SystemRelativeTime()) {
            auto source_ns_signed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    system_time.Value()
                ).count();

            if (source_ns_signed > 0) {
                source_timestamp_ns = static_cast<std::uint64_t>(
                    source_ns_signed
                );
            }
        }

        auto video = frame_reference.VideoMediaFrame();

        if (!video) {
            record_failure(
                "Frame bukan VideoMediaFrame yang dapat dipakai"
            );
            return;
        }

        auto surface = video.Direct3DSurface();

        if (!surface) {
            // The production path intentionally has no SoftwareBitmap fallback.
            record_failure(
                "Direct3DSurface null; GPU-backed capture tidak tersedia"
            );
            return;
        }

        auto texture = texture_from_surface(surface);

        // Width, height, and format are immutable for an active camera mode, so
        // query GetDesc() only for the first GPU frame.
        if (!texture_metadata_ready) {
            D3D11_TEXTURE2D_DESC desc{};
            texture->GetDesc(&desc);

            std::scoped_lock state_lock(state_mutex);
            if (!texture_metadata_ready) {
                texture_width = desc.Width;
                texture_height = desc.Height;
                texture_format = desc.Format;
                texture_metadata_ready = true;
            }
        }

        GpuFrame frame;
        frame.texture = std::move(texture);
        frame.frame_id = current_frame_id.fetch_add(
            1,
            std::memory_order_relaxed
        ) + 1;
        frame.callback_entry_ns = callback_entry_ns;
        frame.source_timestamp_ns = source_timestamp_ns;
        frame.timestamp_ns = steady_now_ns();

        {
            std::scoped_lock state_lock(state_mutex);
            frame.width = texture_width;
            frame.height = texture_height;
            frame.dxgi_format = texture_format;
            latest = std::move(frame);
        }

        consecutive_failures.store(0, std::memory_order_relaxed);
        state_cv.notify_one();

    } catch (...) {
        record_failure(
            "Exception saat menerima GPU camera frame"
        );
    }
}

void Camera::Impl::record_failure(std::string const& message) noexcept {
    failed_reads_count.fetch_add(1, std::memory_order_relaxed);

    auto failures = consecutive_failures.fetch_add(
        1,
        std::memory_order_relaxed
    ) + 1;

    if (failures < kMaxConsecutiveFailures) {
        return;
    }

    try {
        auto exception = std::make_exception_ptr(
            std::runtime_error(
                message + " (30 kegagalan berturut-turut)"
            )
        );

        {
            std::scoped_lock state_lock(state_mutex);
            if (!error) {
                error = exception;
            }
        }

        stop_requested.store(true, std::memory_order_release);
        state_cv.notify_all();
        control_cv.notify_all();

    } catch (...) {
        // MediaFrameReader callbacks must never leak an exception.
    }
}

void Camera::Impl::cleanup_capture() {
    if (reader) {
        if (frame_handler_registered) {
            reader.FrameArrived(frame_token);
            frame_handler_registered = false;
        }

        try {
            reader.StopAsync().get();
        } catch (...) {
            // The device may already be gone; continue cleanup.
        }

        reader.Close();
        reader = nullptr;
    }

    source = nullptr;

    if (capture) {
        capture.Close();
        capture = nullptr;
    }

    // Restore only after Media Foundation releases the device so the legacy
    // IAMCameraControl adapter cannot compete for exclusive control.
    restore_legacy_camera_control();

    selected_friendly_name.clear();
    selected_group = nullptr;
}

} // namespace vision_runtime
