#include "pipeline/attendance_processor.hpp"
#include <stdexcept>

namespace vision_runtime::pipeline
{
struct BiometricAttendanceRuntime::Impl
{
    static BiometricAttendanceConfig checked(BiometricAttendanceConfig c)
    {
        validate_attendance_config(c);
        return c;
    }
    BiometricAttendanceConfig cfg;
    FaceTrackingRuntime tracking;
    liveness::LivenessDetector pad;
    arcface::ArcFaceRecognizer recognizer;
    AttendanceProcessor processor;
    bool started = false;
    explicit Impl(BiometricAttendanceConfig c)
        : cfg(checked(std::move(c))), tracking(cfg.tracking), pad(cfg.pad), recognizer(cfg.arcface),
          processor(cfg,
                    {[this](GpuFrame const &f, std::vector<FaceDetection> const &o) { return pad.evaluate_many(f, o); },
                     [this](GpuFrame const &f, std::vector<FaceDetection> const &o) {
                         return recognizer.evaluate_many(f, o);
                     }})
    {
    }
};
BiometricAttendanceRuntime::BiometricAttendanceRuntime(BiometricAttendanceConfig c)
    : impl_(std::make_unique<Impl>(std::move(c)))
{
}
BiometricAttendanceRuntime::~BiometricAttendanceRuntime()
{
    stop();
}
void BiometricAttendanceRuntime::set_gallery(std::vector<GalleryTemplate> g)
{
    impl_->processor.set_gallery(g);
}
std::size_t BiometricAttendanceRuntime::gallery_size() const noexcept
{
    return impl_->processor.gallery_size();
}
BiometricAttendanceRuntime &BiometricAttendanceRuntime::start()
{
    if (!impl_->started)
    {
        impl_->processor.reset();
        impl_->tracking.start();
        impl_->started = true;
    }
    return *this;
}
void BiometricAttendanceRuntime::stop() noexcept
{
    if (impl_ && impl_->started)
    {
        impl_->tracking.stop();
        impl_->started = false;
    }
}
std::optional<BiometricAttendanceFrame> BiometricAttendanceRuntime::wait_for_frame(std::chrono::milliseconds timeout)
{
    if (!impl_->started)
        throw std::logic_error("Runtime not started");
    auto f = impl_->tracking.wait_for_frame(timeout);
    if (!f)
        return std::nullopt;
    return impl_->processor.process(*f);
}
void BiometricAttendanceRuntime::reset_session()
{
    impl_->processor.reset();
    impl_->tracking.reset_tracker();
}
bool BiometricAttendanceRuntime::running() const noexcept
{
    return impl_->started;
}
BiometricAttendanceConfig BiometricAttendanceRuntime::config() const
{
    return impl_->cfg;
}
} // namespace vision_runtime::pipeline
