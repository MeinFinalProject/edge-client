#pragma once
#include "pipeline/biometric_attendance_runtime.hpp"
#include <functional>

namespace vision_runtime::pipeline
{
struct BiometricEvaluators
{
    std::function<std::vector<liveness::LivenessResult>(GpuFrame const &, std::vector<FaceDetection> const &)> pad;
    std::function<std::vector<arcface::ArcFaceResult>(GpuFrame const &, std::vector<FaceDetection> const &)> recognize;
};
// Shared by camera runtime and interactive enrollment demo. Tests inject model
// observations, but execute the same scheduling/voting/matching implementation.
class AttendanceProcessor
{
  public:
    using Time = std::chrono::steady_clock::time_point;
    AttendanceProcessor(BiometricAttendanceConfig config, BiometricEvaluators evaluators);
    ~AttendanceProcessor();
    void set_gallery(std::vector<GalleryTemplate> const &templates);
    std::size_t gallery_size() const;
    void reset(); // clear track evidence; retain anti-flood cooldown
    BiometricAttendanceFrame process(FaceTrackingFrame const &frame, Time now = std::chrono::steady_clock::now());

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
void validate_attendance_config(BiometricAttendanceConfig const &config);
} // namespace vision_runtime::pipeline
