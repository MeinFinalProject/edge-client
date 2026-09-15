#include "camera/camera.hpp"

#include <chrono>
#include <iostream>
#include <type_traits>

namespace {

int failures = 0;

void expect(bool condition, char const* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

} // namespace

int main() {
    using namespace std::chrono_literals;
    using vision_runtime::Camera;
    using vision_runtime::CameraAcquisitionMode;
    using vision_runtime::CameraStreamPreference;

    static_assert(!std::is_copy_constructible_v<Camera>);
    static_assert(!std::is_copy_assignable_v<Camera>);
    static_assert(!std::is_move_constructible_v<Camera>);
    static_assert(!std::is_move_assignable_v<Camera>);

    Camera camera;
    auto config = camera.config();

    expect(config.index == 0, "default camera index");
    expect(config.width == 1280, "default camera width");
    expect(config.height == 720, "default camera height");
    expect(config.fps == 30, "default camera FPS");
    expect(
        config.acquisition_mode == CameraAcquisitionMode::Realtime,
        "default acquisition mode"
    );
    expect(
        config.stream_preference == CameraStreamPreference::Auto,
        "default stream preference"
    );
    expect(config.source_id.empty(), "default source ID");
    expect(config.source_subtype.empty(), "default source subtype");
    expect(config.output_subtype == L"NV12", "default output subtype");
    expect(config.force_constant_framerate, "constant frame-rate default");
    expect(
        config.restore_legacy_control_on_stop,
        "legacy control restore default"
    );

    expect(!camera.running(), "new camera is stopped");
    expect(camera.frame_id() == 0, "new camera frame ID");
    expect(camera.failed_reads() == 0, "new camera failed-read count");
    expect(
        camera.acquisition_mode() == CameraAcquisitionMode::Realtime,
        "facade acquisition mode"
    );
    expect(
        camera.stream_preference() == CameraStreamPreference::Auto,
        "facade stream preference"
    );

    auto mode = camera.mode();
    expect(mode.width == 0 && mode.height == 0, "inactive camera mode");

    auto frame = camera.wait_for_frame(0, 50ms);
    expect(!frame.has_value(), "stopped camera returns no frame");

    // stop() is intentionally safe before start and on repeated calls.
    camera.stop();
    camera.stop();
    expect(!camera.running(), "repeated stop remains stopped");

    if (failures != 0) {
        std::cerr << failures << " camera contract test(s) failed\n";
        return 1;
    }

    std::cout << "camera contract tests passed\n";
    return 0;
}
