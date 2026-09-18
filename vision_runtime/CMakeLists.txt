add_library(vision_runtime STATIC
    camera/camera.cpp
    camera/directshow_low_light_control.cpp
    camera/format_selector.cpp

    scrfd/detector.cpp
    scrfd/gpu_preprocessor.cpp
    scrfd/scrfd.cpp

    bytetrack/kalman_filter.cpp
    bytetrack/assignment.cpp
    bytetrack/association.cpp
    bytetrack/bytetrack.cpp

    liveness/gpu_preprocessor.cpp
    liveness/liveness.cpp

    arcface/gpu_preprocessor.cpp
    arcface/arcface.cpp

    pipeline/face_tracking_runtime.cpp
    pipeline/biometric_attendance_runtime.cpp
    pipeline/attendance_processor.cpp
    pipeline/gallery_matcher.cpp
)


target_include_directories(vision_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(vision_runtime PUBLIC cxx_std_20)

target_compile_definitions(vision_runtime
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
)

target_link_libraries(vision_runtime
    PUBLIC
        windowsapp
        d3d11
        d3d12
        dxgi
    PRIVATE
        d3dcompiler
        strmiids
        ole32
        oleaut32
)
