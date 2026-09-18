function(add_vision_tool target source)
    add_executable(${target} ${source})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target}
        PRIVATE
            WIN32_LEAN_AND_MEAN
            NOMINMAX
            UNICODE
            _UNICODE
    )
    target_link_libraries(${target}
        PRIVATE
            vision_runtime
            windowsapp
    )
endfunction()

add_vision_tool(camera_probe camera_probe.cpp)
add_vision_tool(scrfd_probe scrfd_probe.cpp)
add_vision_tool(bytetrack_probe bytetrack_probe.cpp)


add_vision_tool(liveness_probe liveness_probe.cpp)
add_vision_tool(liveness_capture liveness_capture.cpp)
target_link_libraries(liveness_capture PRIVATE bcrypt)

add_vision_tool(arcface_probe arcface_probe.cpp)

add_vision_tool(ta_demo ta_demo.cpp)
target_link_libraries(ta_demo
    PRIVATE
        d3d11
        dxgi
        user32
        gdi32
)

add_vision_tool(env_probe env_probe.cpp)
target_link_libraries(env_probe
    PRIVATE
        dxgi
)

add_vision_tool(model_inspect model_inspect.cpp)

add_vision_tool(edge_client edge_client.cpp)
target_link_libraries(edge_client PRIVATE edge_app_core)

add_vision_tool(biometric_benchmark biometric_benchmark.cpp)
