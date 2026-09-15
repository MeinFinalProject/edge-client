add_executable(camera_format_selector_test
    camera_format_selector_test.cpp
)

target_compile_features(camera_format_selector_test PRIVATE cxx_std_20)
target_compile_definitions(camera_format_selector_test
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
)
target_link_libraries(camera_format_selector_test PRIVATE vision_runtime)

add_test(NAME camera.format_selector COMMAND camera_format_selector_test)
set_tests_properties(camera.format_selector PROPERTIES LABELS unit)

add_executable(camera_contract_test
    camera_contract_test.cpp
)

target_compile_features(camera_contract_test PRIVATE cxx_std_20)
target_compile_definitions(camera_contract_test
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
)
target_link_libraries(camera_contract_test PRIVATE vision_runtime)

add_test(NAME camera.contract COMMAND camera_contract_test)
set_tests_properties(camera.contract PROPERTIES LABELS unit)


add_executable(liveness_contract_test liveness_contract_test.cpp)
target_compile_features(liveness_contract_test PRIVATE cxx_std_20)
target_compile_definitions(liveness_contract_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_link_libraries(liveness_contract_test PRIVATE vision_runtime)
add_test(NAME liveness.contract COMMAND liveness_contract_test)
set_tests_properties(liveness.contract PROPERTIES LABELS unit)

add_executable(liveness_gpu_preprocess_test liveness_gpu_preprocess_test.cpp)
target_compile_features(liveness_gpu_preprocess_test PRIVATE cxx_std_20)
target_compile_definitions(liveness_gpu_preprocess_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_link_libraries(liveness_gpu_preprocess_test PRIVATE vision_runtime)
option(VISION_RUNTIME_GPU_TESTS "Register tests requiring a hardware D3D11/D3D12 adapter" OFF)
if(VISION_RUNTIME_GPU_TESTS)
    add_test(NAME liveness.gpu_preprocess COMMAND liveness_gpu_preprocess_test)
    set_tests_properties(liveness.gpu_preprocess PROPERTIES LABELS gpu TIMEOUT 60)
endif()
