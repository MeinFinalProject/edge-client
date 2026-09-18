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

add_executable(attendance_processor_test attendance/attendance_processor_test.cpp)
target_compile_features(attendance_processor_test PRIVATE cxx_std_20)
target_compile_definitions(attendance_processor_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_link_libraries(attendance_processor_test PRIVATE vision_runtime)
add_test(NAME attendance.processor COMMAND attendance_processor_test)

function(add_edge_test target source test_name)
    add_executable(${target} ${source})
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE edge_app_core)
    add_test(NAME ${test_name} COMMAND ${target})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
endfunction()
add_edge_test(edge_repositories_test storage/repositories_test.cpp edge.storage)
add_edge_test(edge_credential_test security/device_credential_test.cpp edge.credential)
add_edge_test(edge_wire_format_test sync/wire_format_test.cpp edge.wire_format)
add_edge_test(attendance_coordinator_test attendance/attendance_coordinator_test.cpp attendance.coordinator)
add_edge_test(edge_sync_service_test sync/sync_service_test.cpp edge.sync)
add_edge_test(edge_offline_recovery_test failure/offline_recovery_test.cpp edge.failure)
add_edge_test(edge_http_test sync/http_client_test.cpp edge.http_loopback)
target_link_libraries(edge_http_test PRIVATE ws2_32)

add_executable(liveness_gpu_preprocess_test liveness_gpu_preprocess_test.cpp)
target_compile_features(liveness_gpu_preprocess_test PRIVATE cxx_std_20)
target_compile_definitions(liveness_gpu_preprocess_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_link_libraries(liveness_gpu_preprocess_test PRIVATE vision_runtime)
option(VISION_RUNTIME_GPU_TESTS "Register tests requiring a hardware D3D11/D3D12 adapter" OFF)
if(VISION_RUNTIME_GPU_TESTS)
    add_test(NAME liveness.gpu_preprocess COMMAND liveness_gpu_preprocess_test)
    set_tests_properties(liveness.gpu_preprocess PROPERTIES LABELS gpu TIMEOUT 60)
endif()
