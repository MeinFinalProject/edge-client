# One native library; folders separate responsibilities, not processes or threads.
add_library(edge_app_core STATIC
    app/application_config.cpp
    app/attendance_coordinator.cpp
    app/edge_application.cpp
    domain/gallery_snapshot.cpp
    domain/uuid_v7.cpp
    security/device_credential.cpp
    security/model_hash.cpp
    storage/database.cpp
    storage/gallery_repository.cpp
    storage/outbox_repository.cpp
    sync/attendance_sync.cpp
    sync/gallery_sync.cpp
    sync/http_client.cpp
    sync/network_watcher.cpp
    sync/retry_policy.cpp
    sync/sync_service.cpp
    sync/sync_worker.cpp
    sync/wire_format.cpp
)
target_compile_features(edge_app_core PUBLIC cxx_std_20)
target_include_directories(edge_app_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(edge_app_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(edge_app_core PUBLIC vision_runtime PRIVATE winsqlite3 bcrypt crypt32 windowsapp)
