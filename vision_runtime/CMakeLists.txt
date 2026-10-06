# Cari program fxc (DirectX Shader Compiler)
find_program(FXC_EXECUTABLE fxc
    HINTS
        "$ENV{WindowsSdkVerBinPath}/x64"
        "$ENV{WindowsSdkDir}/bin/$ENV{WindowsSDKVersion}/x64"
    REQUIRED)

# Fungsi untuk menambahkan file header HLSL ke dalam proyek CMake.
function(add_hlsl_header HLSL_FILE OUT_HEADER VAR_NAME ENTRY)
    # Ambil direktori dari file header output.
    get_filename_component(out_dir "${OUT_HEADER}" DIRECTORY)
    # Buat perintah kustom untuk mengkompilasi file HLSL menjadi header C++.
    add_custom_command(
        OUTPUT "${OUT_HEADER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${out_dir}"
        COMMAND "${FXC_EXECUTABLE}" /nologo /T cs_5_0 /E ${ENTRY} /O3 /WX
                /Fh "${OUT_HEADER}" /Vn ${VAR_NAME} "${HLSL_FILE}"
        DEPENDS "${HLSL_FILE}"
        COMMENT "fxc: ${HLSL_FILE} [${ENTRY}]"
        VERBATIM)
endfunction()

set(SHADER_HEADER                 "${CMAKE_CURRENT_BINARY_DIR}/scrfd/scrfd_tensorize.h")
set(SHADER_HEADER_ARCFACE         "${CMAKE_CURRENT_BINARY_DIR}/arcface/arcface_tensorize.h")
set(SHADER_HEADER_LIVENESS        "${CMAKE_CURRENT_BINARY_DIR}/liveness/liveness_tensorize.h")
set(SHADER_HEADER_LIVENESS_REDUCE "${CMAKE_CURRENT_BINARY_DIR}/liveness/liveness_coverage_reduce.h")

add_hlsl_header("${CMAKE_CURRENT_SOURCE_DIR}/scrfd/scrfd_tensorize.hlsl"
                "${SHADER_HEADER}" g_scrfd_tensorize main)

add_hlsl_header("${CMAKE_CURRENT_SOURCE_DIR}/arcface/arcface_tensorize.hlsl"
                "${SHADER_HEADER_ARCFACE}" g_arcface_tensorize main)

set(LIVENESS_HLSL "${CMAKE_CURRENT_SOURCE_DIR}/liveness/liveness_tensorize.hlsl")
add_hlsl_header("${LIVENESS_HLSL}" "${SHADER_HEADER_LIVENESS}"
                g_liveness_tensorize main)
add_hlsl_header("${LIVENESS_HLSL}" "${SHADER_HEADER_LIVENESS_REDUCE}"
                g_liveness_coverage_reduce reduce)

# Memerintahkan kompilator untuk mengambil seluruh file .cpp di bawah ini, 
# mengkompilasinya secara paralel, lalu membungkusnya menjadi satu kesatuan Static Library 
# (di Windows berformat .lib).
add_library(vision_runtime STATIC
    camera/camera.cpp
    camera/directshow_low_light_control.cpp
    camera/format_selector.cpp

    scrfd/detector.cpp
    scrfd/gpu_preprocessor.cpp
    ${SHADER_HEADER}
    ${SHADER_HEADER_ARCFACE}
    ${SHADER_HEADER_LIVENESS}
    ${SHADER_HEADER_LIVENESS_REDUCE}
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

# Memberitahu sistem bahwa folder saat ini adalah direktori include. 
# Ini krusial agar file lain di luar folder ini bisa mengakses header menggunakan path yang bersih 
# (misal: #include "scrfd/detector.hpp").
target_include_directories(vision_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
)

# Mengunci standar bahasa yang digunakan di dalam modul ini wajib menggunakan C++20.
target_compile_features(vision_runtime PUBLIC cxx_std_20)

# Menambahkan definisi variabel yang akan digunakan dalam kompilasi.
target_compile_definitions(vision_runtime
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
)

# Menyambungkan dependensi sistem bawaan Windows yang dibutuhkan modul computer vision ini 
# (seperti d3d11, d3d12, dan dxgi untuk akselerasi GPU/DirectML).
target_link_libraries(vision_runtime
    # PUBLIC (PROPAGASI AKTIF)
    # Deklarasi: "Saya butuh library ini, dan SIAPAPUN yang nanti memakai vision_runtime 
    # (misalnya modul edge_app) juga akan OTOMATIS diwariskan library ini."
    PUBLIC
        windowsapp
        d3d11
        d3d12
        dxgi
    # PRIVATE (ENKAPSULASI INTERNAL)
    # Deklarasi: "Saya butuh library ini HANYA untuk kebutuhan dapur internal saya. 
    # Modul luar yang memakai vision_runtime TIDAK akan mewarisinya."
    PRIVATE
        strmiids
        ole32
        oleaut32
)
