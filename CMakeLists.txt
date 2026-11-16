cmake_minimum_required(VERSION 3.16)

project(meshlets VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

if(NOT EXISTS ${EXTERNAL_DIR}/meshoptimizer/CMakeLists.txt)
    message(FATAL_ERROR "Submodules missing; run: git submodule update --init --recursive")
endif()

# --- meshoptimizer ---------------------------------------------------------
set(MESHOPT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/meshoptimizer EXCLUDE_FROM_ALL)

# --- argparse (header-only) ------------------------------------------------
set(ARGPARSE_INSTALL OFF CACHE BOOL "" FORCE)
set(ARGPARSE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ARGPARSE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/argparse EXCLUDE_FROM_ALL)

# --- Vulkan headers + volk (loads libvulkan.so.1 at runtime; no SDK needed) -
add_subdirectory(${EXTERNAL_DIR}/vulkan-headers EXCLUDE_FROM_ALL)
set(VOLK_PULL_IN_VULKAN OFF CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/volk EXCLUDE_FROM_ALL)
target_link_libraries(volk PUBLIC Vulkan::Headers)

# --- GLFW ------------------------------------------------------------------
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/glfw EXCLUDE_FROM_ALL)

# --- imgui (core + GLFW/Vulkan backends) ------------------------------------
set(IMGUI_DIR ${EXTERNAL_DIR}/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui SYSTEM PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK GLFW_INCLUDE_NONE)
target_link_libraries(imgui PUBLIC glfw volk::volk)
add_library(imgui::imgui ALIAS imgui)

# --- glslang (host shader compiler, used at build time) ----------------------
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXTERNAL_DIR}/glslang EXCLUDE_FROM_ALL)

# liblava (external/liblava) is checked out but not built: it is a full
# Vulkan framework that fetches its own glfw/volk/imgui via CPM.

# --- shaders: GLSL -> SPIR-V embedded as C arrays ---------------------------
set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})
file(GLOB SHADER_INCLUDES CONFIGURE_DEPENDS ${SHADER_DIR}/*.glsl)
set(SHADER_HEADERS)

# compile_shader(<source> <stage> <variable name> [defines...])
function(compile_shader source stage name)
    set(output ${SHADER_OUT_DIR}/${name}.h)
    set(defines)
    foreach(def ${ARGN})
        list(APPEND defines -D${def})
    endforeach()
    add_custom_command(
        OUTPUT ${output}
        COMMAND glslang-standalone -V --target-env vulkan1.3 -g -S ${stage} ${defines}
                -I${SHADER_DIR} --vn ${name} -o ${output} ${SHADER_DIR}/${source}
        DEPENDS ${SHADER_DIR}/${source} ${SHADER_INCLUDES} glslang-standalone
        COMMENT "Compiling ${source} -> ${name}"
        VERBATIM)
    set(SHADER_HEADERS ${SHADER_HEADERS} ${output} PARENT_SCOPE)
endfunction()

compile_shader(meshlet.task task kMeshletTaskSpv)
compile_shader(meshlet.mesh mesh kMeshletMeshSpv)
compile_shader(meshlet.vert vert kMeshletVertSpv)
compile_shader(meshlet.frag frag kMeshletFragMeshSpv       MESH_PIPELINE=1 BARYCENTRIC=1)
compile_shader(meshlet.frag frag kMeshletFragMeshNoBarySpv MESH_PIPELINE=1 BARYCENTRIC=0)
compile_shader(meshlet.frag frag kMeshletFragVertSpv       MESH_PIPELINE=0 BARYCENTRIC=0)

add_custom_target(shaders DEPENDS ${SHADER_HEADERS})

# --- meshlets --------------------------------------------------------------
add_executable(meshlets
    src/main.cpp
    src/vk_context.cpp
    src/model.cpp
    src/renderer.cpp
    src/matcap.cpp
    ${SHADER_HEADERS}
)
add_dependencies(meshlets shaders)
target_compile_definitions(meshlets PRIVATE MESHLETS_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
target_include_directories(meshlets PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${SHADER_OUT_DIR})
target_include_directories(meshlets SYSTEM PRIVATE ${EXTERNAL_DIR}/stb) # stb_image (matcap PNGs)
target_link_libraries(meshlets PRIVATE meshoptimizer argparse::argparse imgui::imgui glfw volk::volk Threads::Threads)

if(MSVC)
    target_compile_options(meshlets PRIVATE /W4)
else()
    target_compile_options(meshlets PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
endif()
