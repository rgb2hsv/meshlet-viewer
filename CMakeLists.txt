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
compile_shader(lod_traverse.comp comp kLodTraverseSpv)
compile_shader(lod_finalize.comp comp kLodFinalizeSpv)

add_custom_target(shaders DEPENDS ${SHADER_HEADERS})

# --- meshlets --------------------------------------------------------------
add_executable(meshlets
    src/main.cpp
    src/vk_context.cpp
    src/model.cpp
    src/renderer.cpp
    src/matcap.cpp
    src/meshlet_file.cpp
    src/meshlet_optimizer.cpp
    src/lod_palette.cpp
    ${SHADER_HEADERS}
)
add_dependencies(meshlets shaders)
target_compile_definitions(meshlets PRIVATE MESHLETS_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")
target_include_directories(meshlets PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${SHADER_OUT_DIR})
target_include_directories(meshlets SYSTEM PRIVATE ${EXTERNAL_DIR}/stb)   # stb_image (matcap PNGs)
target_include_directories(meshlets SYSTEM PRIVATE ${EXTERNAL_DIR}/eigen) # Eigen (header-only vector math)
target_include_directories(meshlets SYSTEM PRIVATE ${EXTERNAL_DIR}/cereal/include) # cereal (header-only serialization)
target_link_libraries(meshlets PRIVATE meshoptimizer argparse::argparse imgui::imgui glfw volk::volk Threads::Threads)

# --- meshlet-optimizer -------------------------------------------------------
# Built as its own project (C++20, OpenMP, its own dependencies) and run by the viewer as a child
# process: the "meshlet-optimizer" LOD source writes the mesh to a temporary PLY, runs the tool and
# loads the .mlod it writes. Release even in debug viewer builds: the hierarchy build is heavy.
set(MESHLET_OPTIMIZER_DIR ${EXTERNAL_DIR}/meshlet-optimizer CACHE PATH "meshlet-optimizer source tree")
if(EXISTS ${MESHLET_OPTIMIZER_DIR}/CMakeLists.txt)
    include(ExternalProject)
    set(MOPT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/meshlet-optimizer)
    if(CMAKE_CONFIGURATION_TYPES)
        set(MOPT_EXE ${MOPT_BINARY_DIR}/Release/meshlet-optimizer${CMAKE_EXECUTABLE_SUFFIX})
    else()
        set(MOPT_EXE ${MOPT_BINARY_DIR}/meshlet-optimizer${CMAKE_EXECUTABLE_SUFFIX})
    endif()
    ExternalProject_Add(meshlet_optimizer
        SOURCE_DIR ${MESHLET_OPTIMIZER_DIR}
        BINARY_DIR ${MOPT_BINARY_DIR}
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target meshlet-optimizer
        BUILD_BYPRODUCTS ${MOPT_EXE}
        INSTALL_COMMAND ""
        BUILD_ALWAYS ON
        USES_TERMINAL_BUILD ON)
    add_dependencies(meshlets meshlet_optimizer)
    target_compile_definitions(meshlets PRIVATE MESHLET_OPTIMIZER_EXE="${MOPT_EXE}")
else()
    message(WARNING "${MESHLET_OPTIMIZER_DIR} is missing: the meshlet-optimizer LOD source is unavailable "
                    "(run: git submodule update --init --recursive)")
endif()

if(MSVC)
    target_compile_options(meshlets PRIVATE /W4)
    # M_PI etc. from <cmath>; silence fopen/sscanf "unsafe" deprecation warnings
    target_compile_definitions(meshlets PRIVATE _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(meshlets PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
endif()
