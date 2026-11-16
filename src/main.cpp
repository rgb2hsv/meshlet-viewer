// meshlets: load a PLY/OBJ model, split it into meshlets with meshoptimizer and
// visualize them with Vulkan mesh shaders (vertex-shader fallback) + Dear ImGui.
#include "matcap.h"
#include "math.h"
#include "model.h"
#include "renderer.h"
#include "vk_context.h"

#include <argparse/argparse.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cstring>
#include <future>
#include <memory>
#include <string>

namespace
{

using Clock = std::chrono::steady_clock;

std::string g_droppedPath;

void checkVkResult(VkResult err)
{
    if (err < 0)
    {
        std::fprintf(stderr, "imgui vulkan error %d\n", int(err));
        std::abort();
    }
}

std::string formatCount(size_t n)
{
    char buf[32];
    if (n >= 1000000)
        std::snprintf(buf, sizeof(buf), "%.2fM", double(n) / 1e6);
    else if (n >= 10000)
        std::snprintf(buf, sizeof(buf), "%.1fK", double(n) / 1e3);
    else
        std::snprintf(buf, sizeof(buf), "%zu", n);
    return buf;
}

std::string fileName(const std::string& path)
{
    size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// --------------------------------------------------------------------------
// Orbit camera
// --------------------------------------------------------------------------
struct Camera
{
    vec3 target;
    float distance = 1.f;
    float yaw = 0.6f;
    float pitch = 0.3f;
    float radius = 1.f; // scene radius, for near plane / speeds
    float fovY = 0.8f;
    bool zUp = false;

    vec3 up() const { return zUp ? vec3{0, 0, 1} : vec3{0, 1, 0}; }

    vec3 eye() const
    {
        float cp = std::cos(pitch), sp = std::sin(pitch);
        vec3 dir = zUp ? vec3{cp * std::cos(yaw), cp * std::sin(yaw), sp} : vec3{cp * std::sin(yaw), sp, cp * std::cos(yaw)};
        return target + dir * distance;
    }

    void basis(vec3& right, vec3& camUp, vec3& back) const
    {
        vec3 forward = normalize(target - eye());
        right = normalize(cross(forward, up()));
        camUp = cross(right, forward);
        back = forward * -1.f;
    }

    // Light from the upper left, slightly behind the viewer
    vec3 lightDir() const
    {
        vec3 forward = normalize(target - eye());
        vec3 right = normalize(cross(forward, up()));
        vec3 camUp = cross(right, forward);
        return normalize(forward * -0.6f + right * -0.5f + camUp * 0.7f);
    }

    void frame(const Model& model)
    {
        vec3 lo = {model.boundsMin[0], model.boundsMin[1], model.boundsMin[2]};
        vec3 hi = {model.boundsMax[0], model.boundsMax[1], model.boundsMax[2]};
        target = (lo + hi) * 0.5f;
        radius = std::max(length(hi - lo) * 0.5f, 1e-6f);
        distance = radius / std::sin(fovY * 0.5f) * 1.05f;
    }

    mat4 viewProj(float aspect) const
    {
        return perspectiveReverseZ(fovY, aspect, radius * 0.0005f) * lookAt(eye(), target, up());
    }

    void update(const ImGuiIO& io)
    {
        if (io.WantCaptureMouse)
            return;
        const float dx = io.MouseDelta.x, dy = io.MouseDelta.y;
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.KeyShift)
        {
            yaw -= dx * 0.005f;
            pitch = std::clamp(pitch + dy * 0.005f, -1.55f, 1.55f);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) || (ImGui::IsMouseDown(ImGuiMouseButton_Left) && io.KeyShift))
        {
            vec3 forward = normalize(target - eye());
            vec3 right = normalize(cross(forward, up()));
            vec3 camUp = cross(right, forward);
            float scale = distance * 0.0015f;
            target = target - right * (dx * scale) + camUp * (dy * scale);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right))
            distance *= std::exp(dy * 0.005f);
        if (io.MouseWheel != 0)
            distance *= std::exp(-io.MouseWheel * 0.1f);
        distance = std::clamp(distance, radius * 0.001f, radius * 100.f);
    }
};

// --------------------------------------------------------------------------
// Background jobs (loading + meshlet building can take seconds on big scans)
// --------------------------------------------------------------------------
struct JobResult
{
    std::shared_ptr<Model> model; // non-null when a new model was loaded
    std::shared_ptr<MeshletData> meshlets;
    std::string error;
};

struct App
{
    VulkanContext ctx;
    MeshletRenderer renderer;
    Camera camera;

    std::shared_ptr<Model> model;
    std::shared_ptr<MeshletData> meshlets;
    MeshletSettings settings;
    bool optimizeVertexCache = true;
    bool autoRebuild = true;
    bool rebuildPending = false;
    float initialZoom = 1.f;
    float wireWidthOverride = 0.f; // from --wire-width; 0 = automatic
    static constexpr float kAutoWireFraction = 0.1f; // default width = 10% of the average edge
    bool firstLoad = true;
    bool loadFailed = false; // a load failed and no model is available

    std::future<JobResult> job;
    std::string jobLabel;
    Clock::time_point jobStart;
    std::string status;
    bool statusIsError = false;

    char pathInput[1024] = {};

    DrawParams draw;
    std::vector<ImTextureID> matcapTextures; // ImGui handles for the selector thumbnails
    FrameStats stats;

    bool busy() const { return job.valid(); }

    void startLoad(const std::string& path)
    {
        if (busy())
            return;
        std::snprintf(pathInput, sizeof(pathInput), "%s", path.c_str());
        jobLabel = "Loading " + fileName(path);
        jobStart = Clock::now();
        MeshletSettings s = settings;
        bool vcache = optimizeVertexCache;
        job = std::async(std::launch::async, [path, s, vcache] {
            JobResult result;
            auto loaded = std::make_shared<Model>();
            if (!loadModel(path, *loaded, result.error, vcache))
                return result;
            auto built = std::make_shared<MeshletData>();
            if (!buildMeshlets(*loaded, s, *built, result.error))
                return result;
            result.model = loaded;
            result.meshlets = built;
            return result;
        });
    }

    void startBuild()
    {
        if (!model)
            return;
        if (busy())
        {
            rebuildPending = true;
            return;
        }
        rebuildPending = false;
        jobLabel = "Building meshlets";
        jobStart = Clock::now();
        std::shared_ptr<const Model> m = model;
        MeshletSettings s = settings;
        job = std::async(std::launch::async, [m, s] {
            JobResult result;
            auto built = std::make_shared<MeshletData>();
            if (buildMeshlets(*m, s, *built, result.error))
                result.meshlets = built;
            return result;
        });
    }

    void pollJob()
    {
        if (!busy() || job.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return;

        JobResult result = job.get();
        if (!result.error.empty())
        {
            status = result.error;
            statusIsError = true;
            std::fprintf(stderr, "error: %s\n", result.error.c_str());
            loadFailed = !model;
        }
        else
        {
            if (result.model)
            {
                model = result.model;
                camera.frame(*model);
                camera.distance /= initialZoom;
                draw.wireWidth = firstLoad && wireWidthOverride > 0.f ? wireWidthOverride : model->avgEdgeLength * kAutoWireFraction;
                if (!firstLoad)
                    draw.highlight = -1; // keep --highlight for the first model
                firstLoad = false;
                glfwSetWindowTitle(ctx.window, ("meshlets - " + fileName(model->path)).c_str());
            }
            meshlets = result.meshlets;
            draw.highlight = std::min(draw.highlight, int(meshlets->meshlets.size()) - 1);
            renderer.upload(*model, *meshlets);

            const MeshletData& m = *meshlets;
            const char* algorithmNames[] = {"greedy", "flex", "spatial", "scan"};
            std::printf("%s: %zu vertices, %zu triangles, avg edge %.4g | %s v%d t%d -> %zu meshlets, avg %.1f verts / %.1f tris, "
                        "cone-cullable %.0f%%, build %.2fs\n",
                fileName(model->path).c_str(), model->vertices.size(), model->indices.size() / 3, model->avgEdgeLength, algorithmNames[int(settings.algorithm)],
                settings.maxVertices, settings.maxTriangles, m.meshlets.size(), m.avgVertices, m.avgTriangles,
                100.0 * double(m.coneCullable) / double(m.meshlets.size()), m.buildSeconds);
            std::fflush(stdout);

            char buf[256];
            std::snprintf(buf, sizeof(buf), "%s meshlets in %.2fs", formatCount(m.meshlets.size()).c_str(), m.buildSeconds);
            status = buf;
            statusIsError = false;
        }

        if (rebuildPending)
            startBuild();
    }

    // Upper bounds the UI allows for meshlet sizes on this device
    int maxVertexLimit() const
    {
        int limit = 256; // meshoptimizer + MAX_MESHLET_VERTICES in common.glsl
        if (renderer.usesMeshShaders())
            limit = std::min(limit, int(ctx.caps.maxMeshOutputVertices));
        return limit;
    }
    int maxTriangleLimit() const
    {
        int limit = 256; // MAX_MESHLET_TRIANGLES in common.glsl
        if (renderer.usesMeshShaders())
            limit = std::min(limit, int(ctx.caps.maxMeshOutputPrimitives));
        return limit;
    }

    void ui();
};

void App::ui()
{
    const ImGuiIO& io = ImGui::GetIO();

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(380, 0), ImVec2(380, FLT_MAX));
    ImGui::Begin("Meshlets", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    // ---- Model ----
    if (ImGui::CollapsingHeader("Model", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SetNextItemWidth(-60);
        bool enter = ImGui::InputTextWithHint("##path", "path/to/model.ply", pathInput, sizeof(pathInput), ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        ImGui::BeginDisabled(busy());
        if ((ImGui::Button("Load", ImVec2(-1, 0)) || enter) && pathInput[0])
            startLoad(pathInput);
        ImGui::EndDisabled();
        ImGui::TextDisabled("Tip: drag & drop a .ply/.obj onto the window");
        ImGui::Checkbox("Optimize vertex cache on load", &optimizeVertexCache);

        if (model)
        {
            ImGui::Text("%s", fileName(model->path).c_str());
            ImGui::Text("%s vertices, %s triangles (%.2fs)", formatCount(model->vertices.size()).c_str(),
                formatCount(model->indices.size() / 3).c_str(), model->loadSeconds);
            if (ImGui::Checkbox("Z is up", &camera.zUp))
                camera.frame(*model);
            ImGui::SameLine();
            if (ImGui::Button("Reset camera"))
                camera.frame(*model);
        }
    }

    // ---- Meshlet builder ----
    if (ImGui::CollapsingHeader("Meshlet builder", ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool changed = false;
        const char* algorithms[] = {"buildMeshlets (greedy)", "buildMeshletsFlex", "buildMeshletsSpatial (RT)", "buildMeshletsScan (fast)"};
        int algorithm = int(settings.algorithm);
        if (ImGui::Combo("Algorithm", &algorithm, algorithms, IM_ARRAYSIZE(algorithms)))
        {
            settings.algorithm = MeshletAlgorithm(algorithm);
            changed = true;
        }

        ImGui::SliderInt("Max vertices", &settings.maxVertices, 3, maxVertexLimit());
        changed |= ImGui::IsItemDeactivatedAfterEdit();
        ImGui::SliderInt("Max triangles", &settings.maxTriangles, 4, maxTriangleLimit());
        changed |= ImGui::IsItemDeactivatedAfterEdit();

        const bool flex = settings.algorithm == MeshletAlgorithm::Flex;
        const bool spatial = settings.algorithm == MeshletAlgorithm::Spatial;
        if (flex || spatial)
        {
            ImGui::SliderInt("Min triangles", &settings.minTriangles, 1, settings.maxTriangles);
            changed |= ImGui::IsItemDeactivatedAfterEdit();
        }
        if (settings.algorithm == MeshletAlgorithm::Greedy || flex)
        {
            ImGui::SliderFloat("Cone weight", &settings.coneWeight, 0.f, 1.f);
            changed |= ImGui::IsItemDeactivatedAfterEdit();
        }
        if (flex)
        {
            ImGui::SliderFloat("Split factor", &settings.splitFactor, 0.f, 4.f);
            changed |= ImGui::IsItemDeactivatedAfterEdit();
        }
        if (spatial)
        {
            ImGui::SliderFloat("Fill weight", &settings.fillWeight, 0.f, 1.f);
            changed |= ImGui::IsItemDeactivatedAfterEdit();
        }
        changed |= ImGui::Checkbox("optimizeMeshlet", &settings.optimize);

        settings.maxTriangles = std::clamp(settings.maxTriangles, 4, maxTriangleLimit());
        settings.minTriangles = std::clamp(settings.minTriangles, 1, settings.maxTriangles);

        ImGui::Checkbox("Auto rebuild", &autoRebuild);
        ImGui::SameLine();
        ImGui::BeginDisabled(!model);
        if (ImGui::Button("Build now"))
            startBuild();
        ImGui::EndDisabled();
        if (changed && autoRebuild)
            startBuild();

        if (meshlets)
        {
            const MeshletData& m = *meshlets;
            if (ImGui::BeginTable("meshletstats", 2, ImGuiTableFlags_SizingStretchProp))
            {
                auto row = [](const char* label, const char* fmt, auto... args) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%s", label);
                    ImGui::TableNextColumn();
                    ImGui::Text(fmt, args...);
                };
                row("Meshlets", "%s", formatCount(m.meshlets.size()).c_str());
                row("Avg vertices", "%.1f (%.0f%% of max)", m.avgVertices, m.vertexFill * 100.f);
                row("Avg triangles", "%.1f (%.0f%% of max)", m.avgTriangles, m.triangleFill * 100.f);
                row("Vertex refs / vertex", "%.2f", model ? double(m.vertices.size()) / double(model->vertices.size()) : 0.0);
                row("Cone-cullable", "%.0f%% (avg spread %.0f deg)", 100.0 * double(m.coneCullable) / double(m.meshlets.size()), m.avgConeAngleDeg);
                row("Build time", "%.2f s", m.buildSeconds);
                ImGui::EndTable();
            }
        }
    }

    // ---- Rendering ----
    if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::TextDisabled("%s", ctx.caps.props.deviceName);
        ImGui::TextDisabled("%s", renderer.usesMeshShaders() ? "Pipeline: task + mesh shaders (VK_EXT_mesh_shader)"
                                                             : "Pipeline: vertex shader fallback (no culling)");

        const char* modes[] = {"Meshlet", "Triangle", "Normal", "Shaded"};
        int mode = int(draw.mode);
        if (ImGui::Combo("Color", &mode, modes, IM_ARRAYSIZE(modes)))
            draw.mode = ColorMode(mode);

        if (draw.mode == ColorMode::Shaded && renderer.matcapCount() > 0)
        {
            // Blender matcap selector: combo + thumbnail grid
            const char* preview = draw.matcap >= 0 ? renderer.matcapName(size_t(draw.matcap)).c_str() : "none (plain lighting)";
            if (ImGui::BeginCombo("Matcap", preview))
            {
                if (ImGui::Selectable("none (plain lighting)", draw.matcap < 0))
                    draw.matcap = -1;
                for (size_t i = 0; i < renderer.matcapCount(); ++i)
                    if (ImGui::Selectable(renderer.matcapName(i).c_str(), draw.matcap == int(i)))
                        draw.matcap = int(i);
                ImGui::EndCombo();
            }

            const float thumb = 26.f;
            const ImGuiStyle& style = ImGui::GetStyle();
            const float rowWidth = ImGui::GetContentRegionAvail().x;
            float x = 0.f;
            for (size_t i = 0; i < matcapTextures.size(); ++i)
            {
                const float buttonWidth = thumb + style.FramePadding.x * 2.f;
                if (i > 0 && x + style.ItemSpacing.x + buttonWidth <= rowWidth)
                {
                    ImGui::SameLine();
                    x += style.ItemSpacing.x;
                }
                else
                    x = 0.f;
                x += buttonWidth;

                const bool selected = draw.matcap == int(i);
                ImGui::PushID(int(i));
                ImVec4 bg = selected ? ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive) : ImVec4(0, 0, 0, 0);
                if (ImGui::ImageButton("##matcap", matcapTextures[i], ImVec2(thumb, thumb), ImVec2(0, 0), ImVec2(1, 1), bg))
                    draw.matcap = int(i);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", renderer.matcapName(i).c_str());
                ImGui::PopID();
            }
        }

        ImGui::BeginDisabled(!renderer.supportsWireframe());
        ImGui::Checkbox("Wireframe", &draw.wireframe);
        if (!renderer.supportsWireframe() && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("needs VK_KHR_fragment_shader_barycentric");
        ImGui::EndDisabled();
        if (draw.wireframe && model)
        {
            // width is in model units; the slider range follows the mesh resolution
            const float edge = std::max(model->avgEdgeLength, 1e-12f);
            ImGui::SetNextItemWidth(160);
            ImGui::SliderFloat("Line width", &draw.wireWidth, edge * 0.005f, edge * 2.f, "%.3g units", ImGuiSliderFlags_Logarithmic);
            ImGui::SameLine();
            ImGui::TextDisabled("(%.0f%% of avg edge)", 100.f * draw.wireWidth / edge);
        }

        ImGui::BeginDisabled(!renderer.usesMeshShaders());
        ImGui::Checkbox("Frustum culling", &draw.frustumCull);
        ImGui::SameLine();
        ImGui::Checkbox("Cone culling", &draw.coneCull);
        ImGui::EndDisabled();

        ImGui::ColorEdit3("Background", draw.clearColor, ImGuiColorEditFlags_NoInputs);

        if (stats.valid && meshlets)
        {
            ImGui::Text("Visible: %s / %s meshlets", formatCount(stats.visibleMeshlets).c_str(), formatCount(meshlets->meshlets.size()).c_str());
            ImGui::Text("         %s / %s triangles", formatCount(stats.visibleTriangles).c_str(), formatCount(meshlets->triangleCount).c_str());
        }
        ImGui::Text("GPU %.2f ms | CPU %.2f ms (%.0f FPS)", stats.gpuMs, 1000.f / io.Framerate, io.Framerate);
        if (renderer.hasScene())
            ImGui::TextDisabled("GPU scene memory: %.1f MB", double(renderer.gpuBytes()) / (1024.0 * 1024.0));
    }

    // ---- Inspector ----
    if (meshlets && ImGui::CollapsingHeader("Inspect meshlet"))
    {
        int count = int(meshlets->meshlets.size());
        ImGui::SliderInt("Highlight", &draw.highlight, -1, count - 1, draw.highlight < 0 ? "none" : "%d");
        ImGui::SameLine();
        if (ImGui::ArrowButton("##prev", ImGuiDir_Left))
            draw.highlight = std::max(-1, draw.highlight - 1);
        ImGui::SameLine();
        if (ImGui::ArrowButton("##next", ImGuiDir_Right))
            draw.highlight = std::min(count - 1, draw.highlight + 1);

        if (draw.highlight >= 0)
        {
            const meshopt_Meshlet& m = meshlets->meshlets[size_t(draw.highlight)];
            const MeshletBounds& b = meshlets->bounds[size_t(draw.highlight)];
            ImGui::Text("%u vertices, %u triangles", m.vertex_count, m.triangle_count);
            ImGui::Text("sphere (%.3g, %.3g, %.3g) r=%.3g", b.center[0], b.center[1], b.center[2], b.radius);
            if (b.coneCutoff < 1.f)
                ImGui::Text("normal cone spread %.1f deg", 2.0 * std::asin(double(b.coneCutoff)) * 180.0 / M_PI);
            else
                ImGui::Text("normal cone: not cullable");
            if (ImGui::Button("Focus"))
            {
                camera.target = {b.center[0], b.center[1], b.center[2]};
                camera.distance = std::max(b.radius * 4.f, camera.radius * 0.001f);
            }
        }
    }

    // ---- Status ----
    ImGui::Separator();
    if (busy())
    {
        double elapsed = std::chrono::duration<double>(Clock::now() - jobStart).count();
        const char spinner[] = "|/-\\";
        ImGui::Text("%c %s... %.1fs", spinner[int(elapsed * 8) & 3], jobLabel.c_str(), elapsed);
    }
    else if (!status.empty())
        ImGui::TextColored(statusIsError ? ImVec4(1, 0.4f, 0.4f, 1) : ImVec4(0.6f, 0.9f, 0.6f, 1), "%s", status.c_str());

    ImGui::TextDisabled("LMB orbit | Shift+LMB/MMB pan | RMB/wheel zoom");
    ImGui::End();
}

// Copies the swapchain image (in PRESENT_SRC layout) into a host buffer and writes a binary PPM.
void saveScreenshot(VulkanContext& ctx, uint32_t imageIndex, const char* path)
{
    const Swapchain& sc = ctx.swapchain;
    if (!sc.canReadback)
    {
        std::fprintf(stderr, "screenshot: swapchain does not support TRANSFER_SRC\n");
        return;
    }
    VK_CHECK(vkDeviceWaitIdle(ctx.device));

    const uint32_t w = sc.extent.width, h = sc.extent.height;
    Buffer readback = ctx.createBuffer(VkDeviceSize(w) * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);

    VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx.uploadPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd));
    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    VkImage image = sc.images[imageIndex];
    transitionImage(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy region = {};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &region);
    transitionImage(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(ctx.queue));
    vkFreeCommandBuffers(ctx.device, ctx.uploadPool, 1, &cmd);

    const bool bgra = sc.format == VK_FORMAT_B8G8R8A8_UNORM || sc.format == VK_FORMAT_B8G8R8A8_SRGB;
    FILE* file = std::fopen(path, "wb");
    if (file)
    {
        std::fprintf(file, "P6\n%u %u\n255\n", w, h);
        const uint8_t* pixels = static_cast<const uint8_t*>(readback.mapped);
        std::vector<uint8_t> rgb(size_t(w) * h * 3);
        for (size_t i = 0; i < size_t(w) * h; ++i)
        {
            rgb[i * 3 + 0] = pixels[i * 4 + (bgra ? 2 : 0)];
            rgb[i * 3 + 1] = pixels[i * 4 + 1];
            rgb[i * 3 + 2] = pixels[i * 4 + (bgra ? 0 : 2)];
        }
        std::fwrite(rgb.data(), 1, rgb.size(), file);
        std::fclose(file);
        std::printf("screenshot saved to %s\n", path);
    }
    ctx.destroyBuffer(readback);
}

struct Options
{
    std::string input;
    char upAxis = 0; // 0 = auto (Z for the default Lucy model, else Y), 'y' or 'z'
    bool vertexCache = true;

    MeshletSettings meshlets;

    ColorMode colorMode = ColorMode::Meshlet;
    std::string matcap = "clay_studio"; // Blender matcap for --color shaded; "none" = plain lighting
    bool wireframe = false;
    float wireWidth = 0.f; // model units; 0 = automatic
    bool frustumCull = true;
    bool coneCull = true;
    int highlight = -1;
    float zoom = 1.f;

    bool noMeshShaders = false;
    bool validation = false;
    int width = 1600, height = 1000;
    int frames = 0;
    std::string screenshot;
};

bool parseOptions(int argc, char** argv, Options& opt)
{
    argparse::ArgumentParser program("meshlets", "0.1.0");
    program.add_description("Splits a PLY/OBJ model into meshlets with meshoptimizer and visualizes them with Vulkan mesh shaders.");
    program.add_epilog("Example:\n  meshlets assets/lucy.ply --z-up --algorithm flex --max-vertices 128 --max-triangles 128 --min-triangles 64");

    // ---- input ----
    program.add_group("Input");
    program.add_argument("model").nargs(argparse::nargs_pattern::optional).help("model to load (.ply or .obj); defaults to assets/lucy.ply");
    program.add_argument("-i", "--input").metavar("PATH").help("model to load (same as the positional argument)");
    program.add_argument("--z-up").flag().help("treat +Z as up (default for assets/lucy.ply)");
    program.add_argument("--y-up").flag().help("treat +Y as up (default for other models)");
    program.add_argument("--no-vertex-cache").flag().help("skip meshopt_optimizeVertexCache on the index buffer after loading");

    // ---- meshlets ----
    const MeshletSettings defaults;
    program.add_group("Meshlet builder");
    program.add_argument("-a", "--algorithm")
        .default_value(std::string("greedy")).nargs(1)
        .choices("greedy", "flex", "spatial", "scan")
        .help("builder: greedy = buildMeshlets, flex = buildMeshletsFlex, spatial = buildMeshletsSpatial, scan = buildMeshletsScan");
    program.add_argument("--max-vertices").metavar("N").default_value(defaults.maxVertices).nargs(1).scan<'i', int>().help("max vertices per meshlet [3, 256]");
    program.add_argument("--max-triangles").metavar("N").default_value(defaults.maxTriangles).nargs(1).scan<'i', int>().help("max triangles per meshlet [1, 256]");
    program.add_argument("--min-triangles").metavar("N").default_value(defaults.minTriangles).nargs(1).scan<'i', int>().help("min triangles per meshlet (flex, spatial)");
    program.add_argument("--cone-weight").metavar("W").default_value(defaults.coneWeight).nargs(1).scan<'g', float>().help("trade size for cone culling efficiency [0, 1] (greedy, flex)");
    program.add_argument("--split-factor").metavar("F").default_value(defaults.splitFactor).nargs(1).scan<'g', float>().help("split clusters with large bounds, >= 0 (flex)");
    program.add_argument("--fill-weight").metavar("W").default_value(defaults.fillWeight).nargs(1).scan<'g', float>().help("prefer fuller clusters over SAH quality [0, 1] (spatial)");
    program.add_argument("--no-optimize-meshlets").flag().help("skip meshopt_optimizeMeshlet on each meshlet");

    // ---- view ----
    program.add_group("View");
    program.add_argument("-c", "--color").default_value(std::string("meshlet")).nargs(1).choices("meshlet", "triangle", "normal", "shaded").help("color mode");
    program.add_argument("--matcap").metavar("NAME").default_value(std::string("clay_studio")).nargs(1).help("Blender matcap from assets/matcap for --color shaded (file name without .png, or none)");
    program.add_argument("--wireframe").flag().help("draw a wireframe overlay");
    program.add_argument("--wire-width").metavar("W").default_value(0.f).nargs(1).scan<'g', float>().help("wireframe line width in model units (0 = 10% of the average edge length)");
    program.add_argument("--no-frustum-cull").flag().help("disable task-shader frustum culling");
    program.add_argument("--no-cone-cull").flag().help("disable task-shader normal cone (backface) culling");
    program.add_argument("--highlight").metavar("INDEX").default_value(-1).nargs(1).scan<'i', int>().help("highlight one meshlet by index");
    program.add_argument("--zoom").metavar("F").default_value(1.f).nargs(1).scan<'g', float>().help("initial camera zoom factor");

    // ---- app ----
    program.add_group("Application");
    program.add_argument("--no-mesh-shaders").flag().help("force the vertex-shader fallback");
    program.add_argument("--validation").flag().help("enable VK_LAYER_KHRONOS_validation");
    program.add_argument("--width").metavar("PX").default_value(opt.width).nargs(1).scan<'i', int>().help("window width");
    program.add_argument("--height").metavar("PX").default_value(opt.height).nargs(1).scan<'i', int>().help("window height");
    program.add_argument("--frames").metavar("N").default_value(0).nargs(1).scan<'i', int>().help("exit after N frames once the model is ready (0 = run until closed)");
    program.add_argument("--screenshot").metavar("FILE.ppm").default_value(std::string()).nargs(1).help("save the last frame as PPM (use with --frames)");

    auto fail = [&](const std::string& message) {
        std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), program.usage().c_str());
        std::fprintf(stderr, "\nRun with --help for all options.\n");
        return false;
    };

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err)
    {
        return fail(err.what());
    }

    // input
    if (program.is_used("model") && program.is_used("--input"))
        return fail("give the model either as a positional argument or with --input, not both");
    if (program.is_used("model"))
        opt.input = program.get<std::string>("model");
    else if (program.is_used("--input"))
        opt.input = program.get<std::string>("--input");
    if (program.get<bool>("--z-up") && program.get<bool>("--y-up"))
        return fail("--z-up and --y-up are mutually exclusive");
    opt.upAxis = program.get<bool>("--z-up") ? 'z' : program.get<bool>("--y-up") ? 'y' : 0;
    opt.vertexCache = !program.get<bool>("--no-vertex-cache");

    // meshlets
    MeshletSettings& m = opt.meshlets;
    const std::string algorithm = program.get<std::string>("--algorithm");
    m.algorithm = algorithm == "flex"      ? MeshletAlgorithm::Flex
                  : algorithm == "spatial" ? MeshletAlgorithm::Spatial
                  : algorithm == "scan"    ? MeshletAlgorithm::Scan
                                           : MeshletAlgorithm::Greedy;
    m.maxVertices = program.get<int>("--max-vertices");
    m.maxTriangles = program.get<int>("--max-triangles");
    m.minTriangles = program.get<int>("--min-triangles");
    m.coneWeight = program.get<float>("--cone-weight");
    m.splitFactor = program.get<float>("--split-factor");
    m.fillWeight = program.get<float>("--fill-weight");
    m.optimize = !program.get<bool>("--no-optimize-meshlets");

    if (m.maxVertices < 3 || m.maxVertices > 256)
        return fail("--max-vertices must be in [3, 256]");
    if (m.maxTriangles < 1 || m.maxTriangles > 256) // 256 = MAX_MESHLET_TRIANGLES in shaders/common.glsl
        return fail("--max-triangles must be in [1, 256]");
    if (m.minTriangles < 1 || m.minTriangles > m.maxTriangles)
    {
        if (program.is_used("--min-triangles"))
            return fail("--min-triangles must be in [1, --max-triangles]");
        m.minTriangles = std::min(m.minTriangles, m.maxTriangles); // default may exceed a small max
    }
    if (m.coneWeight < 0.f || m.coneWeight > 1.f)
        return fail("--cone-weight must be in [0, 1]");
    if (m.splitFactor < 0.f)
        return fail("--split-factor must be >= 0");
    if (m.fillWeight < 0.f || m.fillWeight > 1.f)
        return fail("--fill-weight must be in [0, 1]");

    auto warnUnused = [&](const char* name, bool usedBy) {
        if (program.is_used(name) && !usedBy)
            std::fprintf(stderr, "warning: %s has no effect with --algorithm %s\n", name, algorithm.c_str());
    };
    const bool flex = m.algorithm == MeshletAlgorithm::Flex, spatial = m.algorithm == MeshletAlgorithm::Spatial;
    warnUnused("--min-triangles", flex || spatial);
    warnUnused("--cone-weight", flex || m.algorithm == MeshletAlgorithm::Greedy);
    warnUnused("--split-factor", flex);
    warnUnused("--fill-weight", spatial);

    // view
    const std::string color = program.get<std::string>("--color");
    opt.matcap = program.get<std::string>("--matcap");
    if (program.is_used("--matcap") && color != "shaded")
        std::fprintf(stderr, "warning: --matcap only affects --color shaded\n");
    opt.colorMode = color == "triangle" ? ColorMode::Triangle
                    : color == "normal" ? ColorMode::Normal
                    : color == "shaded" ? ColorMode::Shaded
                                        : ColorMode::Meshlet;
    opt.wireframe = program.get<bool>("--wireframe");
    opt.wireWidth = program.get<float>("--wire-width");
    if (opt.wireWidth < 0.f)
        return fail("--wire-width must be >= 0");
    if (program.is_used("--wire-width") && !opt.wireframe)
        std::fprintf(stderr, "warning: --wire-width has no effect without --wireframe\n");
    opt.frustumCull = !program.get<bool>("--no-frustum-cull");
    opt.coneCull = !program.get<bool>("--no-cone-cull");
    opt.highlight = std::max(-1, program.get<int>("--highlight"));
    opt.zoom = program.get<float>("--zoom");
    if (opt.zoom <= 0.f)
        return fail("--zoom must be > 0");

    // app
    opt.noMeshShaders = program.get<bool>("--no-mesh-shaders");
    opt.validation = program.get<bool>("--validation");
    opt.width = program.get<int>("--width");
    opt.height = program.get<int>("--height");
    opt.frames = std::max(0, program.get<int>("--frames"));
    opt.screenshot = program.get<std::string>("--screenshot");
    if (opt.width < 64 || opt.height < 64)
        return fail("--width/--height must be at least 64");
    if (!opt.screenshot.empty() && opt.frames == 0)
        return fail("--screenshot requires --frames N");
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Options opt;
    if (!parseOptions(argc, argv, opt))
        return 1;

    if (volkInitialize() != VK_SUCCESS)
    {
        std::fprintf(stderr, "failed to load the Vulkan loader (libvulkan.so.1)\n");
        return 1;
    }

    glfwInitVulkanLoader(vkGetInstanceProcAddr);
    if (!glfwInit() || !glfwVulkanSupported())
    {
        std::fprintf(stderr, "GLFW init failed or Vulkan unsupported\n");
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(opt.width, opt.height, "meshlets", nullptr, nullptr);
    if (!window)
    {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        return 1;
    }
    glfwSetDropCallback(window, [](GLFWwindow*, int count, const char** paths) {
        if (count > 0)
            g_droppedPath = paths[0];
    });

    auto app = std::make_unique<App>();
    VulkanContext& ctx = app->ctx;
    ctx.init(window, opt.validation, opt.noMeshShaders);
    std::string matcapError;
    std::vector<Matcap> matcaps = loadMatcaps(matcapError);
    if (!matcapError.empty())
        std::fprintf(stderr, "warning: %s\n", matcapError.c_str());
    app->renderer.init(ctx, matcaps);

    // --matcap NAME -> index
    if (opt.matcap != "none")
    {
        for (size_t i = 0; i < matcaps.size(); ++i)
            if (matcaps[i].name == opt.matcap)
                app->draw.matcap = int(i);
        if (app->draw.matcap < 0 && !matcaps.empty())
        {
            std::string names;
            for (const Matcap& matcap : matcaps)
                names += " " + matcap.name;
            std::fprintf(stderr, "error: unknown matcap '%s'. Available: none%s\n", opt.matcap.c_str(), names.c_str());
            return 1;
        }
    }
    matcaps.clear(); // pixels now live on the GPU

    std::printf("GPU: %s (%s)\n", ctx.caps.props.deviceName, app->renderer.usesMeshShaders() ? "mesh shaders" : "vertex fallback");

    // --- settings from the command line ---
    app->settings = opt.meshlets;
    if (app->settings.maxVertices > app->maxVertexLimit() || app->settings.maxTriangles > app->maxTriangleLimit())
    {
        std::fprintf(stderr, "note: clamping meshlet size to this GPU's mesh shader limits (%d vertices, %d triangles)\n",
            app->maxVertexLimit(), app->maxTriangleLimit());
        app->settings.maxVertices = std::min(app->settings.maxVertices, app->maxVertexLimit());
        app->settings.maxTriangles = std::min(app->settings.maxTriangles, app->maxTriangleLimit());
        app->settings.minTriangles = std::min(app->settings.minTriangles, app->settings.maxTriangles);
    }
    app->optimizeVertexCache = opt.vertexCache;
    app->draw.mode = opt.colorMode;
    app->draw.wireframe = opt.wireframe;
    app->wireWidthOverride = opt.wireWidth;
    if (opt.wireframe && !app->renderer.supportsWireframe())
        std::fprintf(stderr, "warning: wireframe needs VK_KHR_fragment_shader_barycentric on the mesh shader path\n");
    app->draw.frustumCull = opt.frustumCull;
    app->draw.coneCull = opt.coneCull;
    app->draw.highlight = opt.highlight;
    app->initialZoom = opt.zoom;

    // --- ImGui ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 6.f;
    ImGui_ImplGlfw_InitForVulkan(window, true);

    VkFormat colorFormat = ctx.swapchain.format;
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = ctx.instance;
    initInfo.PhysicalDevice = ctx.physicalDevice;
    initInfo.Device = ctx.device;
    initInfo.QueueFamily = ctx.queueFamily;
    initInfo.Queue = ctx.queue;
    initInfo.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE + uint32_t(app->renderer.matcapCount());
    initInfo.MinImageCount = ctx.swapchain.minImageCount;
    initInfo.ImageCount = uint32_t(ctx.swapchain.images.size());
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &colorFormat;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo.depthAttachmentFormat = ctx.swapchain.depth.format;
    initInfo.CheckVkResultFn = checkVkResult;
    ImGui_ImplVulkan_Init(&initInfo);
    for (size_t i = 0; i < app->renderer.matcapCount(); ++i)
        app->matcapTextures.push_back(
            ImTextureID(ImGui_ImplVulkan_AddTexture(app->renderer.matcapUiView(i), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)));

    // Default model: assets/lucy.ply (Stanford Lucy, Z-up), looked up in the working
    // directory first, then in the source tree the binary was built from.
    std::string modelPath = opt.input;
    bool defaultModel = false;
    if (modelPath.empty())
    {
        for (const std::string& candidate : {std::string("assets/lucy.ply"), std::string(MESHLETS_ASSET_DIR) + "/lucy.ply"})
            if (FILE* f = std::fopen(candidate.c_str(), "rb"))
            {
                std::fclose(f);
                modelPath = candidate;
                defaultModel = true;
                break;
            }
        if (modelPath.empty())
            std::fprintf(stderr, "note: assets/lucy.ply not found; drop a model onto the window or pass a path\n");
    }
    app->camera.zUp = opt.upAxis == 'z' || (opt.upAxis == 0 && defaultModel);
    if (!modelPath.empty())
        app->startLoad(modelPath);

    const int frameLimit = opt.frames;
    const std::string screenshotPath = opt.screenshot;
    int readyFrames = 0;
    int exitCode = 0;
    uint32_t frameIndex = 0;
    bool swapchainDirty = false;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (!g_droppedPath.empty())
        {
            app->startLoad(g_droppedPath);
            g_droppedPath.clear();
        }
        app->pollJob();
        if (frameLimit > 0 && app->loadFailed)
        {
            exitCode = 1;
            break;
        }

        int fbWidth = 0, fbHeight = 0;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (fbWidth == 0 || fbHeight == 0)
        {
            glfwWaitEventsTimeout(0.1); // minimized
            continue;
        }
        if (swapchainDirty || uint32_t(fbWidth) != ctx.swapchain.extent.width || uint32_t(fbHeight) != ctx.swapchain.extent.height)
        {
            ctx.recreateSwapchain();
            ImGui_ImplVulkan_SetMinImageCount(ctx.swapchain.minImageCount);
            swapchainDirty = false;
        }

        Frame& frame = ctx.frames[frameIndex];
        VK_CHECK(vkWaitForFences(ctx.device, 1, &frame.fence, VK_TRUE, UINT64_MAX));
        FrameStats frameStats = app->renderer.collectStats(frameIndex);
        if (frameStats.valid)
            app->stats = frameStats;

        // --- UI + camera ---
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        app->camera.update(ImGui::GetIO());
        app->ui();
        ImGui::Render();

        uint32_t imageIndex = 0;
        VkResult acquire = vkAcquireNextImageKHR(ctx.device, ctx.swapchain.handle, UINT64_MAX, frame.imageAcquired, VK_NULL_HANDLE, &imageIndex);
        if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
        {
            swapchainDirty = true;
            continue;
        }
        if (acquire != VK_SUBOPTIMAL_KHR)
            VK_CHECK(acquire);

        VK_CHECK(vkResetFences(ctx.device, 1, &frame.fence));
        VK_CHECK(vkResetCommandPool(ctx.device, frame.commandPool, 0));

        VkCommandBuffer cmd = frame.commandBuffer;
        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

        float aspect = float(ctx.swapchain.extent.width) / float(ctx.swapchain.extent.height);
        app->draw.viewProj = app->camera.viewProj(aspect);
        app->draw.cameraPos = app->camera.eye();
        app->draw.lightDir = app->camera.lightDir();
        app->camera.basis(app->draw.viewRight, app->draw.viewUp, app->draw.viewBack);
        app->renderer.record(cmd, frameIndex, imageIndex, app->draw,
            [](VkCommandBuffer c) { ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), c); });

        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSemaphore renderFinished = ctx.swapchain.renderFinished[imageIndex];
        VkSemaphoreSubmitInfo waitInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitInfo.semaphore = frame.imageAcquired;
        waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSemaphoreSubmitInfo signalInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signalInfo.semaphore = renderFinished;
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkCommandBufferSubmitInfo cmdInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cmdInfo.commandBuffer = cmd;

        VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.waitSemaphoreInfoCount = 1;
        submitInfo.pWaitSemaphoreInfos = &waitInfo;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdInfo;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos = &signalInfo;
        VK_CHECK(vkQueueSubmit2(ctx.queue, 1, &submitInfo, frame.fence));

        // --frames counts only frames rendered after loading finished, so screenshots show the model
        bool lastFrame = false;
        if (frameLimit > 0 && !app->busy() && (app->model || modelPath.empty()))
            lastFrame = ++readyFrames >= frameLimit;
        VkPresentInfoKHR presentInfo = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &ctx.swapchain.handle;
        presentInfo.pImageIndices = &imageIndex;

        if (lastFrame && !screenshotPath.empty())
        {
            // Read back from the image before handing it to the presentation engine.
            // The image is in PRESENT_SRC layout and rendering is complete (queue idle).
            saveScreenshot(ctx, imageIndex, screenshotPath.c_str());
        }

        VkResult present = vkQueuePresentKHR(ctx.queue, &presentInfo);
        if (present == VK_ERROR_OUT_OF_DATE_KHR || present == VK_SUBOPTIMAL_KHR)
            swapchainDirty = true;
        else
            VK_CHECK(present);

        frameIndex = (frameIndex + 1) % kFramesInFlight;
        if (lastFrame)
        {
            if (app->stats.valid)
                std::printf("visible meshlets %u, triangles %u, gpu %.2f ms\n", app->stats.visibleMeshlets, app->stats.visibleTriangles,
                    app->stats.gpuMs);
            break;
        }
    }

    // --- Shutdown ---
    VK_CHECK(vkDeviceWaitIdle(ctx.device));
    if (app->busy())
        app->job.wait();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    app->renderer.destroy();
    ctx.destroy();

    glfwDestroyWindow(window);
    glfwTerminate();
    return exitCode;
}
