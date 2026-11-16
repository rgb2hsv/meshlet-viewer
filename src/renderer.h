#pragma once

#include "math.h"
#include "model.h"
#include "vk_context.h"

enum class ColorMode : uint32_t
{
    Meshlet = 0,
    Triangle = 1,
    Normal = 2,
    Shaded = 3,
};

struct DrawParams
{
    mat4 viewProj;
    vec3 cameraPos;
    vec3 lightDir = {0, 0, 1}; // world space, towards the light
    ColorMode mode = ColorMode::Meshlet;
    int highlight = -1;
    bool frustumCull = true;
    bool coneCull = true;
    bool wireframe = false;
    float clearColor[3] = {0.08f, 0.09f, 0.11f};
};

struct FrameStats
{
    uint32_t visibleMeshlets = 0;
    uint32_t visibleTriangles = 0;
    double gpuMs = 0;
    bool valid = false;
};

class MeshletRenderer
{
public:
    void init(VulkanContext& ctx);
    void destroy();

    // Uploads model + meshlets to the GPU (blocks; waits for the device to be idle first).
    void upload(const Model& model, const MeshletData& meshlets);
    void releaseScene();
    bool hasScene() const { return meshletCount > 0; }
    bool usesMeshShaders() const { return meshPipeline != VK_NULL_HANDLE; }
    size_t gpuBytes() const { return sceneBytes; }

    // Must be called after the frame's fence was waited on; returns the stats from that frame's last use.
    FrameStats collectStats(uint32_t frameIndex);

    // Records the full frame into cmd: clears, draws meshlets, then invokes `overlay` inside the same rendering pass.
    template <typename Overlay>
    void record(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex, const DrawParams& params, Overlay&& overlay)
    {
        begin(cmd, frameIndex, imageIndex, params);
        overlay(cmd);
        end(cmd, frameIndex, imageIndex);
    }

private:
    struct PerFrame
    {
        Buffer globals;
        Buffer stats;
        bool submitted = false;
    };

    void begin(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex, const DrawParams& params);
    void end(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex);
    void createPipelines();
    VkPipeline createPipeline(bool mesh, bool wireframe);

    VulkanContext* ctx = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE;
    VkPipeline meshWirePipeline = VK_NULL_HANDLE;
    VkPipeline vertexPipeline = VK_NULL_HANDLE;
    VkPipeline vertexWirePipeline = VK_NULL_HANDLE;
    VkQueryPool timestamps = VK_NULL_HANDLE;
    float timestampPeriod = 1.f;

    PerFrame frames[kFramesInFlight];

    Buffer vertices, meshlets, meshletVertices, meshletTriangles, bounds;
    Buffer fallbackIndices, fallbackTriangles;
    uint32_t meshletCount = 0;
    uint32_t triangleCount = 0;
    size_t sceneBytes = 0;
};
