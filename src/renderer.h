#pragma once

#include "matcap.h"
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
    vec3 viewRight, viewUp, viewBack; // camera basis (world space), for matcap lookups
    int matcap = -1;                  // index into the loaded matcaps; -1 = plain lighting
    ColorMode mode = ColorMode::Meshlet;
    int highlight = -1;
    bool frustumCull = true;
    bool coneCull = true;
    bool wireframe = false;
    float wireWidth = 0.f; // line width in model units
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
    // matcaps: textures for the Shaded mode (may be empty)
    void init(VulkanContext& ctx, const std::vector<Matcap>& matcaps);
    void destroy();

    // Uploads model + meshlets to the GPU (blocks; waits for the device to be idle first).
    void upload(const Model& model, const MeshletData& meshlets);
    void releaseScene();
    bool hasScene() const { return meshletCount > 0; }
    bool usesMeshShaders() const { return meshPipeline != VK_NULL_HANDLE; }
    // The vertex path passes barycentrics itself; the mesh path needs VK_KHR_fragment_shader_barycentric.
    bool supportsWireframe() const { return !usesMeshShaders() || ctx->caps.barycentric; }
    size_t gpuBytes() const { return sceneBytes; }
    size_t matcapCount() const { return matcapNames.size(); }
    const std::string& matcapName(size_t i) const { return matcapNames[i]; }
    VkImageView matcapUiView(size_t i) const { return matcapImages[i].unormView; }

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
    VkPipeline createPipeline(bool mesh);

    VulkanContext* ctx = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE;
    VkPipeline vertexPipeline = VK_NULL_HANDLE;
    VkQueryPool timestamps = VK_NULL_HANDLE;

    // matcap textures, bound as one sampler2D array (descriptor set 0, binding 0)
    std::vector<Image> matcapImages; // always >= 1 (a white placeholder when none were found)
    std::vector<std::string> matcapNames;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    float timestampPeriod = 1.f;

    PerFrame frames[kFramesInFlight];

    Buffer vertices, meshlets, meshletVertices, meshletTriangles, bounds;
    Buffer fallbackIndices, fallbackTriangles;
    uint32_t meshletCount = 0;
    uint32_t triangleCount = 0;
    size_t sceneBytes = 0;
};
