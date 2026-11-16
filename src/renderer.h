#pragma once

#include "matcap.h"
#include "camera_math.h"
#include "model.h"
#include "vk_context.h"

enum class ColorMode : uint32_t
{
    Meshlet = 0,
    Triangle = 1,
    Normal = 2,
    Shaded = 3,
};

constexpr uint32_t kMaxLodLevels = 32; // MAX_LOD_LEVELS in shaders/common.glsl

struct DrawParams
{
    Eigen::Matrix4f viewProj = Eigen::Matrix4f::Identity();
    Eigen::Vector3f cameraPos = Eigen::Vector3f::Zero();
    Eigen::Vector3f lightDir = Eigen::Vector3f::UnitZ(); // world space, towards the light
    // camera basis (world space), for matcap lookups
    Eigen::Vector3f viewRight = Eigen::Vector3f::UnitX(), viewUp = Eigen::Vector3f::UnitY(), viewBack = Eigen::Vector3f::UnitZ();
    int matcap = -1;                  // index into the loaded matcaps; -1 = plain lighting

    // LOD selection and culling use their own camera so they can be frozen while orbiting
    Eigen::Matrix4f cullViewProj = Eigen::Matrix4f::Identity();
    Eigen::Vector3f cullCameraPos = Eigen::Vector3f::Zero();
    bool lodAuto = true;       // true: pixel error threshold; false: fixed DAG level
    float lodThreshold = 1.f;  // pixels
    int lodLevel = 0;
    float lodScale = 1.f;      // cot(fovY / 2) * 0.5 * viewport height
    float zNear = 0.01f;
    ColorMode mode = ColorMode::Meshlet;
    int highlight = -1;
    bool frustumCull = true;
    bool coneCull = true;
    bool wireframe = false;
    float wireWidth = 0.f;    // triangle wireframe line width in model units (0 = off)
    float outlineWidth = 0.f; // meshlet outline width in model units (0 = off)
    float clearColor[3] = {0.08f, 0.09f, 0.11f};
    float viewportX = 0.f; // left edge of the 3D view in framebuffer pixels (right of the sidebar)
};

struct FrameStats
{
    uint32_t visibleMeshlets = 0;
    uint32_t visibleTriangles = 0;
    double gpuMs = 0;
    uint32_t levelClusters[kMaxLodLevels] = {};
    uint32_t levelTriangles[kMaxLodLevels] = {};
    bool bvh = false;           // automatic LOD selected through the BVH this frame
    uint32_t bvhNodesVisited = 0;
    uint32_t bvhCandidates = 0; // clusters the BVH selected (before frustum/cone culling)
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
    // Cluster LOD selection runs in the task shader; the vertex fallback draws level 0 only.
    bool supportsLod() const { return usesMeshShaders(); }
    bool hasLodBvh() const { return lodNodes.buffer != VK_NULL_HANDLE; }
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
        bool usedBvh = false;
        // BVH traversal (sized with the scene)
        Buffer traversal; // host-visible counters + indirect mesh-task command
        Buffer candidates;
        Buffer queues[2];
    };

    void begin(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex, const DrawParams& params);
    void end(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex);
    void createPipelines();
    VkPipeline createPipeline(bool mesh);

    VulkanContext* ctx = nullptr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline meshPipeline = VK_NULL_HANDLE;
    VkPipeline vertexPipeline = VK_NULL_HANDLE;
    VkPipeline traversePipeline = VK_NULL_HANDLE; // lod_traverse.comp
    VkPipeline finalizePipeline = VK_NULL_HANDLE; // lod_finalize.comp
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
    Buffer lods;
    Buffer edgeMasks;      // meshlet perimeter edges per triangle
    uint32_t lodLevelCount = 1;
    Buffer lodNodes, lodGroups;              // BVH (cluster LOD builds only)
    uint32_t lodRootCount = 0;
    std::vector<uint32_t> lodNodesPerDepth;
    Buffer fallbackIndices, fallbackTriangles;
    uint32_t fallbackClusters = 0;  // fallback draws level 0 only
    uint32_t fallbackTriangleCount = 0;
    uint32_t meshletCount = 0;
    uint32_t triangleCount = 0;
    size_t sceneBytes = 0;
};
