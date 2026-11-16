#include "renderer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

// SPIR-V generated at build time by glslang (see CMakeLists.txt)
#include "kMeshletFragMeshNoBarySpv.h"
#include "kMeshletFragMeshSpv.h"
#include "kMeshletFragVertSpv.h"
#include "kMeshletMeshSpv.h"
#include "kMeshletTaskSpv.h"
#include "kMeshletVertSpv.h"

namespace
{

constexpr uint32_t kTaskGroupSize = 32; // must match TASK_GROUP_SIZE in common.glsl

// Must match `Globals` in shaders/common.glsl (std430)
struct GpuGlobals
{
    float viewProj[16];
    float frustum[4][4];
    float cameraPos[4];
    float lightDir[4];
    uint32_t meshletCount;
    uint32_t mode;
    int32_t highlight;
    uint32_t flags;
    float wireWidth;
    int32_t matcap;
    float pad[2];
    float viewRight[4];
    float viewUp[4];
    float viewBack[4];
    float lodCamera[4];
    float lodScale;
    float lodThreshold;
    uint32_t lodMode;
    uint32_t lodLevel;
};
static_assert(sizeof(GpuGlobals) == 272, "GpuGlobals layout mismatch");

// Must match `PushConstants` in shaders/common.glsl
struct PushConstants
{
    VkDeviceAddress globals;
    VkDeviceAddress vertices;
    VkDeviceAddress meshlets;
    VkDeviceAddress meshletVertices;
    VkDeviceAddress meshletTriangles;
    VkDeviceAddress bounds;
    VkDeviceAddress fallbackIndices;
    VkDeviceAddress fallbackTriangles;
    VkDeviceAddress stats;
    VkDeviceAddress lods;
};

struct GpuStats
{
    uint32_t visibleMeshlets;
    uint32_t visibleTriangles;
    uint32_t levelClusters[kMaxLodLevels];
    uint32_t levelTriangles[kMaxLodLevels];
};

constexpr uint32_t kTriBits = 9; // TRI_BITS in common.glsl

VkShaderModule createShaderModule(VkDevice device, const uint32_t* code, size_t size)
{
    VkShaderModuleCreateInfo info = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = size;
    info.pCode = code;
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

// Extracts normalized left/right/bottom/top planes from a column-major view-projection matrix.
void extractFrustum(const mat4& vp, float planes[4][4])
{
    for (int i = 0; i < 4; ++i)
    {
        int axis = i / 2;            // 0 = x, 1 = y
        float sign = (i % 2) ? -1.f : 1.f;
        float p[4];
        for (int c = 0; c < 4; ++c)
            p[c] = vp.at(3, c) + sign * vp.at(axis, c);
        float len = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        for (int c = 0; c < 4; ++c)
            planes[i][c] = p[c] / len;
    }
}

} // namespace

void MeshletRenderer::init(VulkanContext& context, const std::vector<Matcap>& matcaps)
{
    ctx = &context;
    timestampPeriod = ctx->caps.props.limits.timestampPeriod;

    // --- matcap textures ---
    for (const Matcap& matcap : matcaps)
    {
        matcapImages.push_back(ctx->createTexture(uint32_t(matcap.width), uint32_t(matcap.height), matcap.pixels.data()));
        matcapNames.push_back(matcap.name);
    }
    if (matcapImages.empty())
    {
        const uint8_t white[4] = {255, 255, 255, 255};
        matcapImages.push_back(ctx->createTexture(1, 1, white)); // keeps the descriptor array valid
    }

    VkSamplerCreateInfo samplerInfo = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(ctx->device, &samplerInfo, nullptr, &sampler));

    const uint32_t textureCount = uint32_t(matcapImages.size());
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = textureCount;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo setLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayoutInfo.bindingCount = 1;
    setLayoutInfo.pBindings = &binding;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->device, &setLayoutInfo, nullptr, &setLayout));

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureCount};
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx->device, &poolInfo, nullptr, &descriptorPool));

    VkDescriptorSetAllocateInfo setInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    setInfo.descriptorPool = descriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &setLayout;
    VK_CHECK(vkAllocateDescriptorSets(ctx->device, &setInfo, &descriptorSet));

    std::vector<VkDescriptorImageInfo> imageInfos;
    for (const Image& image : matcapImages)
        imageInfos.push_back({sampler, image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = textureCount;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = imageInfos.data();
    vkUpdateDescriptorSets(ctx->device, 1, &write, 0, nullptr);

    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_ALL;
    range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(ctx->device, &layoutInfo, nullptr, &layout));

    createPipelines();

    VkQueryPoolCreateInfo queryInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 2 * kFramesInFlight;
    VK_CHECK(vkCreateQueryPool(ctx->device, &queryInfo, nullptr, &timestamps));

    for (PerFrame& frame : frames)
    {
        frame.globals = ctx->createBuffer(sizeof(GpuGlobals), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
        frame.stats = ctx->createBuffer(sizeof(GpuStats), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true);
        std::memset(frame.stats.mapped, 0, sizeof(GpuStats));
    }
}

void MeshletRenderer::destroy()
{
    releaseScene();
    for (PerFrame& frame : frames)
    {
        ctx->destroyBuffer(frame.globals);
        ctx->destroyBuffer(frame.stats);
    }
    vkDestroyQueryPool(ctx->device, timestamps, nullptr);
    for (VkPipeline pipeline : {meshPipeline, vertexPipeline})
        if (pipeline)
            vkDestroyPipeline(ctx->device, pipeline, nullptr);
    vkDestroyPipelineLayout(ctx->device, layout, nullptr);
    vkDestroyDescriptorPool(ctx->device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx->device, setLayout, nullptr);
    vkDestroySampler(ctx->device, sampler, nullptr);
    for (Image& image : matcapImages)
        ctx->destroyImage(image);
    matcapImages.clear();
}

void MeshletRenderer::createPipelines()
{
    if (ctx->caps.meshShader)
        meshPipeline = createPipeline(true);
    else
        vertexPipeline = createPipeline(false);
}

VkPipeline MeshletRenderer::createPipeline(bool mesh)
{
    VkDevice device = ctx->device;

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<VkShaderModule> modules;

    // MATCAP_COUNT (constant_id 0) sizes the sampler array in meshlet.frag
    const uint32_t matcapTextureCount = uint32_t(matcapImages.size());
    VkSpecializationMapEntry specEntry = {0, 0, sizeof(uint32_t)};
    VkSpecializationInfo specInfo = {1, &specEntry, sizeof(uint32_t), &matcapTextureCount};

    auto addStage = [&](VkShaderStageFlagBits stage, const uint32_t* code, size_t size) {
        VkShaderModule module = createShaderModule(device, code, size);
        modules.push_back(module);
        VkPipelineShaderStageCreateInfo info = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage = stage;
        info.module = module;
        info.pName = "main";
        if (stage == VK_SHADER_STAGE_FRAGMENT_BIT)
            info.pSpecializationInfo = &specInfo;
        stages.push_back(info);
    };

    if (mesh)
    {
        addStage(VK_SHADER_STAGE_TASK_BIT_EXT, kMeshletTaskSpv, sizeof(kMeshletTaskSpv));
        addStage(VK_SHADER_STAGE_MESH_BIT_EXT, kMeshletMeshSpv, sizeof(kMeshletMeshSpv));
        if (ctx->caps.barycentric)
            addStage(VK_SHADER_STAGE_FRAGMENT_BIT, kMeshletFragMeshSpv, sizeof(kMeshletFragMeshSpv));
        else
            addStage(VK_SHADER_STAGE_FRAGMENT_BIT, kMeshletFragMeshNoBarySpv, sizeof(kMeshletFragMeshNoBarySpv));
    }
    else
    {
        addStage(VK_SHADER_STAGE_VERTEX_BIT, kMeshletVertSpv, sizeof(kMeshletVertSpv));
        addStage(VK_SHADER_STAGE_FRAGMENT_BIT, kMeshletFragVertSpv, sizeof(kMeshletFragVertSpv));
    }

    VkPipelineVertexInputStateCreateInfo vertexInput = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL; // wireframe is drawn in the fragment shader
    raster.cullMode = VK_CULL_MODE_NONE; // scanned models often have inconsistent winding
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.f;

    VkPipelineMultisampleStateCreateInfo multisample = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // reversed-Z

    VkPipelineColorBlendAttachmentState blendAttachment = {};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    VkPipelineRenderingCreateInfo rendering = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &ctx->swapchain.format;
    rendering.depthAttachmentFormat = ctx->swapchain.depth.format;

    VkGraphicsPipelineCreateInfo info = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = uint32_t(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = mesh ? nullptr : &vertexInput;
    info.pInputAssemblyState = mesh ? nullptr : &inputAssembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    for (VkShaderModule module : modules)
        vkDestroyShaderModule(device, module, nullptr);
    return pipeline;
}

void MeshletRenderer::upload(const Model& model, const MeshletData& data)
{
    VK_CHECK(vkDeviceWaitIdle(ctx->device));
    releaseScene();

    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vertices = ctx->createDeviceBuffer(model.vertices.data(), model.vertices.size() * sizeof(Vertex), usage);

    if (usesMeshShaders())
    {
        meshlets = ctx->createDeviceBuffer(data.meshlets.data(), data.meshlets.size() * sizeof(meshopt_Meshlet), usage);
        meshletVertices = ctx->createDeviceBuffer(data.vertices.data(), data.vertices.size() * sizeof(uint32_t), usage);
        meshletTriangles = ctx->createDeviceBuffer(data.triangles.data(), data.triangles.size(), usage);
        bounds = ctx->createDeviceBuffer(data.bounds.data(), data.bounds.size() * sizeof(MeshletBounds), usage);
    }
    else
    {
        // Vertex fallback: expand the level-0 meshlets into a flat corner list + per-triangle ids
        std::vector<uint32_t> indices;
        std::vector<uint32_t> triangles;
        indices.reserve(data.baseTriangleCount * 3);
        triangles.reserve(data.baseTriangleCount);
        fallbackClusters = 0;
        for (size_t mi = 0; mi < data.meshlets.size(); ++mi)
        {
            if (data.lods[mi].level != 0)
                continue;
            fallbackClusters++;
            const meshopt_Meshlet& m = data.meshlets[mi];
            for (uint32_t t = 0; t < m.triangle_count; ++t)
            {
                for (uint32_t k = 0; k < 3; ++k)
                    indices.push_back(data.vertices[m.vertex_offset + data.triangles[m.triangle_offset + t * 3 + k]]);
                triangles.push_back(uint32_t(mi << kTriBits) | t);
            }
        }
        fallbackIndices = ctx->createDeviceBuffer(indices.data(), indices.size() * sizeof(uint32_t), usage);
        fallbackTriangles = ctx->createDeviceBuffer(triangles.data(), triangles.size() * sizeof(uint32_t), usage);
        fallbackTriangleCount = uint32_t(triangles.size());
    }
    lods = ctx->createDeviceBuffer(data.lods.data(), data.lods.size() * sizeof(MeshletLod), usage); // also read by the fragment shader

    meshletCount = uint32_t(data.meshlets.size());
    triangleCount = uint32_t(data.triangleCount);
    sceneBytes = size_t(vertices.size + meshlets.size + meshletVertices.size + meshletTriangles.size + bounds.size + lods.size +
                        fallbackIndices.size + fallbackTriangles.size);
}

void MeshletRenderer::releaseScene()
{
    if (!ctx)
        return;
    if (vertices.buffer)
        VK_CHECK(vkDeviceWaitIdle(ctx->device));
    for (Buffer* buffer : {&vertices, &meshlets, &meshletVertices, &meshletTriangles, &bounds, &lods, &fallbackIndices, &fallbackTriangles})
        ctx->destroyBuffer(*buffer);
    meshletCount = triangleCount = 0;
    fallbackClusters = fallbackTriangleCount = 0;
    sceneBytes = 0;
}

FrameStats MeshletRenderer::collectStats(uint32_t frameIndex)
{
    FrameStats result;
    PerFrame& frame = frames[frameIndex];
    if (!frame.submitted)
        return result;

    const GpuStats* stats = static_cast<const GpuStats*>(frame.stats.mapped);
    result.visibleMeshlets = stats->visibleMeshlets;
    result.visibleTriangles = stats->visibleTriangles;
    std::memcpy(result.levelClusters, stats->levelClusters, sizeof(result.levelClusters));
    std::memcpy(result.levelTriangles, stats->levelTriangles, sizeof(result.levelTriangles));

    uint64_t ticks[2] = {};
    if (vkGetQueryPoolResults(ctx->device, timestamps, frameIndex * 2, 2, sizeof(ticks), ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) ==
        VK_SUCCESS)
        result.gpuMs = double(ticks[1] - ticks[0]) * double(timestampPeriod) * 1e-6;

    result.valid = true;
    return result;
}

void MeshletRenderer::begin(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex, const DrawParams& params)
{
    PerFrame& frame = frames[frameIndex];
    const Swapchain& sc = ctx->swapchain;

    // per-frame constants (frame fence has been waited on, so the buffers are free)
    GpuGlobals globals = {};
    std::memcpy(globals.viewProj, params.viewProj.m, sizeof(globals.viewProj));
    extractFrustum(params.cullViewProj, globals.frustum);
    globals.cameraPos[0] = params.cameraPos.x;
    globals.cameraPos[1] = params.cameraPos.y;
    globals.cameraPos[2] = params.cameraPos.z;
    globals.lightDir[0] = params.lightDir.x;
    globals.lightDir[1] = params.lightDir.y;
    globals.lightDir[2] = params.lightDir.z;
    globals.meshletCount = meshletCount;
    globals.mode = uint32_t(params.mode);
    globals.highlight = params.highlight;
    const bool wireframe = params.wireframe && params.wireWidth > 0.f && supportsWireframe();
    globals.flags = (params.frustumCull ? 1u : 0u) | (params.coneCull ? 2u : 0u) | (wireframe ? 4u : 0u);
    globals.wireWidth = params.wireWidth;
    globals.matcap = params.matcap >= 0 && size_t(params.matcap) < matcapNames.size() ? params.matcap : -1;
    const vec3 basis[3] = {params.viewRight, params.viewUp, params.viewBack};
    float* basisOut[3] = {globals.viewRight, globals.viewUp, globals.viewBack};
    for (int i = 0; i < 3; ++i)
    {
        basisOut[i][0] = basis[i].x;
        basisOut[i][1] = basis[i].y;
        basisOut[i][2] = basis[i].z;
    }

    globals.lodCamera[0] = params.cullCameraPos.x;
    globals.lodCamera[1] = params.cullCameraPos.y;
    globals.lodCamera[2] = params.cullCameraPos.z;
    globals.lodCamera[3] = params.zNear;
    globals.lodScale = params.lodScale;
    globals.lodThreshold = params.lodThreshold;
    globals.lodMode = params.lodAuto ? 0u : 1u;
    globals.lodLevel = uint32_t(std::max(params.lodLevel, 0));
    std::memcpy(frame.globals.mapped, &globals, sizeof(globals));

    GpuStats zero = {};
    if (!usesMeshShaders())
    {
        // no culling or LOD in the fallback path: level 0 is always drawn in full
        zero.visibleMeshlets = zero.levelClusters[0] = fallbackClusters;
        zero.visibleTriangles = zero.levelTriangles[0] = fallbackTriangleCount;
    }
    std::memcpy(frame.stats.mapped, &zero, sizeof(zero));
    frame.submitted = true;

    vkCmdResetQueryPool(cmd, timestamps, frameIndex * 2, 2);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, timestamps, frameIndex * 2);

    VkImage image = sc.images[imageIndex];
    transitionImage(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transitionImage(cmd, sc.depth.image, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo color = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = sc.views[imageIndex];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color = {{params.clearColor[0], params.clearColor[1], params.clearColor[2], 1.f}};

    VkRenderingAttachmentInfo depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = sc.depth.view;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue.depthStencil = {0.f, 0}; // reversed-Z

    VkRenderingInfo renderingInfo = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea = {{0, 0}, sc.extent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &color;
    renderingInfo.pDepthAttachment = &depthAttachment;
    vkCmdBeginRendering(cmd, &renderingInfo);

    VkViewport vp = {0, 0, float(sc.extent.width), float(sc.extent.height), 0, 1};
    VkRect2D scissor = {{0, 0}, sc.extent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    if (meshletCount == 0)
        return;

    PushConstants push = {};
    push.globals = frame.globals.address;
    push.vertices = vertices.address;
    push.meshlets = meshlets.address;
    push.meshletVertices = meshletVertices.address;
    push.meshletTriangles = meshletTriangles.address;
    push.bounds = bounds.address;
    push.fallbackIndices = fallbackIndices.address;
    push.fallbackTriangles = fallbackTriangles.address;
    push.stats = frame.stats.address;
    push.lods = lods.address;

    const uint32_t taskGroups = (meshletCount + kTaskGroupSize - 1) / kTaskGroupSize;

    // Single pass: the wireframe (if enabled) is composited in the fragment shader.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, usesMeshShaders() ? meshPipeline : vertexPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_ALL, 0, sizeof(push), &push);
    if (usesMeshShaders())
        vkCmdDrawMeshTasksEXT(cmd, taskGroups, 1, 1);
    else
        vkCmdDraw(cmd, fallbackTriangleCount * 3, 1, 0, 0);
}

void MeshletRenderer::end(VkCommandBuffer cmd, uint32_t frameIndex, uint32_t imageIndex)
{
    vkCmdEndRendering(cmd);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, timestamps, frameIndex * 2 + 1);

    transitionImage(cmd, ctx->swapchain.images[imageIndex], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE, 0);
}
