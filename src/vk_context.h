#pragma once

#include <volk.h>

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define VK_CHECK(call)                                                                              \
    do                                                                                              \
    {                                                                                               \
        VkResult result_ = (call);                                                                  \
        if (result_ != VK_SUCCESS)                                                                  \
        {                                                                                           \
            std::fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #call, int(result_)); \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)

constexpr uint32_t kFramesInFlight = 2;

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* mapped = nullptr; // non-null for host-visible buffers
};

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView unormView = VK_NULL_HANDLE; // textures only: raw sRGB bytes, for ImGui
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct Frame
{
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore imageAcquired = VK_NULL_HANDLE;
};

struct Swapchain
{
    VkSwapchainKHR handle = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent = {};
    uint32_t minImageCount = 0;
    bool canReadback = false; // images support TRANSFER_SRC (screenshots)
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    std::vector<VkSemaphore> renderFinished; // one per image
    Image depth;
};

struct DeviceCaps
{
    bool meshShader = false;
    uint32_t maxMeshOutputVertices = 0;
    uint32_t maxMeshOutputPrimitives = 0;
    uint32_t maxTaskWorkGroupCount = 0;
    bool barycentric = false; // VK_KHR_fragment_shader_barycentric (wireframe on the mesh shader path)
    VkPhysicalDeviceProperties props = {};
};

struct VulkanContext
{
    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool uploadPool = VK_NULL_HANDLE;
    Swapchain swapchain;
    Frame frames[kFramesInFlight];
    DeviceCaps caps;

    // validation: enable VK_LAYER_KHRONOS_validation if installed
    // disableMeshShaders: force the vertex-shader fallback path
    void init(GLFWwindow* window, bool validation, bool disableMeshShaders);
    void destroy();

    void recreateSwapchain();

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible);
    // Creates a device-local buffer and fills it through a staging buffer (blocking).
    Buffer createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage);
    void destroyBuffer(Buffer& buffer);

    // RGBA8 sRGB texture, uploaded and left in SHADER_READ_ONLY_OPTIMAL (blocking).
    // `view` samples as linear (SRGB format); `unormView` returns the stored bytes.
    Image createTexture(uint32_t width, uint32_t height, const void* rgba8);
    void destroyImage(Image& image);

private:
    void createInstance(bool validation);
    void pickPhysicalDevice(bool disableMeshShaders);
    void createDevice();
    void createSwapchain();
    void destroySwapchainResources();
    void createFrames();
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags);
    void submitImmediate(void (*record)(VkCommandBuffer, void*), void* user);
};

void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);
