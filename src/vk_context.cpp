#include "vk_context.h"

#include <algorithm>
#include <cstring>

namespace
{

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    const char* level = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : "WARNING";
    std::fprintf(stderr, "[vulkan %s] %s\n", level, data->pMessage);
    return VK_FALSE;
}

bool hasLayer(const char* name)
{
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const VkLayerProperties& layer : layers)
        if (std::strcmp(layer.layerName, name) == 0)
            return true;
    return false;
}

bool hasDeviceExtension(VkPhysicalDevice device, const char* name)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
    for (const VkExtensionProperties& ext : extensions)
        if (std::strcmp(ext.extensionName, name) == 0)
            return true;
    return false;
}

} // namespace

void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess)
{
    VkImageMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};

    VkDependencyInfo dep = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void VulkanContext::init(GLFWwindow* win, bool validation, bool disableMeshShaders)
{
    window = win;
    createInstance(validation);
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    pickPhysicalDevice(disableMeshShaders);
    createDevice();
    createSwapchain();
    createFrames();

    VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &uploadPool));
}

void VulkanContext::destroy()
{
    vkDestroyCommandPool(device, uploadPool, nullptr);
    for (Frame& frame : frames)
    {
        vkDestroySemaphore(device, frame.imageAcquired, nullptr);
        vkDestroyFence(device, frame.fence, nullptr);
        vkDestroyCommandPool(device, frame.commandPool, nullptr);
    }
    destroySwapchainResources();
    vkDestroySwapchainKHR(device, swapchain.handle, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    if (messenger)
        vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
    vkDestroyInstance(instance, nullptr);
}

void VulkanContext::createInstance(bool validation)
{
    const char* validationLayer = "VK_LAYER_KHRONOS_validation";
    if (validation && !hasLayer(validationLayer))
    {
        std::fprintf(stderr, "warning: %s not installed; continuing without validation\n", validationLayer);
        validation = false;
    }

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (validation)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "meshlets";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = uint32_t(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    if (validation)
    {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &validationLayer;
    }

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
    volkLoadInstanceOnly(instance);

    if (validation)
    {
        VkDebugUtilsMessengerCreateInfoEXT messengerInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messengerInfo.pfnUserCallback = debugCallback;
        VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &messengerInfo, nullptr, &messenger));
    }
}

void VulkanContext::pickPhysicalDevice(bool disableMeshShaders)
{
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

    int bestScore = -1;
    for (VkPhysicalDevice candidate : devices)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_3 || !hasDeviceExtension(candidate, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
            continue;

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());

        for (uint32_t i = 0; i < familyCount; ++i)
        {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface, &present);
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present)
                continue;

            int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
            if (hasDeviceExtension(candidate, VK_EXT_MESH_SHADER_EXTENSION_NAME))
                score += 2;
            if (score > bestScore)
            {
                bestScore = score;
                physicalDevice = candidate;
                queueFamily = i;
            }
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE)
    {
        std::fprintf(stderr, "no Vulkan 1.3 GPU with graphics+present support found\n");
        std::exit(1);
    }

    vkGetPhysicalDeviceProperties(physicalDevice, &caps.props);

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(physicalDevice, &features);
    caps.fillModeNonSolid = features.fillModeNonSolid;

    if (!disableMeshShaders && hasDeviceExtension(physicalDevice, VK_EXT_MESH_SHADER_EXTENSION_NAME))
    {
        VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &meshFeatures;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

        VkPhysicalDeviceMeshShaderPropertiesEXT meshProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &meshProps;
        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        caps.meshShader = meshFeatures.meshShader && meshFeatures.taskShader;
        caps.maxMeshOutputVertices = meshProps.maxMeshOutputVertices;
        caps.maxMeshOutputPrimitives = meshProps.maxMeshOutputPrimitives;
        caps.maxTaskWorkGroupCount = meshProps.maxTaskWorkGroupCount[0];
    }
}

void VulkanContext::createDevice()
{
    float priority = 1.f;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    meshFeatures.meshShader = VK_TRUE;
    meshFeatures.taskShader = VK_TRUE;

    VkPhysicalDeviceVulkan13Features features13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.maintenance4 = VK_TRUE;
    if (caps.meshShader)
        features13.pNext = &meshFeatures;

    VkPhysicalDeviceVulkan12Features features12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.pNext = &features13;
    features12.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &features12;
    features2.features.fillModeNonSolid = caps.fillModeNonSolid ? VK_TRUE : VK_FALSE;

    std::vector<const char*> extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (caps.meshShader)
        extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);

    VkDeviceCreateInfo createInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.pNext = &features2;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueInfo;
    createInfo.enabledExtensionCount = uint32_t(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));
    volkLoadDevice(device);
    vkGetDeviceQueue(device, queueFamily, 0, &queue);
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    std::fprintf(stderr, "no suitable memory type\n");
    std::abort();
}

void VulkanContext::destroySwapchainResources()
{
    for (VkImageView view : swapchain.views)
        vkDestroyImageView(device, view, nullptr);
    for (VkSemaphore sem : swapchain.renderFinished)
        vkDestroySemaphore(device, sem, nullptr);
    swapchain.views.clear();
    swapchain.renderFinished.clear();
    swapchain.images.clear();

    Image& depth = swapchain.depth;
    if (depth.image)
    {
        vkDestroyImageView(device, depth.view, nullptr);
        vkDestroyImage(device, depth.image, nullptr);
        vkFreeMemory(device, depth.memory, nullptr);
        depth = Image();
    }
}

void VulkanContext::recreateSwapchain()
{
    VK_CHECK(vkDeviceWaitIdle(device));
    createSwapchain();
}

void VulkanContext::createSwapchain()
{
    VkSurfaceCapabilitiesKHR surfaceCaps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps));

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    VkSurfaceFormatKHR surfaceFormat = formats[0];
    for (const VkSurfaceFormatKHR& f : formats)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            surfaceFormat = f;
            break;
        }

    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D extent = surfaceCaps.currentExtent;
    if (extent.width == UINT32_MAX)
    {
        extent.width = std::clamp(uint32_t(width), surfaceCaps.minImageExtent.width, surfaceCaps.maxImageExtent.width);
        extent.height = std::clamp(uint32_t(height), surfaceCaps.minImageExtent.height, surfaceCaps.maxImageExtent.height);
    }

    uint32_t imageCount = std::max(surfaceCaps.minImageCount + 1, 3u);
    if (surfaceCaps.maxImageCount > 0)
        imageCount = std::min(imageCount, surfaceCaps.maxImageCount);

    VkSwapchainKHR oldSwapchain = swapchain.handle;

    VkSwapchainCreateInfoKHR createInfo = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = surfaceCaps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = oldSwapchain;

    VK_CHECK(vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain.handle));

    destroySwapchainResources();
    if (oldSwapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(device, oldSwapchain, nullptr);

    swapchain.format = surfaceFormat.format;
    swapchain.extent = extent;
    swapchain.minImageCount = surfaceCaps.minImageCount;
    swapchain.canReadback = (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;

    uint32_t count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain.handle, &count, nullptr));
    swapchain.images.resize(count);
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain.handle, &count, swapchain.images.data()));

    swapchain.views.resize(count);
    swapchain.renderFinished.resize(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        VkImageViewCreateInfo viewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = swapchain.images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchain.format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &swapchain.views[i]));

        VkSemaphoreCreateInfo semInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &swapchain.renderFinished[i]));
    }

    // depth buffer
    Image& depth = swapchain.depth;
    depth.format = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = depth.format;
    imageInfo.extent = {extent.width, extent.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &depth.image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, depth.image, &req);
    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = req.size;
    allocInfo.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &depth.memory));
    VK_CHECK(vkBindImageMemory(device, depth.image, depth.memory, 0));

    VkImageViewCreateInfo depthViewInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthViewInfo.image = depth.image;
    depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.format = depth.format;
    depthViewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &depthViewInfo, nullptr, &depth.view));
}

void VulkanContext::createFrames()
{
    for (Frame& frame : frames)
    {
        VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = queueFamily;
        VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &frame.commandPool));

        VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocInfo.commandPool = frame.commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &frame.commandBuffer));

        VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &frame.fence));

        VkSemaphoreCreateInfo semInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &frame.imageAcquired));
    }
}

Buffer VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible)
{
    Buffer result;
    result.size = std::max<VkDeviceSize>(size, 16);

    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = result.size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &result.buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, result.buffer, &req);

    VkMemoryAllocateFlagsInfo flagsInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &flagsInfo;
    allocInfo.allocationSize = req.size;
    allocInfo.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
        hostVisible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &result.memory));
    VK_CHECK(vkBindBufferMemory(device, result.buffer, result.memory, 0));

    if (hostVisible)
        VK_CHECK(vkMapMemory(device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped));

    VkBufferDeviceAddressInfo addressInfo = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = result.buffer;
    result.address = vkGetBufferDeviceAddress(device, &addressInfo);
    return result;
}

Buffer VulkanContext::createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buffer result = createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    if (size == 0)
        return result;

    // Upload in chunks so huge models do not need an equally huge staging buffer.
    const VkDeviceSize chunkSize = std::min<VkDeviceSize>(size, 64ull << 20);
    Buffer staging = createBuffer(chunkSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);

    VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = uploadPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));

    for (VkDeviceSize offset = 0; offset < size; offset += chunkSize)
    {
        VkDeviceSize bytes = std::min(chunkSize, size - offset);
        std::memcpy(staging.mapped, static_cast<const char*>(data) + offset, size_t(bytes));

        VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
        VkBufferCopy region = {0, offset, bytes};
        vkCmdCopyBuffer(cmd, staging.buffer, result.buffer, 1, &region);
        VK_CHECK(vkEndCommandBuffer(cmd));

        VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
    }

    vkFreeCommandBuffers(device, uploadPool, 1, &cmd);
    destroyBuffer(staging);
    return result;
}

void VulkanContext::destroyBuffer(Buffer& buffer)
{
    if (buffer.buffer)
    {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
        vkFreeMemory(device, buffer.memory, nullptr);
    }
    buffer = Buffer();
}
