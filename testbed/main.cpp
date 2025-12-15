#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <veekay/veekay.hpp>
#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <lodepng.h>

// Функция для поиска подходящего типа памяти
namespace veekay {
    namespace graphics {
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
            VkPhysicalDeviceMemoryProperties memProperties;
            vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);

            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
                if ((typeFilter & (1 << i)) &&
                    (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                    return i;
                }
            }

            std::cerr << "Failed to find suitable memory type!" << std::endl;
            veekay::app.running = false;
            return 0;
        }
    } // namespace graphics
} // namespace veekay


namespace {
    constexpr uint32_t max_models = 1024;
    constexpr uint32_t max_point_lights = 16;
    constexpr uint32_t max_spot_lights = 16;

    static inline VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
        return (a > 0) ? ((v + a - 1) & ~(a - 1)) : v;
    }

    static uint32_t g_ubo_align = 0;
    static uint32_t g_model_stride = 0;

    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
    };

    struct SceneUniforms {
        veekay::mat4 view_projection;
        veekay::vec3 view_position;
        float _pad0;
        veekay::vec3 ambient_light_intensity;
        float _pad1;
        veekay::vec3 sun_light_direction;
        float _pad2;
        veekay::vec3 sun_light_color;
        float _pad3;
        uint32_t point_lights_count;
        uint32_t spot_lights_count;
        float _pad4;
        float _pad5;
        veekay::mat4 shadow_projection; // Добавлено для теней
    };

    struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float _pad0;
        veekay::vec3 specular_color;
        float _pad1;
        float shininess;
        float _pad2;
        float _pad3;
        float _pad4;
    };

    struct PointLight {
        veekay::vec3 position;
        float radius;
        veekay::vec3 color;
        float _pad0;
    };

    struct SpotLight {
        veekay::vec3 position;
        float radius;
        veekay::vec3 direction;
        float angle;
        veekay::vec3 color;
        float _pad0;
    };

    struct Mesh {
        veekay::graphics::Buffer *vertex_buffer;
        veekay::graphics::Buffer *index_buffer;
        uint32_t indices;
    };

    struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        veekay::mat4 matrix() const;
    };

    struct SavedState {
        veekay::vec3 position;
        veekay::vec3 rotation;
        veekay::vec3 target;
    };

    struct Material {
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;

        veekay::graphics::Texture *albedo = nullptr;
        veekay::graphics::Texture *specular = nullptr;
        veekay::graphics::Texture *emissive = nullptr;
        VkSampler sampler = VK_NULL_HANDLE;

        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        Material material;
    };

    struct Camera {
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;
        veekay::vec3 position = {};
        veekay::vec3 rotation = {-45.0f, 0.0f, 0.0f};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        SavedState normal_view_state;
        SavedState look_at_view_state;

        veekay::mat4 view() const;

        veekay::mat4 look_at(veekay::vec3 at) const;

        veekay::mat4 view_projection(float aspect_ratio) const;

        void saveNormalViewState();

        void saveLookAtViewState(const veekay::vec3 &target);

        void restoreNormalViewState();

        void restoreLookAtViewState();
    };

    // Shadow mapping structure
    struct {
        // Объекты для изображения, куда будет записываться информация о глубине
        VkFormat depth_image_format;
        VkImage depth_image;
        VkDeviceMemory depth_image_memory;
        VkImageView depth_image_view;

        VkFramebuffer framebuffer;
        VkRenderPass render_pass;

        VkShaderModule vertex_shader; // Простой шейдер для трансформации геометрии

        // Объекты графического конвейера
        VkDescriptorSetLayout descriptor_set_layout;
        VkDescriptorSet descriptor_set;
        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        veekay::graphics::Buffer *uniform_buffer; // Буфер для матрицы проекции теней
        VkSampler sampler; // Специальный сэмплер для текстуры теней
        veekay::mat4 matrix; // Матрица проекции теней

        // Размер карты теней
        uint32_t size = 2048;
    } shadow;

    veekay::vec3 sun_light_direction{0.0f, 1.0f, 1.0f};

    PFN_vkCmdBeginRenderingKHR vkCmdBeginRenderingKHR;
    PFN_vkCmdEndRenderingKHR vkCmdEndRenderingKHR;

    VkRenderPass createShadowRenderPass(VkDevice device, VkFormat depthFormat) {
        VkAttachmentDescription depthAttachment{
            .format = depthFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };

        VkAttachmentReference depthAttachmentRef{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 0,
            .pDepthStencilAttachment = &depthAttachmentRef,
        };

        VkSubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        };

        VkSubpassDependency dependency2{
            .srcSubpass = 0,
            .dstSubpass = VK_SUBPASS_EXTERNAL,
            .srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        };

        VkSubpassDependency dependencies[] = {dependency, dependency2};

        VkRenderPassCreateInfo renderPassInfo{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &depthAttachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 2,
            .pDependencies = dependencies,
        };

        VkRenderPass renderPass;
        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow render pass!");
        }

        return renderPass;
    }

    inline namespace {
        Camera camera{
            .position = {0.0f, -2.5f, -3.0f}
        };
        bool look_at_view = false;
        veekay::vec3 target_look_at = {1.0f, 1.0f, 1.0f};
        std::vector<Model> models;
        std::vector<PointLight> point_lights;
        std::vector<SpotLight> spot_lights;
    }

    inline namespace {
        veekay::vec3 ambient_light = {1.f, 1.f, 1.f};
        veekay::vec3 sun_direction = {0.0f, 1.0f, -1.0f};
        veekay::vec3 sun_color = {0.2f, 0.2f, 0.2f};

        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;
        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout_global;
        VkDescriptorSetLayout descriptor_set_layout_material;
        VkDescriptorSet descriptor_set_global;
        std::vector<VkDescriptorSet> descriptor_sets_material;
        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;
        veekay::graphics::Buffer *scene_uniforms_buffer;
        veekay::graphics::Buffer *model_uniforms_buffer;
        veekay::graphics::Buffer *point_lights_buffer;
        veekay::graphics::Buffer *spot_lights_buffer;
        Mesh plane_mesh;
        Mesh cube_mesh;
        Mesh sphere_mesh;
        veekay::graphics::Texture *missing_texture;
        veekay::graphics::Texture *g_black1x1;
        veekay::graphics::Texture *g_white1x1;
        VkSampler missing_texture_sampler;
    }

    float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
    }

    veekay::mat4 Transform::matrix() const {
        auto t = veekay::mat4::translation(position) *
                 veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z)) *
                 veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y)) *
                 veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x)) *
                 veekay::mat4::scaling(scale);
        return t;
    }

    veekay::mat4 Camera::look_at(veekay::vec3 at) const {
        const veekay::vec3 forward = veekay::vec3::normalized(position - at);
        veekay::vec3 world_up = {0, 1, 0};
        veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(forward, world_up));
        veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(right, forward));

        const veekay::mat4 basis = {
            right.x, up.x, -forward.x, 0,
            right.y, up.y, -forward.y, 0,
            right.z, up.z, -forward.z, 0,
            0, 0, 0, 1
        };
        return veekay::mat4::translation(-position) * basis;
    }

    void Camera::saveNormalViewState() {
        normal_view_state.position = position;
        normal_view_state.rotation = rotation;
    }

    void Camera::saveLookAtViewState(const veekay::vec3 &target) {
        look_at_view_state.position = position;
        look_at_view_state.rotation = rotation;
        look_at_view_state.target = target;
    }

    void Camera::restoreNormalViewState() {
        position = normal_view_state.position;
        rotation = normal_view_state.rotation;
    }

    void Camera::restoreLookAtViewState() {
        position = look_at_view_state.position;
        rotation = look_at_view_state.rotation;
    }

    veekay::mat4 Camera::view() const {
        auto t = veekay::mat4::translation(-position);
        auto r = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x)) *
                 veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y)) *
                 veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
        r = veekay::mat4::transpose(r);
        return t * r;
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
        auto view_matrix = look_at_view ? look_at(target_look_at) : view();
        return view_matrix * projection;
    }

    VkShaderModule loadShaderModule(const char *path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = buffer.data(),
        };
        VkShaderModule result;
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }
        return result;
    }

    VkSampler makeSampler(VkDevice device,
                          VkFilter fmin, VkFilter fmag,
                          VkSamplerAddressMode addr = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                          float maxAniso = 16.0f) {
        VkSamplerCreateInfo sinfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = fmag,
            .minFilter = fmin,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = addr, .addressModeV = addr, .addressModeW = addr,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = maxAniso,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
        };
        VkSampler s{};
        if (vkCreateSampler(device, &sinfo, nullptr, &s) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan texture sampler\n";
            veekay::app.running = false;
            return nullptr;
        }
        return s;
    }

    veekay::graphics::Texture *makeSolidTexture(VkCommandBuffer cmd, uint32_t rgba) {
        return new veekay::graphics::Texture(cmd, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &rgba);
    }

    void createShadowMapImage(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;

        shadow.depth_image_format = VK_FORMAT_D32_SFLOAT;

        VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = shadow.depth_image_format,
            .extent = {shadow.size, shadow.size, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        if (vkCreateImage(device, &image_info, nullptr, &shadow.depth_image) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow map image\n";
            veekay::app.running = false;
            return;
        }

        VkMemoryRequirements mem_reqs;
        vkGetImageMemoryRequirements(device, shadow.depth_image, &mem_reqs);

        VkMemoryAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = veekay::graphics::findMemoryType(
                mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        };

        if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow.depth_image_memory) != VK_SUCCESS) {
            std::cerr << "Failed to allocate shadow map memory\n";
            veekay::app.running = false;
            return;
        }

        vkBindImageMemory(device, shadow.depth_image, shadow.depth_image_memory, 0);

        VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = shadow.depth_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = shadow.depth_image_format,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        if (vkCreateImageView(device, &view_info, nullptr, &shadow.depth_image_view) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow map image view\n";
            veekay::app.running = false;
            return;
        }

        // Создаем render pass для теней


        // Переводим изображение в правильный layout
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .image = shadow.depth_image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

        vkCmdBeginRenderingKHR = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
            vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR"));

        vkCmdEndRenderingKHR = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
            vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR"));

        // Сначала получаем свойства устройства и вычисляем выравнивание
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(physical_device, &props);
            g_ubo_align = uint32_t(props.limits.minUniformBufferOffsetAlignment);
            g_model_stride = uint32_t(align_up(sizeof(ModelUniforms), g_ubo_align));

            // Создаем основные буферы ДО их использования
            scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * g_model_stride,
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            point_lights_buffer = new veekay::graphics::Buffer(
                max_point_lights * sizeof(PointLight),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            spot_lights_buffer = new veekay::graphics::Buffer(
                max_spot_lights * sizeof(SpotLight),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }

        //  Создаем основные текстуры
        {
            g_black1x1 = makeSolidTexture(cmd, 0xff000000u);
            g_white1x1 = makeSolidTexture(cmd, 0xffffffffu);

            missing_texture_sampler = makeSampler(device,
                                                  VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                                  VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                  1.0f);
            std::vector<uint32_t> pixels(16 * 16);
            for (int y = 0; y < 16; ++y) {
                for (int x = 0; x < 16; ++x) {
                    bool black = ((x / 8) + (y / 8)) % 2 == 0;
                    pixels[y * 16 + x] = black ? 0xff000000u : 0xffff00ffu;
                }
            }
            missing_texture = new veekay::graphics::Texture(cmd, 16, 16, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
        }

        //  Создаем основной графический конвейер
        {
            vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
            if (!vertex_shader_module) {
                std::cerr << "Failed to load Vulkan vertex shader from file\n";
                veekay::app.running = false;
                return;
            }
            fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
            if (!fragment_shader_module) {
                std::cerr << "Failed to load Vulkan fragment shader from file\n";
                veekay::app.running = false;
                return;
            }
            VkPipelineShaderStageCreateInfo stage_infos[2];
            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertex_shader_module,
                .pName = "main",
            };
            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragment_shader_module,
                .pName = "main",
            };
            VkVertexInputBindingDescription buffer_binding{
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };
            VkVertexInputAttributeDescription attributes[] = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
                },
                {
                    .location = 1,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, normal),
                },
                {
                    .location = 2,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32_SFLOAT,
                    .offset = offsetof(Vertex, uv),
                },
            };
            VkPipelineVertexInputStateCreateInfo input_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &buffer_binding,
                .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
                .pVertexAttributeDescriptions = attributes,
            };
            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };
            VkPipelineRasterizationStateCreateInfo raster_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .lineWidth = 1.0f,
            };
            VkPipelineMultisampleStateCreateInfo sample_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = false,
                .minSampleShading = 1.0f,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = false,
                .alphaToOneEnable = false,
            };
            VkViewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(veekay::app.window_width),
                .height = static_cast<float>(veekay::app.window_height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            VkRect2D scissor{
                .offset = {0, 0},
                .extent = {veekay::app.window_width, veekay::app.window_height},
            };
            VkPipelineViewportStateCreateInfo viewport_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .pViewports = &viewport,
                .scissorCount = 1,
                .pScissors = &scissor,
            };
            VkPipelineDepthStencilStateCreateInfo depth_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = true,
                .depthWriteEnable = true,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                .depthBoundsTestEnable = false,
                .stencilTestEnable = false,
            };
            VkPipelineColorBlendAttachmentState attachment_info{
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                  VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT |
                                  VK_COLOR_COMPONENT_A_BIT,
            };
            VkPipelineColorBlendStateCreateInfo blend_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = false,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = 1,
                .pAttachments = &attachment_info
            };

            // Создаем descriptor pool
            {
                VkDescriptorPoolSize pools[] = {
                    {
                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 8,
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 8,
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 3 * max_models + 8,
                    },
                    {
                        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 8
                    },
                };
                VkDescriptorPoolCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    .maxSets = 1 + max_models + 1, // +1 для теневого дескриптора
                    .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                    .pPoolSizes = pools,
                };
                if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // Создаем descriptor set layout для глобальных uniform
            {
                VkDescriptorSetLayoutBinding bindings[] = {
                    {
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 2,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 3,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 4,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    }
                };
                VkDescriptorSetLayoutCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                    .pBindings = bindings,
                };
                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout_global) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // Создаем descriptor set layout для материалов
            {
                VkDescriptorSetLayoutBinding bindings[] = {
                    {
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                    {
                        .binding = 2,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                    },
                };
                VkDescriptorSetLayoutCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                    .pBindings = bindings,
                };
                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout_material) !=
                    VK_SUCCESS) {
                    std::cerr << "Failed to create material descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // Создаем pipeline layout
            {
                VkDescriptorSetLayout set_layouts[] = {descriptor_set_layout_global, descriptor_set_layout_material};
                VkPipelineLayoutCreateInfo layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 2,
                    .pSetLayouts = set_layouts,
                };
                if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan pipeline layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // Создаем основной графический pipeline
            VkGraphicsPipelineCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 2,
                .pStages = stage_infos,
                .pVertexInputState = &input_state_info,
                .pInputAssemblyState = &assembly_state_info,
                .pViewportState = &viewport_info,
                .pRasterizationState = &raster_info,
                .pMultisampleState = &sample_info,
                .pDepthStencilState = &depth_info,
                .pColorBlendState = &blend_info,
                .layout = pipeline_layout,
                .renderPass = veekay::app.vk_render_pass,
            };
            if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        //  Инициализация shadow mapping
        {
            // Создаем изображение для карты теней
            shadow.size = 2048;
            shadow.depth_image_format = VK_FORMAT_D32_SFLOAT;

            // Создаем изображение глубины
            VkImageCreateInfo image_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = shadow.depth_image_format,
                .extent = {shadow.size, shadow.size, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            };

            if (vkCreateImage(device, &image_info, nullptr, &shadow.depth_image) != VK_SUCCESS) {
                std::cerr << "Failed to create shadow depth image\n";
                veekay::app.running = false;
                return;
            }

            // Выделяем память для изображения
            VkMemoryRequirements mem_reqs;
            vkGetImageMemoryRequirements(device, shadow.depth_image, &mem_reqs);

            VkMemoryAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = mem_reqs.size,
                .memoryTypeIndex = veekay::graphics::findMemoryType(
                    mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            };

            if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow.depth_image_memory) != VK_SUCCESS) {
                std::cerr << "Failed to allocate shadow depth image memory\n";
                veekay::app.running = false;
                return;
            }

            vkBindImageMemory(device, shadow.depth_image, shadow.depth_image_memory, 0);

            // Создаем image view
            VkImageViewCreateInfo view_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = shadow.depth_image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = shadow.depth_image_format,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };

            if (vkCreateImageView(device, &view_info, nullptr, &shadow.depth_image_view) != VK_SUCCESS) {
                std::cerr << "Failed to create shadow depth image view\n";
                veekay::app.running = false;
                return;
            }

            shadow.vertex_shader = loadShaderModule("./shaders/shadow.vert.spv");
            if (!shadow.vertex_shader) {
                std::cerr << "Failed to load shadow vertex shader\n";
                veekay::app.running = false;
                return;
            }

            // Проверяем, что буферы созданы
            if (!model_uniforms_buffer) {
                std::cerr << "ERROR: model_uniforms_buffer is null! Cannot create shadow descriptors.\n";
                veekay::app.running = false;
                return;
            }

            VkDescriptorSetLayoutBinding bindings[] = {
                {
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                },
                {
                    .binding = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                },
            };

            VkDescriptorSetLayoutCreateInfo layout_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                .pBindings = bindings,
            };

            if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &shadow.descriptor_set_layout) !=
                VK_SUCCESS) {
                std::cerr << "Failed to create shadow descriptor set layout\n";
                veekay::app.running = false;
                return;
            }

            VkDescriptorSetAllocateInfo alloc_info2{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &shadow.descriptor_set_layout,
            };

            if (vkAllocateDescriptorSets(device, &alloc_info2, &shadow.descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate shadow descriptor set\n";
                veekay::app.running = false;
                return;
            }

            shadow.uniform_buffer = new veekay::graphics::Buffer(
                sizeof(veekay::mat4), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

            VkPipelineLayoutCreateInfo pipeline_layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &shadow.descriptor_set_layout,
            };

            if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &shadow.pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create shadow pipeline layout\n";
                veekay::app.running = false;
                return;
            }

            // Создаем конвейер для теней
            VkPipelineShaderStageCreateInfo stage_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = shadow.vertex_shader,
                .pName = "main",
            };

            VkVertexInputBindingDescription buffer_binding{
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            VkVertexInputAttributeDescription attributes[] = {
                {
                    .location = 0,
                    .binding = 0,
                    .format = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset = offsetof(Vertex, position),
                }
            };

            VkPipelineVertexInputStateCreateInfo input_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &buffer_binding,
                .vertexAttributeDescriptionCount = 1,
                .pVertexAttributeDescriptions = attributes,
            };

            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            VkPipelineRasterizationStateCreateInfo raster_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_FRONT_BIT,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .depthBiasEnable = VK_TRUE,
                .depthBiasConstantFactor = 1.25f,
                .depthBiasSlopeFactor = 1.75f,
                .lineWidth = 1.0f,
            };

            VkPipelineMultisampleStateCreateInfo multisample_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = VK_FALSE,
                .minSampleShading = 1.0f,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = VK_FALSE,
                .alphaToOneEnable = VK_FALSE,
            };

            VkViewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(shadow.size),
                .height = static_cast<float>(shadow.size),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };

            VkRect2D scissor{
                .offset = {0, 0},
                .extent = {shadow.size, shadow.size},
            };

            VkPipelineViewportStateCreateInfo viewport_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1,
                .pViewports = &viewport,
                .scissorCount = 1,
                .pScissors = &scissor,
            };

            VkPipelineDepthStencilStateCreateInfo depth_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
            };

            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            VkPipelineColorBlendStateCreateInfo colorBlendState{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = 0,
                .pAttachments = nullptr,
            };

            VkDynamicState dyn_states[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
                VK_DYNAMIC_STATE_DEPTH_BIAS
            };

            VkPipelineDynamicStateCreateInfo dyn_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .dynamicStateCount = sizeof(dyn_states) / sizeof(dyn_states[0]),
                .pDynamicStates = dyn_states,
            };

            VkPipelineRenderingCreateInfoKHR rendering_create_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
                .colorAttachmentCount = 0,
                .pColorAttachmentFormats = nullptr,
                .depthAttachmentFormat = shadow.depth_image_format,
                .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
            };


            VkGraphicsPipelineCreateInfo pipeline_info{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = &rendering_create_info, // Добавьте это
                .stageCount = 1,
                .pStages = &stage_info,
                .pVertexInputState = &input_state_info,
                .pInputAssemblyState = &assembly_state_info,
                .pViewportState = &viewport_info,
                .pRasterizationState = &raster_info,
                .pMultisampleState = &multisample_info,
                .pDepthStencilState = &depth_info,
                .pColorBlendState = &colorBlendState,
                .pDynamicState = &dyn_state_info,
                .layout = shadow.pipeline_layout,
                .renderPass = VK_NULL_HANDLE, // Для динамического рендеринга
                .subpass = 0,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = -1,
            };

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &shadow.pipeline) !=
                VK_SUCCESS) {
                std::cerr << "Failed to create shadow pipeline\n";
                veekay::app.running = false;
                return;
            }

            VkSamplerCreateInfo sampler_info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                .compareEnable = VK_TRUE,
                // Операция: (Ref <= Texture) ? 1.0 : 0.0
                // Если наша глубина меньше или равна глубине в карте, значит мы освещены
                .compareOp = VK_COMPARE_OP_LESS,
                .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
            };

            if (vkCreateSampler(device, &sampler_info, nullptr, &shadow.sampler) != VK_SUCCESS) {
                std::cerr << "Failed to create shadow sampler\n";
                veekay::app.running = false;
                return;
            }

            // Обновляем дескрипторы теней
            VkDescriptorBufferInfo buffer_infos[] = {
                {
                    .buffer = shadow.uniform_buffer->buffer,
                    .range = sizeof(veekay::mat4),
                },
                {
                    .buffer = model_uniforms_buffer->buffer,
                    .range = sizeof(ModelUniforms),
                }
            };

            VkWriteDescriptorSet write_infos[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = shadow.descriptor_set,
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[0],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = shadow.descriptor_set,
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &buffer_infos[1],
                }
            };

            vkUpdateDescriptorSets(device, 2, write_infos, 0, nullptr);
        }

        //  Создаем основной глобальный descriptor set
        {
            VkDescriptorSetAllocateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptor_set_layout_global,
            };
            if (vkAllocateDescriptorSets(device, &info, &descriptor_set_global) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan descriptor set\n";
                veekay::app.running = false;
                return;
            }
        }

        //  Добавляем текстуру теней к основным дескрипторам
        {
            VkDescriptorImageInfo shadow_image_info{
                .sampler = shadow.sampler,
                .imageView = shadow.depth_image_view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkWriteDescriptorSet shadow_write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptor_set_global,
                .dstBinding = 4,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &shadow_image_info,
            };

            vkUpdateDescriptorSets(device, 1, &shadow_write, 0, nullptr);
        }

        //  Создаем меши
        // NOTE: Plane mesh initialization
        {
            std::vector<Vertex> vertices = {
                {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };
            std::vector<uint32_t> indices = {
                0, 1, 2, 2, 3, 0
            };
            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                vertices.size() * sizeof(Vertex), vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            plane_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Cube mesh initialization
        {
            std::vector<Vertex> vertices = {
                {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
                {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
                {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            };
            std::vector<uint32_t> indices = {
                0, 1, 2, 2, 3, 0,
                4, 5, 6, 6, 7, 4,
                8, 9, 10, 10, 11, 8,
                12, 13, 14, 14, 15, 12,
                16, 17, 18, 18, 19, 16,
                20, 21, 22, 22, 23, 20,
            };
            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
                vertices.size() * sizeof(Vertex), vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            cube_mesh.index_buffer = new veekay::graphics::Buffer(
                indices.size() * sizeof(uint32_t), indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            cube_mesh.indices = uint32_t(indices.size());
        } {
            // NOTE: Sphere mesh initialization
            std::vector<Vertex> sphere_vertices;
            std::vector<uint32_t> sphere_indices;
            const int stacks = 20;
            const int slices = 20;
            const float radius = 0.5f;

            for (int i = 0; i <= stacks; ++i) {
                float V = i / (float) stacks;
                float phi = V * M_PI;

                for (int j = 0; j <= slices; ++j) {
                    float U = j / (float) slices;
                    float theta = U * (M_PI * 2);

                    float x = cosf(theta) * sinf(phi);
                    float y = cosf(phi);
                    float z = sinf(theta) * sinf(phi);

                    sphere_vertices.push_back({
                        {x * radius, y * radius, z * radius}, // Позиция
                        {x, y, z}, // Нормаль
                        {U, V} // UV
                    });
                }
            }

            for (int i = 0; i < stacks; ++i) {
                for (int j = 0; j < slices; ++j) {
                    int p1 = (i * (slices + 1)) + j;
                    int p2 = p1 + (slices + 1);

                    sphere_indices.push_back(p1);
                    sphere_indices.push_back(p2);
                    sphere_indices.push_back(p1 + 1);

                    sphere_indices.push_back(p1 + 1);
                    sphere_indices.push_back(p2);
                    sphere_indices.push_back(p2 + 1);
                }
            }

            sphere_mesh.vertex_buffer = new veekay::graphics::Buffer(
                sphere_vertices.size() * sizeof(Vertex), sphere_vertices.data(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            sphere_mesh.index_buffer = new veekay::graphics::Buffer(
                sphere_indices.size() * sizeof(uint32_t), sphere_indices.data(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            sphere_mesh.indices = uint32_t(sphere_indices.size());
        }

        //  Добавляем источники света
        // NOTE: Добавляем точечные источники света
        point_lights.emplace_back(PointLight{
            .position = {0.0f, 2.0f, 0.0f},
            .radius = 8.0f,
            .color = {1.0f, 0.8f, 0.2f},
        });
        point_lights.emplace_back(PointLight{
            .position = {-2.5f, 1.0f, -1.5f},
            .radius = 6.0f,
            .color = {1.0f, 0.3f, 0.3f},
        });
        point_lights.emplace_back(PointLight{
            .position = {2.0f, 1.5f, -0.5f},
            .radius = 7.0f,
            .color = {0.3f, 1.0f, 0.3f},
        });
        point_lights.emplace_back(PointLight{
            .position = {0.5f, 0.8f, 1.5f},
            .radius = 5.0f,
            .color = {0.3f, 0.3f, 1.0f},
        });
        point_lights.emplace_back(PointLight{
            .position = {-4.0f, 1.2f, 3.0f},
            .radius = 10.0f,
            .color = {0.8f, 0.5f, 1.0f},
        });

        // NOTE: Добавляем прожекторы
        spot_lights.emplace_back(SpotLight{
            .position = {3.0f, 3.0f, 0.0f},
            .radius = 15.0f,
            .direction = {-0.7f, -0.7f, 0.0f},
            .angle = cosf(toRadians(25.0f)),
            .color = {1.0f, 0.2f, 0.2f},
        });
        spot_lights.emplace_back(SpotLight{
            .position = {-3.0f, 2.0f, 3.0f},
            .radius = 12.0f,
            .direction = {0.5f, -0.3f, -0.8f},
            .angle = cosf(toRadians(35.0f)),
            .color = {0.2f, 1.0f, 0.2f},
        });
        spot_lights.emplace_back(SpotLight{
            .position = {0.0f, 4.0f, 4.0f},
            .radius = 18.0f,
            .direction = {0.0f, -0.5f, -1.0f},
            .angle = cosf(toRadians(45.0f)),
            .color = {0.2f, 0.2f, 1.0f},
        });
        spot_lights.emplace_back(SpotLight{
            .position = {0.0f, -1.0f, 2.0f},
            .radius = 10.0f,
            .direction = {0.0f, 1.0f, -0.5f},
            .angle = cosf(toRadians(40.0f)),
            .color = {1.0f, 1.0f, 0.5f},
        });

        //  Добавляем модели в сцену с материалами
        models.clear();
        // 0: Пол
        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{},
            .material = Material{
                .albedo_color = {0.3f, 0.3f, 0.3f},
                .specular_color = {0.5f, 0.5f, 0.5f},
                .shininess = 90.0f,
            }
        });

        //  Стоящий Куб
        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{.position = {-2.0f, -0.5f, -1.5f}},
            .material = Material{
                .albedo_color = {1.f, 1.f, 1.f},
                .specular_color = {1.0f, 1.0f, 1.0f},
                .shininess = 64.0f
            }
        });

        //: Сфера над полом
        models.emplace_back(Model{
            .mesh = sphere_mesh,
            .transform = Transform{.position = {0.f, -2.0f, 0.0f}}, // Справа и выше пола
            .material = Material{
                .albedo_color = {1.f, 1.f, 1.f},
                .specular_color = {1.0f, 1.0f, 1.0f},
                .shininess = 128.0f
            }
        });

        //  Создаем descriptor sets для материалов
        {
            descriptor_sets_material.resize(models.size());
            if (!models.empty()) {
                std::vector<VkDescriptorSetLayout> layouts(models.size(), descriptor_set_layout_material);
                VkDescriptorSetAllocateInfo ainfo{
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    .descriptorPool = descriptor_pool,
                    .descriptorSetCount = (uint32_t) layouts.size(),
                    .pSetLayouts = layouts.data(),
                };
                if (vkAllocateDescriptorSets(device, &ainfo, descriptor_sets_material.data()) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate material descriptor sets\n";
                    veekay::app.running = false;
                    return;
                }
                for (size_t i = 0; i < models.size(); ++i) {
                    models[i].material.set = descriptor_sets_material[i];
                }
            }
        }

        //  Загружаем текстуры для материалов и настраиваем дескрипторы
        {
            auto load_png = [&](const char *path) -> veekay::graphics::Texture * {
                uint32_t w = 0, h = 0;
                std::vector<unsigned char> pixels;
                unsigned err = lodepng::decode(pixels, w, h, path);
                if (err != 0 || w == 0 || h == 0) {
                    return nullptr;
                }
                return new veekay::graphics::Texture(cmd, w, h, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
            };

            for (size_t i = 0; i < models.size(); ++i) {
                auto &M = models[i].material;
                if (i == 0) {
                    M.albedo = load_png("./assets/1234.png");
                }

                if (i == 1) {
                    M.albedo = load_png("./assets/11.png");
                }

                if (i == 2) {
                    M.albedo = load_png("./assets/1234.png");
                }


                if (!M.albedo) M.albedo = missing_texture;
                M.specular = g_white1x1;
                M.emissive = g_black1x1;

                M.sampler = makeSampler(device, VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                                        VK_SAMPLER_ADDRESS_MODE_REPEAT, 16.0f);

                VkDescriptorImageInfo i_alb{
                    .sampler = (M.albedo == missing_texture) ? missing_texture_sampler : M.sampler,
                    .imageView = M.albedo->view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };
                VkDescriptorImageInfo i_spc{
                    .sampler = M.sampler,
                    .imageView = M.specular->view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };
                VkDescriptorImageInfo i_emi{
                    .sampler = M.sampler,
                    .imageView = M.emissive->view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };

                VkWriteDescriptorSet writes[] = {
                    {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = M.set,
                        .dstBinding = 0,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = &i_alb
                    },
                    {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = M.set,
                        .dstBinding = 1,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = &i_spc
                    },
                    {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = M.set,
                        .dstBinding = 2,
                        .dstArrayElement = 0,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = &i_emi
                    },
                };
                vkUpdateDescriptorSets(device, sizeof(writes) / sizeof(writes[0]), writes, 0, nullptr);
            }
        }

        //  Обновляем глобальные дескрипторы (uniform буферы и storage буферы)
        {
            VkDescriptorBufferInfo buffer_infos[] = {
                {
                    .buffer = scene_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(SceneUniforms),
                },
                {
                    .buffer = model_uniforms_buffer->buffer,
                    .offset = 0,
                    .range = sizeof(ModelUniforms),
                },
                {
                    .buffer = point_lights_buffer->buffer,
                    .range = max_point_lights * sizeof(PointLight)
                },
                {
                    .buffer = spot_lights_buffer->buffer,
                    .range = max_spot_lights * sizeof(SpotLight)
                }
            };
            VkWriteDescriptorSet write_infos[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set_global,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffer_infos[0],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set_global,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .pBufferInfo = &buffer_infos[1],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set_global,
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[2],
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set_global,
                    .dstBinding = 3,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_infos[3],
                },

            };
            vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                                   write_infos, 0, nullptr);
        }
    }

    void shutdown() {
        VkDevice &device = veekay::app.vk_device;

        // Очистка ресурсов теней
        vkDestroyFramebuffer(device, shadow.framebuffer, nullptr);
        vkDestroyRenderPass(device, shadow.render_pass, nullptr);
        vkDestroySampler(device, shadow.sampler, nullptr);
        vkDestroyPipeline(device, shadow.pipeline, nullptr);
        vkDestroyPipelineLayout(device, shadow.pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, shadow.descriptor_set_layout, nullptr);
        vkDestroyShaderModule(device, shadow.vertex_shader, nullptr);

        delete shadow.uniform_buffer;

        vkDestroyImageView(device, shadow.depth_image_view, nullptr);
        vkDestroyImage(device, shadow.depth_image, nullptr);
        vkFreeMemory(device, shadow.depth_image_memory, nullptr);

        for (auto &model: models) {
            if (model.material.sampler) {
                vkDestroySampler(device, model.material.sampler, nullptr);
            }
            if (model.material.albedo && model.material.albedo != missing_texture) {
                delete model.material.albedo;
            }
            if (model.material.specular && model.material.specular != g_white1x1) {
                delete model.material.specular;
            }
            if (model.material.emissive && model.material.emissive != g_black1x1) {
                delete model.material.emissive;
            }
        }

        vkDestroySampler(device, missing_texture_sampler, nullptr);
        delete missing_texture;
        delete g_white1x1;
        delete g_black1x1;
        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;
        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;
        delete sphere_mesh.index_buffer;
        delete sphere_mesh.vertex_buffer;
        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;
        delete point_lights_buffer;
        delete spot_lights_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout_material, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout_global, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        if (models.size() > 2) {
            float orbit_radius = 2.5f;
            float speed = 2.0f;

            models[2].transform.position.x = orbit_radius * cosf(time * speed);
            models[2].transform.position.z = orbit_radius * sinf(time * speed);

            models[2].transform.rotation.y = time * 90.0f;
        }
        // Отрисовка GUI с вкладками
        ImGui::Begin("Settings"); // Создаем главное окно настроек

        if (ImGui::BeginTabBar("MainTabs")) {
            // Начало панели вкладок

            // --- Вкладка 1: Камера ---
            if (ImGui::BeginTabItem("Camera")) {
                static bool prev_look_at_view = look_at_view;
                ImGui::Checkbox("Look At Mode", &look_at_view);

                if (look_at_view != prev_look_at_view) {
                    if (look_at_view) camera.saveNormalViewState();
                    else camera.saveLookAtViewState(target_look_at);
                    prev_look_at_view = look_at_view;
                }

                if (look_at_view) {
                    ImGui::DragFloat3("Target Point", target_look_at.elements, 0.1f);
                }

                ImGui::Separator();
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Camera Stats:");
                ImGui::Text("Pos: (%.2f, %.2f, %.2f)", camera.position.x, camera.position.y, camera.position.z);
                if (!look_at_view) {
                    ImGui::Text("Rot: (%.1f, %.1f, %.1f)", camera.rotation.x, camera.rotation.y, camera.rotation.z);
                }

                ImGui::EndTabItem();
            }

            // --- Вкладка 2: Освещение ---
            if (ImGui::BeginTabItem("Lighting")) {
                ImGui::Text("Environment");
                ImGui::ColorEdit3("Ambient Color", &ambient_light.x);
                ImGui::ColorEdit3("Sun Color", &sun_color.x);
                if (ImGui::SliderFloat3("Sun Dir", &sun_light_direction.x, -1.0f, 1.0f)) {
                    sun_direction = veekay::vec3::normalized(sun_light_direction);
                }

                ImGui::Separator();

                if (ImGui::CollapsingHeader("Point Lights")) {
                    for (size_t i = 0; i < point_lights.size(); ++i) {
                        ImGui::PushID((int) i);
                        if (ImGui::TreeNode((void *) (intptr_t) i, "Light #%d", (int) i)) {
                            ImGui::SliderFloat3("Pos", reinterpret_cast<float *>(&point_lights[i].position), -10.f,
                                                10.f);
                            ImGui::SliderFloat("Radius", &point_lights[i].radius, 0.1f, 20.f);
                            ImGui::ColorEdit3("Color", reinterpret_cast<float *>(&point_lights[i].color));
                            ImGui::TreePop();
                        }
                        ImGui::PopID();
                    }
                }

                if (ImGui::CollapsingHeader("Spot Lights")) {
                    for (size_t i = 0; i < spot_lights.size(); ++i) {
                        ImGui::PushID((int) i + 100);
                        if (ImGui::TreeNode((void *) (intptr_t) i, "Spot #%d", (int) i)) {
                            ImGui::SliderFloat3("Pos", reinterpret_cast<float *>(&spot_lights[i].position), -10.f,
                                                10.f);
                            ImGui::SliderFloat3("Dir", reinterpret_cast<float *>(&spot_lights[i].direction), -1.f, 1.f);

                            float angle_deg = acosf(spot_lights[i].angle) * 180.0f / float(M_PI);
                            if (ImGui::SliderFloat("Angle", &angle_deg, 0.f, 90.f)) {
                                spot_lights[i].angle = cosf(toRadians(angle_deg));
                            }

                            ImGui::ColorEdit3("Color", reinterpret_cast<float *>(&spot_lights[i].color));
                            ImGui::TreePop();
                        }
                        ImGui::PopID();
                    }
                }
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar(); // Завершение панели вкладок
        }
        ImGui::End();

        //  Обработка ввода (камера)
        ImGuiIO &io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            using namespace veekay::input;
            if (mouse::isButtonDown(mouse::Button::left)) {
                auto move_delta = mouse::cursorDelta();
                if (look_at_view) {
                    // Логика вращения вокруг цели
                    veekay::mat4 current_view = camera.look_at(target_look_at);
                    veekay::vec3 right = veekay::vec3::normalized({
                        current_view.elements[0][0], current_view.elements[1][0], current_view.elements[2][0]
                    });
                    veekay::vec3 up = veekay::vec3::normalized({
                        current_view.elements[0][1], current_view.elements[1][1], current_view.elements[2][1]
                    });
                    target_look_at += right * move_delta.x * 0.01f;
                    target_look_at += up * -move_delta.y * 0.01f;
                } else {
                    // Обычное вращение камеры
                    camera.rotation.y += move_delta.x * 0.1f;
                    camera.rotation.x += move_delta.y * 0.1f;
                    camera.rotation.x = std::clamp(camera.rotation.x, -90.0f, 90.0f);
                }
            }
        }

        // Управление клавиатурой (WASD)
        veekay::mat4 current_view = look_at_view ? camera.look_at(target_look_at) : camera.view();
        veekay::vec3 right = veekay::vec3::normalized({
            current_view.elements[0][0], current_view.elements[1][0], current_view.elements[2][0]
        });
        veekay::vec3 up = veekay::vec3::normalized({
            current_view.elements[0][1], current_view.elements[1][1], current_view.elements[2][1]
        });
        veekay::vec3 front = veekay::vec3::normalized({
            current_view.elements[0][2], current_view.elements[1][2], current_view.elements[2][2]
        });

        float move_speed = 0.1f;
        if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) camera.position += front * move_speed;
        if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) camera.position -= front * move_speed;
        if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) camera.position += right * move_speed;
        if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) camera.position -= right * move_speed;
        if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::q)) camera.position += up * move_speed;
        if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::z)) camera.position -= up * move_speed;

        //  Обновление юниформ-буферов
        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio),
            .view_position = camera.position,
            .ambient_light_intensity = ambient_light,
            .sun_light_direction = sun_light_direction,
            .sun_light_color = sun_color,
            .point_lights_count = static_cast<uint32_t>(point_lights.size()),
            .spot_lights_count = static_cast<uint32_t>(spot_lights.size())
        };

        // Обновление матриц моделей
        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0; i < models.size(); ++i) {
            model_uniforms[i].model = models[i].transform.matrix();
            model_uniforms[i].shininess = models[i].material.shininess;
            model_uniforms[i].specular_color = models[i].material.specular_color;
            model_uniforms[i].albedo_color = models[i].material.albedo_color;
        }

        // Shadow mapping projection
        veekay::vec3 sun_dir_norm = veekay::vec3::normalized(scene_uniforms.sun_light_direction);
        veekay::vec3 shadow_eye = camera.position - (sun_dir_norm * 20.0f);
        shadow.matrix = veekay::mat4::lookAt(shadow_eye, camera.position) *
                        veekay::mat4::ortho(-20.0f, 20.0f, -20.0f, 20.0f, 1.0f, 100.0f);

        scene_uniforms.shadow_projection = shadow.matrix;

        // Копирование данных в GPU буферы
        memcpy(shadow.uniform_buffer->mapped_region, &shadow.matrix, sizeof(veekay::mat4));
        memcpy(scene_uniforms_buffer->mapped_region, &scene_uniforms, sizeof(SceneUniforms));

        uint8_t *base = static_cast<uint8_t *>(model_uniforms_buffer->mapped_region);
        for (size_t i = 0; i < model_uniforms.size(); ++i) {
            memcpy(base + i * g_model_stride, &model_uniforms[i], sizeof(ModelUniforms));
        }

        if (!point_lights.empty()) {
            memcpy(point_lights_buffer->mapped_region, point_lights.data(), point_lights.size() * sizeof(PointLight));
        }
        if (!spot_lights.empty()) {
            memcpy(spot_lights_buffer->mapped_region, spot_lights.data(), spot_lights.size() * sizeof(SpotLight));
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0); {
            // NOTE: Start recording rendering commands
            VkCommandBufferBeginInfo info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            vkBeginCommandBuffer(cmd, &info);
        }

        // Проход для записи глубины в карту теней с использованием динамического рендеринга
        {
            // Барьер для перевода изображения из SHADER_READ_ONLY_OPTIMAL в DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = shadow.depth_image,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };

            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);

            // Настройки динамического рендеринга для теней
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

            VkRenderingAttachmentInfoKHR depth_attachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
                .imageView = shadow.depth_image_view,
                .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, // Очистить карту перед рисованием
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE, // Сохранить результат
                .clearValue = clear_depth, // Очищаем значением 1.0 (макс. глубина)
            };

            VkRenderingInfoKHR rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
                .renderArea = {{0, 0}, {shadow.size, shadow.size}},
                .layerCount = 1,
                .colorAttachmentCount = 0,
                .pColorAttachments = nullptr,
                .pDepthAttachment = &depth_attachment,
                .pStencilAttachment = nullptr,
            };

            vkCmdBeginRenderingKHR(cmd, &rendering_info);

            // Устанавливаем viewport и scissor
            VkViewport viewport{
                .x = 0.0f, .y = 0.0f,
                .width = static_cast<float>(shadow.size),
                .height = static_cast<float>(shadow.size),
                .minDepth = 0.0f, .maxDepth = 1.0f,
            };
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{{0, 0}, {shadow.size, shadow.size}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            // Устанавливаем смещение глубины
            vkCmdSetDepthBias(cmd, 1.25f, 0.0f, 1.75f);

            // Биндим конвейер теней
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline);

            VkDeviceSize zero_offset = 0;
            VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
            VkBuffer current_index_buffer = VK_NULL_HANDLE;

            // Рисуем все модели
            for (size_t i = 0, n = models.size(); i < n; ++i) {
                const Model &model = models[i];
                const Mesh &mesh = model.mesh;

                if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                    current_vertex_buffer = mesh.vertex_buffer->buffer;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
                }

                if (current_index_buffer != mesh.index_buffer->buffer) {
                    current_index_buffer = mesh.index_buffer->buffer;
                    vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
                }

                uint32_t offset = i * g_model_stride;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow.pipeline_layout,
                                        0, 1, &shadow.descriptor_set, 1, &offset);

                vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
            }

            // Заканчиваем динамический рендеринг
            vkCmdEndRenderingKHR(cmd);

            // Барьер для перевода обратно в SHADER_READ_ONLY_OPTIMAL
            {
                VkImageMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = shadow.depth_image,
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1
                    }
                };

                vkCmdPipelineBarrier(cmd,
                                     VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                     0, 0, nullptr, 0, nullptr, 1, &barrier);
            }
        } {
            // NOTE: Use current swapchain framebuffer and clear it
            VkClearValue clear_color{.color = {{0.05f, 0.05f, 0.05f, 1.0f}}};
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
            VkClearValue clear_values[] = {clear_color, clear_depth};
            VkRenderPassBeginInfo info{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .renderPass = veekay::app.vk_render_pass,
                .framebuffer = framebuffer,
                .renderArea = {
                    .extent = {
                        static_cast<uint32_t>(veekay::app.window_width),
                        static_cast<uint32_t>(veekay::app.window_height)
                    },
                },
                .clearValueCount = 2,
                .pClearValues = clear_values,
            };
            vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDeviceSize zero_offset = 0;
        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            const Mesh &mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            uint32_t offset = i * g_model_stride;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor_set_global, 1, &offset);

            VkDescriptorSet mat_set = models[i].material.set;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    1, 1, &mat_set, 0, nullptr);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }
} // namespace

int main() {
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}
