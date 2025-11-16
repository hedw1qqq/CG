#include "veekay/input.hpp"

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "veekay/graphics.hpp"
#include "veekay/veekay.hpp"
#include "imgui.h"
#include "lodepng.h"

namespace {
    constexpr uint32_t max_models = 1024;

    static inline VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) {
        return (a > 0) ? ((v + a - 1) & ~(a - 1)) : v;
    }

    static uint32_t g_ubo_align = 0;
    static uint32_t g_model_stride = 0;

    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
        // NOTE: You can add more attributes
    };

    struct PointLightBufferHeader {
        uint32_t count;
        uint32_t _pad1;
        uint32_t _pad2;
        uint32_t _pad3;
    };

    struct SpotLightBufferHeader {
        uint32_t count;
        uint32_t _pad1;
        uint32_t _pad2;
        uint32_t _pad3;
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
    };

    struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float _pad0;
        veekay::vec3 specular_color;
        float shininess;
        float _pad1;
        float _pad2;
    };

    struct Mesh {
        veekay::graphics::Buffer *vertex_buffer = nullptr;
        veekay::graphics::Buffer *index_buffer = nullptr;
        uint32_t indices = 0;
    };

    struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        // NOTE: Model matrix (translation, rotation and scaling)
        veekay::mat4 matrix() const;
    };

    struct PointLight {
        veekay::vec3 position;
        float _pad0;
        veekay::vec3 color;
        float intensity;
    };

    struct SpotLight {
        veekay::vec3 position;
        float _pad0;
        veekay::vec3 direction;
        float cutoff_angle;
        veekay::vec3 color;
        float outer_cutoff_angle;
        float intensity;
        float _pad1;
        float _pad2;
        float _pad3;
    };


    // материал для текстур и набора дескрипторов
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
        veekay::vec3 rotation = {};
        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        // Сохраненные состояния для каждого режима
        struct SavedState {
            veekay::vec3 position;
            veekay::vec3 rotation;
            veekay::vec3 target;
        };

        SavedState normal_view_state{};
        SavedState look_at_view_state{};

        // NOTE: View matrix of camera (inverse of a transform)
        veekay::mat4 view() const;

        veekay::mat4 look_at(veekay::vec3 at) const;

        // NOTE: View and projection composition
        veekay::mat4 view_projection(float aspect_ratio, const veekay::mat4 &view) const;

        // Методы для сохранения/восстановления состояния
        void saveNormalViewState() {
            normal_view_state.position = position;
            normal_view_state.rotation = rotation;
        }

        void saveLookAtViewState(const veekay::vec3 &target) {
            look_at_view_state.position = position;
            look_at_view_state.rotation = rotation;
            look_at_view_state.target = target;
        }

        void restoreNormalViewState() {
            position = normal_view_state.position;
            rotation = normal_view_state.rotation;
        }

        void restoreLookAtViewState() {
            position = look_at_view_state.position;
            rotation = look_at_view_state.rotation;
        }
    };

    // NOTE: Scene objects
    inline namespace {
        Camera camera{.position = {0.0f, -0.5f, -3.0f}};
        std::vector<Model> models;
    }

    // NOTE: ImGui variables
    inline namespace {
        bool look_at_view = false;
        veekay::vec3 target_look_at{1, 1, 1};
    }

    // NOTE: Vulkan objects
    inline namespace {
        VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
        VkShaderModule fragment_shader_module = VK_NULL_HANDLE;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout_global = VK_NULL_HANDLE; // set = 0
        VkDescriptorSetLayout descriptor_set_layout_material = VK_NULL_HANDLE; // set = 1

        VkDescriptorSet descriptor_set_global = VK_NULL_HANDLE; // set = 0
        std::vector<VkDescriptorSet> descriptor_sets_material; // set = 1, per-model

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        // Параметры освещения
        veekay::vec3 ambient_light = {0.2f, 0.2f, 0.2f};
        veekay::vec3 sun_direction = veekay::vec3::normalized({0.3f, 1.0f, 0.5f});
        veekay::vec3 sun_color = {1.0f, 1.0f, 0.9f};

        // Точечные источники света
        std::vector<PointLight> point_lights;
        veekay::graphics::Buffer *point_lights_buffer = nullptr;

        // Прожекторные источники света
        std::vector<SpotLight> spot_lights;
        veekay::graphics::Buffer *spot_lights_buffer = nullptr;

        veekay::graphics::Buffer *scene_uniforms_buffer = nullptr;
        veekay::graphics::Buffer *model_uniforms_buffer = nullptr;

        Mesh plane_mesh;
        Mesh cube_mesh;

        // Текстуры по умолчанию
        veekay::graphics::Texture *missing_texture = nullptr;
        veekay::graphics::Texture *g_black1x1 = nullptr;
        veekay::graphics::Texture *g_white1x1 = nullptr;
        VkSampler missing_texture_sampler = VK_NULL_HANDLE;
    }

    static inline float toRadians(float degrees) { return degrees * float(M_PI) / 180.0f; }


    veekay::mat4 mat4_inverse_rigid(const veekay::mat4 &m) {
        veekay::mat4 result = veekay::mat4::identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) result[j][i] = m[i][j];
        }
        // -R^T * t
        veekay::vec3 t = {m[3][0], m[3][1], m[3][2]};
        result[3][0] = -(result[0][0] * t.x + result[1][0] * t.y + result[2][0] * t.z);
        result[3][1] = -(result[0][1] * t.x + result[1][1] * t.y + result[2][1] * t.z);
        result[3][2] = -(result[0][2] * t.x + result[1][2] * t.y + result[2][2] * t.z);
        result[3][3] = 1.0f;
        return result;
    }

    veekay::mat4 Transform::matrix() const {
        // Порядок: T * Ry * Rx * Rz * S
        // (сначала scale, потом rotate (Y, X, Z), потом translate)
        auto t = veekay::mat4::translation(position);
        auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x));
        auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y));
        auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
        auto s = veekay::mat4::scaling(scale);
        // Model = T * Ry * Rx * Rz * S
        return t * ry * rx * rz * s;
    }

    veekay::mat4 Camera::view() const {
        auto t = veekay::mat4::translation(-position);
        auto rotX = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
        auto rotY = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
        auto rotZ = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
        auto r = rotY * rotX * rotZ;
        return t * r;
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

    veekay::mat4 Camera::view_projection(const float aspect_ratio, const veekay::mat4 &view) const {
        const auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
        return view * projection;
    }

    // NOTE: Loads shader byte code from file
    // NOTE: Your shaders are compiled via CMake with this code too, look it up
    VkShaderModule loadShaderModule(const char *path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) { return VK_NULL_HANDLE; }
        size_t size = size_t(file.tellg());
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = size,
            .pCode = buffer.data(),
        };

        VkShaderModule result = VK_NULL_HANDLE;
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        return result;
    }

    VkSampler makeSampler(VkDevice device,
                          VkFilter fmin, VkFilter fmag,
                          VkSamplerAddressMode addr = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                          float maxAniso = 16.0f) {
        VkSamplerCreateInfo sinfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = fmag, // Фильтрация если плотность текселей меньше
            .minFilter = fmin, // Фильтрация если плотность больше
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, // Фильтрация мип-мапов
            // Что делать, если по какой-то из осей вышли за границы текстурных коорд-т
            .addressModeU = addr, .addressModeV = addr, .addressModeW = addr,
            .anisotropyEnable = VK_TRUE, // Включить анизотропную фильтрацию?
            .maxAnisotropy = maxAniso, // Кол-во сэмплов анизотропной фильтрации
            .minLod = 0.0f, // Минимальный уровень мипа
            .maxLod = VK_LOD_CLAMP_NONE, // Максимальный уровень мипа
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

    void initialize(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

        // NOTE: Build graphics pipeline
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
        // NOTE: Vertex shader stage
        stage_infos[0] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
        };
        // NOTE: Fragment shader stage
        stage_infos[1] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        };

        // NOTE: How many bytes does a vertex take?
        VkVertexInputBindingDescription buffer_binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        // NOTE: Declare vertex attributes
        VkVertexInputAttributeDescription attributes[] = {
            {
                .location = 0, // NOTE: First attribute
                .binding = 0, // NOTE: First vertex buffer
                .format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
                .offset = (uint32_t) offsetof(Vertex, position) // NOTE: Offset of "position" field in a Vertex struct
            },
            {
                .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = (uint32_t) offsetof(Vertex, normal)
            },
            {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = (uint32_t) offsetof(Vertex, uv)},
        };

        // NOTE: Describe inputs
        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = (uint32_t) (sizeof(attributes) / sizeof(attributes[0])),
            .pVertexAttributeDescriptions = attributes,
        };

        // NOTE: Every three vertices make up a triangle,
        // so our vertex buffer contains a "list of triangles"
        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        // NOTE: Declare clockwise triangle order as front-facing
        // Discard triangles that are facing away
        // Fill triangles, don't draw lines instaed
        VkPipelineRasterizationStateCreateInfo raster_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .lineWidth = 1.0f,
        };

        // NOTE: Use 1 sample per pixel
        VkPipelineMultisampleStateCreateInfo sample_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
            .minSampleShading = 1.0f,
        };
        VkViewport viewport{
            .x = 0.0f, .y = 0.0f,
            .width = (float) veekay::app.window_width,
            .height = (float) veekay::app.window_height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        VkRect2D scissor{.offset = {0, 0}, .extent = {veekay::app.window_width, veekay::app.window_height}};

        // NOTE: Let rasterizer draw on the entire window
        VkPipelineViewportStateCreateInfo viewport_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .pViewports = &viewport,
            .scissorCount = 1, .pScissors = &scissor,
        };

        // NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
        VkPipelineDepthStencilStateCreateInfo depth_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        };

        // NOTE: Let fragment shader write all the color channels
        VkPipelineColorBlendAttachmentState attachment_info{
            .blendEnable = VK_FALSE,
            .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT

        };

        // NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
        VkPipelineColorBlendStateCreateInfo blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &attachment_info,
        }; {
            g_black1x1 = makeSolidTexture(cmd, 0xff000000u);
            g_white1x1 = makeSolidTexture(cmd, 0xffffffffu);
        }
        // Пул дескрипторов: 3 текстуры на материал * max_models
        {
            VkDescriptorPoolSize pools[] = {
                {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 8},
                {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 8},
                {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 8},
                {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 3 * max_models + 8},
            };
            VkDescriptorPoolCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 1 + max_models,
                .poolSizeCount = (uint32_t) (sizeof(pools) / sizeof(pools[0])),
                .pPoolSizes = pools,
            };
            if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan descriptor pool\n";
                veekay::app.running = false;
                return;
            }
        }
        // NOTE: Descriptor set layout specification
        // set = 0 (global): Scene UBO, Model UBO (dynamic), Point/Spot SSBO
        {
            VkDescriptorSetLayoutBinding bindings[] = {
                {
                    .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                },
            };
            VkDescriptorSetLayoutCreateInfo ginfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = (uint32_t) (sizeof(bindings) / sizeof(bindings[0])),
                .pBindings = bindings,
            };
            if (vkCreateDescriptorSetLayout(device, &ginfo, nullptr, &descriptor_set_layout_global) != VK_SUCCESS) {
                std::cerr << "Failed to create global descriptor set layout\n";
                veekay::app.running = false;
                return;
            }
        }

        // set = 1 (material): albedo/specular/emissive как combined image sampler
        {
            VkDescriptorSetLayoutBinding bindings[] = {
                {
                    .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                },
                {
                    .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
                },
            };
            VkDescriptorSetLayoutCreateInfo minfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = (uint32_t) (sizeof(bindings) / sizeof(bindings[0])),
                .pBindings = bindings,
            };
            if (vkCreateDescriptorSetLayout(device, &minfo, nullptr, &descriptor_set_layout_material) != VK_SUCCESS) {
                std::cerr << "Failed to create material descriptor set layout\n";
                veekay::app.running = false;
                return;
            }
        }

        // Pipeline layout: оба набора
        {
            VkDescriptorSetLayout set_layouts[] = {descriptor_set_layout_global, descriptor_set_layout_material};
            VkPipelineLayoutCreateInfo pl{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = (uint32_t) (sizeof(set_layouts) / sizeof(set_layouts[0])),
                .pSetLayouts = set_layouts,
            };
            if (vkCreatePipelineLayout(device, &pl, nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan Pipeline layout\n";
                veekay::app.running = false;
                return;
            }
        }

        // NOTE: Create graphics pipeline
        {
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

        //  Выравнивание UBO и аллокация буферов
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(physical_device, &props);
            g_ubo_align = uint32_t(props.limits.minUniformBufferOffsetAlignment);
            g_model_stride = uint32_t(align_up(sizeof(ModelUniforms), g_ubo_align));

            scene_uniforms_buffer = new veekay::graphics::Buffer(sizeof(SceneUniforms), nullptr,
                                                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            model_uniforms_buffer = new veekay::graphics::Buffer(max_models * g_model_stride, nullptr,
                                                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        }

        // Создание и заполнение буферов света
        {
            // Point lights
            point_lights = {
                {.position = {-2.0f, -1.5f, 0.0f}, ._pad0 = 0.0f, .color = {1.0f, 0.0f, 0.0f}, .intensity = 10.0f},
                {.position = {2.0f, -1.5f, 0.0f}, ._pad0 = 0.0f, .color = {0.0f, 1.0f, 0.0f}, .intensity = 10.0f},
                {.position = {0.0f, -2.0f, 2.0f}, ._pad0 = 0.0f, .color = {0.0f, 0.0f, 1.0f}, .intensity = 10.0f},
            };
            size_t header_size = sizeof(PointLightBufferHeader);
            size_t buffer_size = header_size + point_lights.size() * sizeof(PointLight);
            point_lights_buffer = new
                    veekay::graphics::Buffer(buffer_size, nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            uint8_t *mapped = static_cast<uint8_t *>(point_lights_buffer->mapped_region);
            PointLightBufferHeader header{.count = (uint32_t) point_lights.size(), ._pad1 = 0, ._pad2 = 0, ._pad3 = 0};
            std::memcpy(mapped, &header, header_size);
            std::memcpy(mapped + header_size, point_lights.data(), point_lights.size() * sizeof(PointLight));

            // Spot lights
            spot_lights = {
                {
                    .position = {0.0f, -2.0f, 0.0f}, ._pad0 = 0,
                    .direction = veekay::vec3::normalized({0.0f, 1.0f, 0.0f}),
                    .cutoff_angle = std::cos(toRadians(12.5f)), .color = {1.0f, 1.0f, 1.0f},
                    .outer_cutoff_angle = std::cos(toRadians(17.5f)), .intensity = 50.0f
                },
                {
                    .position = {-3.0f, 1.5f, -2.0f}, ._pad0 = 0,
                    .direction = veekay::vec3::normalized({1.0f, -0.5f, 0.5f}),
                    .cutoff_angle = std::cos(toRadians(15.0f)), .color = {1.0f, 0.5f, 0.0f},
                    .outer_cutoff_angle = std::cos(toRadians(20.0f)), .intensity = 40.0f
                },
                {
                    .position = {3.0f, 1.5f, 1.0f}, ._pad0 = 0,
                    .direction = veekay::vec3::normalized({-1.0f, -0.8f, -0.3f}),
                    .cutoff_angle = std::cos(toRadians(10.0f)), .color = {0.3f, 0.5f, 1.0f},
                    .outer_cutoff_angle = std::cos(toRadians(15.0f)), .intensity = 35.0f
                },
            };
            size_t spot_header_size = sizeof(SpotLightBufferHeader);
            size_t spot_buffer_size = spot_header_size + spot_lights.size() * sizeof(SpotLight);
            spot_lights_buffer = new veekay::graphics::Buffer(spot_buffer_size, nullptr,
                                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            uint8_t *spot_mapped = static_cast<uint8_t *>(spot_lights_buffer->mapped_region);
            SpotLightBufferHeader spot_header{
                .count = (uint32_t) spot_lights.size(), ._pad1 = 0, ._pad2 = 0, ._pad3 = 0
            };
            std::memcpy(spot_mapped, &spot_header, spot_header_size);
            std::memcpy(spot_mapped + spot_header_size, spot_lights.data(), spot_lights.size() * sizeof(SpotLight));
        }

        //  Заглушка-текстура и сэмплер по умолчанию
        {
            missing_texture_sampler = makeSampler(device,
                                                  VK_FILTER_NEAREST, VK_FILTER_NEAREST,
                                                  VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                                  1.0f
            );
            std::vector<uint32_t> pixels(16 * 16);
            for (int y = 0; y < 16; ++y) {
                for (int x = 0; x < 16; ++x) {
                    bool black = ((x / 8) + (y / 8)) % 2 == 0;
                    pixels[y * 16 + x] = black ? 0xff000000u : 0xffff00ffu; // чёрный : магента
                }
            }
            missing_texture = new veekay::graphics::Texture(cmd, 16, 16, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
        }

        // Аллокация наборов: 1 глобальный + per-model material
        {
            VkDescriptorSetLayout gl = descriptor_set_layout_global;
            VkDescriptorSetAllocateInfo ainfo{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &gl,
            };
            if (vkAllocateDescriptorSets(device, &ainfo, &descriptor_set_global) != VK_SUCCESS) {
                std::cerr << "Failed to allocate global descriptor set\n";
                veekay::app.running = false;
                return;
            }
        }

        // NOTE: Plane mesh initialization
        {
            // (v0)------(v1)
            //  |  \     |
            //  |   `--  |
            //  |      \ |
            // (v3)------(v2)
            std::vector<Vertex> vertices = {
                {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };
            std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(),
                                                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            plane_mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(),
                                                                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            plane_mesh.indices = (uint32_t) indices.size();
        }

        // NOTE: Cube mesh initialization
        {
            std::vector<Vertex> vertices = {
                // Back face
                {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
                {{0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
                {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
                {{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
                // Right face
                {{0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{0.5f, 0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                // Front face
                {{0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
                // Left face
                {{-0.5f, -0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{-0.5f, 0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                // Bottom face
                {{-0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
                // Top face
                {{-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                {{0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                {{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            };
            std::vector<uint32_t> indices = {
                0, 1, 2, 2, 3, 0,
                4, 5, 6, 6, 7, 4,
                8, 9, 10, 10, 11, 8,
                12, 13, 14, 14, 15, 12,
                16, 17, 18, 18, 19, 16,
                20, 21, 22, 22, 23, 20,
            };
            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(),
                                                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            cube_mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(),
                                                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            cube_mesh.indices = (uint32_t) indices.size();
        }

        // Модели сцены
        {
            models.clear();
            models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{},
                .material = Material{
                    .albedo_color = {1, 1, 1},
                    .specular_color = {0.5f, 0.5f, 0.5f},
                    .shininess = 32.0f
                }
            });
            models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{.position = {-2.0f, -0.5f, -1.5f}},
                .material = Material{
                    .albedo_color = {1, 1, 1},
                    .specular_color = {1.0f, 1.0f, 1.0f},
                    .shininess = 64.0f
                }
            });
            models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{.position = {1.5f, -0.5f, -0.5f}},
                .material = Material{
                    .albedo_color = {1, 1, 1},
                    .specular_color = {1.0f, 1.0f, 1.0f},
                    .shininess = 64.0f
                }
            });
            models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{.position = {0.0f, -0.5f, 1.0f}},
                .material = Material{
                    .albedo_color = {1, 1, 1},
                    .specular_color = {1.0f, 1.0f, 1.0f},
                    .shininess = 64.0f
                }
            });
        }


        //  Аллокация material-наборов
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

        //  Запись дескрипторов: глобальный набор (только буферы)
        {
            VkDescriptorBufferInfo buffers[] = {
                {.buffer = scene_uniforms_buffer->buffer, .offset = 0, .range = sizeof(SceneUniforms)},
                {.buffer = model_uniforms_buffer->buffer, .offset = 0, .range = sizeof(ModelUniforms)},
                {.buffer = point_lights_buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
                {.buffer = spot_lights_buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
            };
            VkWriteDescriptorSet gwrites[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set_global, .dstBinding = 0,
                    .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &buffers[0]
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set_global, .dstBinding = 1,
                    .dstArrayElement = 0, .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &buffers[1]
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set_global, .dstBinding = 2,
                    .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffers[2]
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set_global, .dstBinding = 3,
                    .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffers[3]
                },
            };
            vkUpdateDescriptorSets(device, (uint32_t) (sizeof(gwrites) / sizeof(gwrites[0])), gwrites, 0, nullptr);
        }

        // Запись per-model material наборов
        {
            for (size_t i = 0; i < models.size(); ++i) {
                auto &M = models[i].material;

                auto load_png = [&](const char *path) -> veekay::graphics::Texture * {
                    uint32_t w = 0, h = 0;
                    std::vector<unsigned char> pixels;
                    unsigned err = lodepng::decode(pixels, w, h, path);
                    if (err != 0 || w == 0 || h == 0) { throw std::runtime_error("png load fail"); }
                    return new veekay::graphics::Texture(cmd, w, h, VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
                };

                try { M.albedo = load_png("./assets/lenna.png"); } catch (...) { M.albedo = missing_texture; }
                try { M.specular = load_png("./assets/zz.png"); } catch (...) { M.specular = g_white1x1; }
                try { M.emissive = load_png("./assets/zzы.png"); } catch (...) { M.emissive = g_black1x1; }

                // Per-material sampler
                M.sampler = makeSampler(device, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                        16.0f);

                // Запись дескрипторов материала (set = 1: 0=albedo, 1=specular, 2=emissive)
                VkDescriptorImageInfo i_alb{
                    .sampler = (M.albedo == missing_texture) ? missing_texture_sampler : M.sampler,
                    .imageView = M.albedo->view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };
                VkDescriptorImageInfo i_spc{
                    .sampler = M.sampler, .imageView = M.specular->view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };
                VkDescriptorImageInfo i_emi{
                    .sampler = M.sampler, .imageView = M.emissive->view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                };

                VkWriteDescriptorSet writes[] = {
                    {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = M.set, .dstBinding = 0,
                        .dstArrayElement = 0, .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &i_alb
                    },
                    {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = M.set, .dstBinding = 1,
                        .dstArrayElement = 0, .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &i_spc
                    },
                    {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = M.set, .dstBinding = 2,
                        .dstArrayElement = 0, .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &i_emi
                    },
                };
                vkUpdateDescriptorSets(device, (uint32_t) (sizeof(writes) / sizeof(writes[0])), writes, 0, nullptr);
            }
        }
    }

    void shutdown() {
        VkDevice device = veekay::app.vk_device;


        for (auto &mdl: models) {
            if (mdl.material.sampler) {
                vkDestroySampler(device, mdl.material.sampler, nullptr);
                mdl.material.sampler = VK_NULL_HANDLE;
            }
            if (mdl.material.albedo && mdl.material.albedo != missing_texture) {
                delete mdl.material.albedo;
                mdl.material.albedo = nullptr;
            }
            if (mdl.material.specular && mdl.material.specular != g_white1x1) {
                delete mdl.material.specular;
                mdl.material.specular = nullptr;
            }
            if (mdl.material.emissive && mdl.material.emissive != g_black1x1) {
                delete mdl.material.emissive;
                mdl.material.emissive = nullptr;
            }
        }

        // Общие заглушки
        if (missing_texture_sampler) {
            vkDestroySampler(device, missing_texture_sampler, nullptr);
            missing_texture_sampler = VK_NULL_HANDLE;
        }
        if (missing_texture) {
            delete missing_texture;
            missing_texture = nullptr;
        }
        if (g_white1x1) {
            delete g_white1x1;
            g_white1x1 = nullptr;
        }
        if (g_black1x1) {
            delete g_black1x1;
            g_black1x1 = nullptr;
        }

        // Меши и буферы
        if (cube_mesh.index_buffer) {
            delete cube_mesh.index_buffer;
            cube_mesh.index_buffer = nullptr;
        }
        if (cube_mesh.vertex_buffer) {
            delete cube_mesh.vertex_buffer;
            cube_mesh.vertex_buffer = nullptr;
        }
        if (plane_mesh.index_buffer) {
            delete plane_mesh.index_buffer;
            plane_mesh.index_buffer = nullptr;
        }
        if (plane_mesh.vertex_buffer) {
            delete plane_mesh.vertex_buffer;
            plane_mesh.vertex_buffer = nullptr;
        }

        if (point_lights_buffer) {
            delete point_lights_buffer;
            point_lights_buffer = nullptr;
        }
        if (spot_lights_buffer) {
            delete spot_lights_buffer;
            spot_lights_buffer = nullptr;
        }
        if (model_uniforms_buffer) {
            delete model_uniforms_buffer;
            model_uniforms_buffer = nullptr;
        }
        if (scene_uniforms_buffer) {
            delete scene_uniforms_buffer;
            scene_uniforms_buffer = nullptr;
        }

        // Пайплайн и лейауты
        if (pipeline) {
            vkDestroyPipeline(device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        if (pipeline_layout) {
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            pipeline_layout = VK_NULL_HANDLE;
        }
        if (descriptor_set_layout_material) {
            vkDestroyDescriptorSetLayout(device, descriptor_set_layout_material, nullptr);
            descriptor_set_layout_material = VK_NULL_HANDLE;
        }
        if (descriptor_set_layout_global) {
            vkDestroyDescriptorSetLayout(device, descriptor_set_layout_global, nullptr);
            descriptor_set_layout_global = VK_NULL_HANDLE;
        }
        if (descriptor_pool) {
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            descriptor_pool = VK_NULL_HANDLE;
        }

        // Шейдеры
        if (fragment_shader_module) {
            vkDestroyShaderModule(device, fragment_shader_module, nullptr);
            fragment_shader_module = VK_NULL_HANDLE;
        }
        if (vertex_shader_module) {
            vkDestroyShaderModule(device, vertex_shader_module, nullptr);
            vertex_shader_module = VK_NULL_HANDLE;
        }
    }

    void update(double time) {
        ImGui::Begin("Scene Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        // СЕКЦИЯ КАМЕРЫ
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            static bool prev_look_at_view = look_at_view;
            ImGui::Checkbox("Look At Matrix?", &look_at_view);
            if (look_at_view != prev_look_at_view) {
                if (look_at_view) {
                    camera.saveNormalViewState();
                } else {
                    camera.saveLookAtViewState(target_look_at);
                }
                prev_look_at_view = look_at_view;
            }

            if (look_at_view) {
                ImGui::InputFloat3("Target", target_look_at.elements);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "State Management:");
            if (ImGui::Button("Save Current State", ImVec2(-1, 0))) {
                if (look_at_view) camera.saveLookAtViewState(target_look_at);
                else camera.saveNormalViewState();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Save current position and rotation for %s mode",
                                  look_at_view ? "Look-At" : "Normal View");
            }
            if (ImGui::Button("Restore Saved State", ImVec2(-1, 0))) {
                if (look_at_view) {
                    camera.restoreLookAtViewState();
                    target_look_at = camera.look_at_view_state.target;
                } else {
                    camera.restoreNormalViewState();
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Restore saved state for %s mode", look_at_view ? "Look-At" : "Normal View");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::Text("Position:");
            ImGui::Text(" X: %.2f Y: %.2f Z: %.2f", camera.position.x, camera.position.y, camera.position.z);
            if (!look_at_view) {
                ImGui::Text("Rotation (radians):");
                ImGui::Text(" X: %.3f Y: %.3f Z: %.3f", camera.rotation.x, camera.rotation.y, camera.rotation.z);
            }
            ImGui::Spacing();
        }

        // СЕКЦИЯ ОСВЕЩЕНИЯ
        if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient", &ambient_light.x);
            ImGui::Spacing();
            ImGui::Text("Directional (Sun):");
            ImGui::DragFloat3("Direction", &sun_direction.x, 0.01f, -1.0f, 1.0f);
            sun_direction = veekay::vec3::normalized(sun_direction);
            ImGui::ColorEdit3("Sun Color", &sun_color.x);
        }

        // СЕКЦИЯ ТОЧЕЧНЫХ ИСТОЧНИКОВ
        if (ImGui::CollapsingHeader("Point Lights")) {
            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID((int) i);
                if (ImGui::TreeNode("##light", "Point Light %zu", i)) {
                    ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
                    ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                    ImGui::DragFloat("Intensity", &point_lights[i].intensity, 0.1f, 0.0f, 100.0f);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        // СЕКЦИЯ ПРОЖЕКТОРНЫХ ИСТОЧНИКОВ
        if (ImGui::CollapsingHeader("Spot Lights")) {
            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID((int) i + 1000);
                if (ImGui::TreeNode("##spotlight", "Spot Light %zu", i)) {
                    ImGui::DragFloat3("Position", &spot_lights[i].position.x, 0.1f);
                    ImGui::DragFloat3("Direction", &spot_lights[i].direction.x, 0.01f, -1.0f, 1.0f);
                    spot_lights[i].direction = veekay::vec3::normalized(spot_lights[i].direction);

                    const float MIN_ANGLE_DIFF = 3.0f;
                    float inner_deg = std::acos(spot_lights[i].cutoff_angle) * 180.0f / float(M_PI);
                    float outer_deg = std::acos(spot_lights[i].outer_cutoff_angle) * 180.0f / float(M_PI);

                    if (ImGui::SliderFloat("Inner Angle", &inner_deg, 5.0f, 45.0f)) {
                        if (outer_deg - inner_deg < MIN_ANGLE_DIFF) {
                            outer_deg = inner_deg + MIN_ANGLE_DIFF;
                            if (outer_deg > 60.0f) {
                                outer_deg = 60.0f;
                                inner_deg = outer_deg - MIN_ANGLE_DIFF;
                            }
                        }
                        spot_lights[i].cutoff_angle = std::cos(toRadians(inner_deg));
                        spot_lights[i].outer_cutoff_angle = std::cos(toRadians(outer_deg));
                    }
                    if (ImGui::SliderFloat("Outer Angle", &outer_deg, 10.0f, 60.0f)) {
                        if (outer_deg - inner_deg < MIN_ANGLE_DIFF) {
                            inner_deg = outer_deg - MIN_ANGLE_DIFF;
                            if (inner_deg < 5.0f) {
                                inner_deg = 5.0f;
                                outer_deg = inner_deg + MIN_ANGLE_DIFF;
                            }
                        }
                        spot_lights[i].cutoff_angle = std::cos(toRadians(inner_deg));
                        spot_lights[i].outer_cutoff_angle = std::cos(toRadians(outer_deg));
                    }

                    ImGui::ColorEdit3("Color", &spot_lights[i].color.x);
                    ImGui::DragFloat("Intensity", &spot_lights[i].intensity, 0.1f, 0.0f, 200.0f);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        ImGui::End();

        // УПРАВЛЕНИЕ КАМЕРОЙ
        ImGuiIO &io = ImGui::GetIO();
        if (!io.WantCaptureMouse && !io.WantCaptureKeyboard) {
            using namespace veekay::input;
            if (mouse::isButtonDown(mouse::Button::left)) {
                auto move_delta = mouse::cursorDelta();
                camera.rotation.x += move_delta.y / 360.0f;
                camera.rotation.y += -move_delta.x / 360.0f;
                camera.rotation.x = std::clamp(camera.rotation.x, -float(M_PI) / 2, float(M_PI) / 2);
            }
            auto view = look_at_view ? camera.look_at(target_look_at) : camera.view();
            veekay::vec3 right = veekay::vec3::normalized({view[0][0], view[1][0], view[2][0]});
            veekay::vec3 up = veekay::vec3::normalized({view[0][1], view[1][1], view[2][1]});
            veekay::vec3 front = veekay::vec3::normalized({view[0][2], view[1][2], view[2][2]});
            if (keyboard::isKeyDown(keyboard::Key::w)) camera.position += front * 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::s)) camera.position -= front * 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::q)) camera.position -= up * 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::z)) camera.position += up * 0.1f;
        }

        veekay::mat4 matrix_view = look_at_view ? camera.look_at(target_look_at) : camera.view();
        const float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio, matrix_view),
            .view_position = camera.position,
            .ambient_light_intensity = ambient_light,
            .sun_light_direction = sun_direction,
            .sun_light_color = sun_color,
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            ModelUniforms &uniforms = model_uniforms[i];
            uniforms.model = model.transform.matrix();
            uniforms.albedo_color = model.material.albedo_color;
            uniforms.specular_color = model.material.specular_color;
            uniforms.shininess = model.material.shininess;
        }

        // Обновляем UBO сцены и модели (динамические смещения)
        *(SceneUniforms *) scene_uniforms_buffer->mapped_region = scene_uniforms;
        uint8_t *base = static_cast<uint8_t *>(model_uniforms_buffer->mapped_region);
        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
            std::memcpy(base + i * g_model_stride, &model_uniforms[i], sizeof(ModelUniforms));
        }

        // Обновляем буферы освещения
        {
            uint8_t *mapped = static_cast<uint8_t *>(point_lights_buffer->mapped_region);
            PointLightBufferHeader header{.count = (uint32_t) point_lights.size(), ._pad1 = 0, ._pad2 = 0, ._pad3 = 0};
            size_t header_size = sizeof(PointLightBufferHeader);
            std::memcpy(mapped, &header, header_size);
            std::memcpy(mapped + header_size, point_lights.data(), point_lights.size() * sizeof(PointLight));
        } {
            uint8_t *spot_mapped = static_cast<uint8_t *>(spot_lights_buffer->mapped_region);
            SpotLightBufferHeader spot_header{
                .count = (uint32_t) spot_lights.size(), ._pad1 = 0, ._pad2 = 0, ._pad3 = 0
            };
            size_t spot_header_size = sizeof(SpotLightBufferHeader);
            std::memcpy(spot_mapped, &spot_header, spot_header_size);
            std::memcpy(spot_mapped + spot_header_size, spot_lights.data(), spot_lights.size() * sizeof(SpotLight));
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        // NOTE: Start recording rendering commands
        VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &begin_info);

        // NOTE: Use current swapchain framebuffer and clear it
        VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
        VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
        VkClearValue clear_values[] = {clear_color, clear_depth};
        VkRenderPassBeginInfo rp_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = veekay::app.vk_render_pass,
            .framebuffer = framebuffer,
            .renderArea = {.offset = {0, 0}, .extent = {veekay::app.window_width, veekay::app.window_height}},
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };
        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;
        VkDeviceSize zero_offset = 0;

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model &model = models[i];
            const Mesh &mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }
            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, 0, VK_INDEX_TYPE_UINT32);
            }

            uint32_t dynamicOffset = (uint32_t) (i * g_model_stride);
            // set = 0 (global + dynamic UBO)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor_set_global,
                                    1, &dynamicOffset);
            // set = 1 (material)
            VkDescriptorSet mat_set = models[i].material.set;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    1, 1, &mat_set,
                                    0, nullptr);

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
