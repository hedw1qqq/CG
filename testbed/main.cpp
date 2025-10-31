#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES

#include <math.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

#include <lodepng.h>
#include <cstring>

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
        veekay::graphics::Buffer *vertex_buffer;
        veekay::graphics::Buffer *index_buffer;
        uint32_t indices;
    };

    struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        // NOTE: Model matrix (translation, rotation and scaling)
        veekay::mat4 matrix() const;
    };

    // ADDED: Структура точечного источника света
    struct PointLight {
        veekay::vec3 position;
        float _pad0;
        veekay::vec3 color;
        float intensity;
    };

    // ADDED: Структура материала с Blinn-Phong параметрами
    struct Material {
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        Material material; // CHANGED: используем Material вместо просто albedo_color
    };

    struct Camera {
        float yaw = -90.0f; // смотрим вдоль -Z по умолчанию
        float pitch = 0.0f;   // горизонтально
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;

        veekay::vec3 position = {};
        veekay::vec3 rotation = {};

        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        // NOTE: View matrix of camera (inverse of a transform)
        veekay::mat4 view() const;

        // NOTE: View and projection composition
        veekay::mat4 view_projection(float aspect_ratio) const;
    };

// NOTE: Scene objects
    inline namespace {
        Camera camera{
                .position = {0.0f, -0.5f, -3.0f}
        };

        std::vector<Model> models;
    }

// NOTE: Vulkan objects
    inline namespace {
        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;
        VkDescriptorSet descriptor_set;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        // ADDED: Параметры освещения
        veekay::vec3 ambient_light = {0.2f, 0.2f, 0.2f};
        veekay::vec3 sun_direction = veekay::vec3::normalized({0.3f, -1.0f, 0.5f});
        veekay::vec3 sun_color = {1.0f, 1.0f, 0.9f};

        // ADDED: Точечные источники света
        std::vector<PointLight> point_lights;
        veekay::graphics::Buffer *point_lights_buffer = nullptr;

        veekay::graphics::Buffer *scene_uniforms_buffer;
        veekay::graphics::Buffer *model_uniforms_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;

        veekay::graphics::Texture *missing_texture;
        VkSampler missing_texture_sampler;

        veekay::graphics::Texture *texture;
        VkSampler texture_sampler;
    }

    float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
    }

    static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

    veekay::mat4 Transform::matrix() const {
        // TODO: Scaling and rotation

        auto t = veekay::mat4::translation(position);

        return t;
    }

    veekay::mat4 Camera::view() const {
        // TODO: Rotation
        veekay::vec3 worldUp = {0.0f, 1.0f, 0.0f};
        float cy = cosf(toRadians(yaw));
        float sy = sinf(toRadians(yaw));
        float cp = cosf(toRadians(pitch));
        float sp = sinf(toRadians(pitch));

        veekay::vec3 front = veekay::vec3::normalized(veekay::vec3{cy * cp, sp, sy * cp});
        veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, worldUp));
        veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(right, front));

        // ADDED: Матрица вида (lookAt): R^T * T^{-1}
        veekay::mat4 view = veekay::mat4::identity();

        // c0 = right
        view[0][0] = right.x;
        view[0][1] = right.y;
        view[0][2] = right.z;
        view[0][3] = 0.0f;

        // c1 = up
        view[1][0] = up.x;
        view[1][1] = up.y;
        view[1][2] = up.z;
        view[1][3] = 0.0f;

        // c2 = -front
        view[2][0] = -front.x;
        view[2][1] = -front.y;
        view[2][2] = -front.z;
        view[2][3] = 0.0f;

        // c3 = translation = {-dot(right,p), -dot(up,p), dot(front,p), 1}
        view[3][0] = -veekay::vec3::dot(right, position);
        view[3][1] = -veekay::vec3::dot(up, position);
        view[3][2] = veekay::vec3::dot(front, position);
        view[3][3] = 1.0f;
        return view;
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

        return view() * projection;
    }

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
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
        if (vkCreateShaderModule(veekay::app.vk_device, &
                info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }

        return result;
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;
        { // NOTE: Build graphics pipeline
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
                            .offset = static_cast<uint32_t>(offsetof(Vertex,
                                                                     position)), // NOTE: Offset of "position" field in a Vertex struct
                    },
                    {
                            .location = 1,
                            .binding = 0,
                            .format = VK_FORMAT_R32G32B32_SFLOAT,
                            .offset = static_cast<uint32_t>(offsetof(Vertex, normal)),
                    },
                    {
                            .location = 2,
                            .binding = 0,
                            .format = VK_FORMAT_R32G32_SFLOAT,
                            .offset = static_cast<uint32_t>(offsetof(Vertex, uv)),
                    },
            };

            // NOTE: Describe inputs
            VkPipelineVertexInputStateCreateInfo input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
                    .pVertexAttributeDescriptions = attributes,
            };

            // NOTE: Every three vertices make up a triangle,
            //       so our vertex buffer contains a "list of triangles"
            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            // NOTE: Declare clockwise triangle order as front-facing
            //       Discard triangles that are facing away
            //       Fill triangles, don't draw lines instaed
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
                    .sampleShadingEnable = false,
                    .minSampleShading = 1.0f,
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

            // NOTE: Let rasterizer draw on the entire window
            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

                    .viewportCount = 1,
                    .pViewports = &viewport,

                    .scissorCount = 1,
                    .pScissors = &scissor,
            };

            // NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
            VkPipelineDepthStencilStateCreateInfo depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            // NOTE: Let fragment shader write all the color channels
            VkPipelineColorBlendAttachmentState attachment_info{
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT,
            };

            // NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
            VkPipelineColorBlendStateCreateInfo blend_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

                    .logicOpEnable = false,
                    .logicOp = VK_LOGIC_OP_COPY,

                    .attachmentCount = 1,
                    .pAttachments = &attachment_info
            };

            {
                // ADDED: Добавляем storage buffer в пул дескрипторов
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
                                .descriptorCount = 8,
                        },
                        {
                                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 8,
                        }
                };

                VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = 1,
                        .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                        .pPoolSizes = pools,
                };

                if (vkCreateDescriptorPool(device, &info, nullptr,
                                           &descriptor_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // NOTE: Descriptor set layout specification
            {
                // ADDED: Добавляем binding для storage buffer с точечными источниками света
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
                };

                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                        .pBindings = bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                                &descriptor_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            {
                VkDescriptorSetAllocateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = descriptor_pool,
                        .descriptorSetCount = 1,
                        .pSetLayouts = &descriptor_set_layout,
                };

                if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // NOTE: Declare external data sources, only push constants this time
            VkPipelineLayoutCreateInfo layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 1,
                    .pSetLayouts = &descriptor_set_layout,
            };

            // NOTE: Create pipeline layout
            if (vkCreatePipelineLayout(device, &layout_info,
                                       nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline layout\n";
                veekay::app.running = false;
                return;
            }

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

            // NOTE: Create graphics pipeline
            if (vkCreateGraphicsPipelines(device, nullptr,
                                          1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device, &props);
        g_ubo_align = uint32_t(props.limits.minUniformBufferOffsetAlignment);
        g_model_stride = uint32_t(align_up(sizeof(ModelUniforms), g_ubo_align)); // например 80 -> 128 при align=64

        scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // ADDED: выделить буфер под модели с учётом выровненного шага
        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * g_model_stride,
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // ADDED: Создание storage buffer для точечных источников света
        {
            // Добавляем несколько точечных источников света
            point_lights.push_back(PointLight{
                    .position = {-2.0f, -1.5f, 0.0f},
                    .color = {1.0f, 0.0f, 0.0f},
                    .intensity = 10.0f
            });

            point_lights.push_back(PointLight{
                    .position = {2.0f, -1.5f, 0.0f},
                    .color = {0.0f, 1.0f, 0.0f},
                    .intensity = 10.0f
            });

            point_lights.push_back(PointLight{
                    .position = {0.0f, -2.0f, 2.0f},
                    .color = {0.0f, 0.0f, 1.0f},
                    .intensity = 10.0f
            });

            // FIXED: Размер буфера с учётом выравненного заголовка (16 байт)
            size_t header_size = sizeof(PointLightBufferHeader);
            size_t buffer_size = header_size + point_lights.size() * sizeof(PointLight);
            point_lights_buffer = new veekay::graphics::Buffer(
                    buffer_size,
                    nullptr,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            );

            // FIXED: Записываем выравненный заголовок + данные
            uint8_t *mapped = static_cast<uint8_t *>(point_lights_buffer->mapped_region);
            PointLightBufferHeader header{
                    .count = static_cast<uint32_t>(point_lights.size()),
                    ._pad1 = 0,
                    ._pad2 = 0,
                    ._pad3 = 0
            };
            std::memcpy(mapped, &header, header_size);
            std::memcpy(mapped + header_size, point_lights.data(),
                        point_lights.size() * sizeof(PointLight));
        }

        // NOTE: This texture and sampler is used when texture could not be loaded
        {
            VkSamplerCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            };

            if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan texture sampler\n";
                veekay::app.running = false;
                return;
            }

            uint32_t pixels[] = {
                    0xff000000, 0xffff00ff,
                    0xffff00ff, 0xff000000,
            };

            missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
                                                            VK_FORMAT_B8G8R8A8_UNORM,
                                                            pixels);
        }

        {
            // ADDED: Обновляем descriptor sets с storage buffer
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
                            .offset = 0,
                            .range = VK_WHOLE_SIZE,
                    },
            };

            VkWriteDescriptorSet write_infos[] = {
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 0,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            .pBufferInfo = &buffer_infos[0],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 1,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                            .pBufferInfo = &buffer_infos[1],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 2,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .pBufferInfo = &buffer_infos[2],
                    },
            };

            vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                                   write_infos, 0, nullptr);
        }

        // NOTE: Plane mesh initialization
        {
            // (v0)------(v1)
            //  |  \       |
            //  |   `--,   |
            //  |       \  |
            // (v3)------(v2)
            std::vector<Vertex> vertices = {
                    {{-5.0f, 0.0f, 5.0f},  {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                    {{5.0f,  0.0f, 5.0f},  {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                    {{5.0f,  0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
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
                    {{-0.5f, -0.5f, -0.5f}, {0.0f,  0.0f,  -1.0f}, {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, -0.5f}, {0.0f,  0.0f,  -1.0f}, {1.0f, 0.0f}},
                    {{+0.5f, +0.5f, -0.5f}, {0.0f,  0.0f,  -1.0f}, {1.0f, 1.0f}},
                    {{-0.5f, +0.5f, -0.5f}, {0.0f,  0.0f,  -1.0f}, {0.0f, 1.0f}},

                    {{+0.5f, -0.5f, -0.5f}, {1.0f,  0.0f,  0.0f},  {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, +0.5f}, {1.0f,  0.0f,  0.0f},  {1.0f, 0.0f}},
                    {{+0.5f, +0.5f, +0.5f}, {1.0f,  0.0f,  0.0f},  {1.0f, 1.0f}},
                    {{+0.5f, +0.5f, -0.5f}, {1.0f,  0.0f,  0.0f},  {0.0f, 1.0f}},

                    {{+0.5f, -0.5f, +0.5f}, {0.0f,  0.0f,  1.0f},  {0.0f, 0.0f}},
                    {{-0.5f, -0.5f, +0.5f}, {0.0f,  0.0f,  1.0f},  {1.0f, 0.0f}},
                    {{-0.5f, +0.5f, +0.5f}, {0.0f,  0.0f,  1.0f},  {1.0f, 1.0f}},
                    {{+0.5f, +0.5f, +0.5f}, {0.0f,  0.0f,  1.0f},  {0.0f, 1.0f}},

                    {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f,  0.0f},  {0.0f, 0.0f}},
                    {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f,  0.0f},  {1.0f, 0.0f}},
                    {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f,  0.0f},  {1.0f, 1.0f}},
                    {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f,  0.0f},  {0.0f, 1.0f}},

                    {{-0.5f, -0.5f, +0.5f}, {0.0f,  -1.0f, 0.0f},  {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, +0.5f}, {0.0f,  -1.0f, 0.0f},  {1.0f, 0.0f}},
                    {{+0.5f, -0.5f, -0.5f}, {0.0f,  -1.0f, 0.0f},  {1.0f, 1.0f}},
                    {{-0.5f, -0.5f, -0.5f}, {0.0f,  -1.0f, 0.0f},  {0.0f, 1.0f}},

                    {{-0.5f, +0.5f, -0.5f}, {0.0f,  1.0f,  0.0f},  {0.0f, 0.0f}},
                    {{+0.5f, +0.5f, -0.5f}, {0.0f,  1.0f,  0.0f},  {1.0f, 0.0f}},
                    {{+0.5f, +0.5f, +0.5f}, {0.0f,  1.0f,  0.0f},  {1.0f, 1.0f}},
                    {{-0.5f, +0.5f, +0.5f}, {0.0f,  1.0f,  0.0f},  {0.0f, 1.0f}},
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
        }

        // NOTE: Add models to scene
        // CHANGED: Добавляем материалы с параметрами Blinn-Phong
        models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{},
                .material = Material{
                        .albedo_color = {0.8f, 0.8f, 0.8f},
                        .specular_color = {0.5f, 0.5f, 0.5f},
                        .shininess = 32.0f
                }
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{
                        .position = {-2.0f, -0.5f, -1.5f},
                },
                .material = Material{
                        .albedo_color = {1.0f, 0.0f, 0.0f},
                        .specular_color = {1.0f, 1.0f, 1.0f},
                        .shininess = 64.0f
                }
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{
                        .position = {1.5f, -0.5f, -0.5f},
                },
                .material = Material{
                        .albedo_color = {0.0f, 1.0f, 0.0f},
                        .specular_color = {1.0f, 1.0f, 1.0f},
                        .shininess = 64.0f
                }
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{
                        .position = {0.0f, -0.5f, 1.0f},
                },
                .material = Material{
                        .albedo_color = {0.0f, 0.0f, 1.0f},
                        .specular_color = {1.0f, 1.0f, 1.0f},
                        .shininess = 64.0f
                }
        });
    }

// NOTE: Destroy resources here, do not cause leaks in your program!
    void shutdown() {
        VkDevice &device = veekay::app.vk_device;

        vkDestroySampler(device, missing_texture_sampler, nullptr);
        delete missing_texture;

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        // ADDED: Удаляем буфер точечных источников света
        delete point_lights_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        // ADDED: ImGui controls для управления освещением
        ImGui::Begin("Lighting Controls");

        if (ImGui::CollapsingHeader("Ambient Light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient Color", &ambient_light.x);
        }

        if (ImGui::CollapsingHeader("Directional Light (Sun)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Direction", &sun_direction.x, 0.01f, -1.0f, 1.0f);
            sun_direction = veekay::vec3::normalized(sun_direction);
            ImGui::ColorEdit3("Sun Color", &sun_color.x);
        }

        // FIXED: Флаг для отслеживания изменений буфера
        bool buffer_resized = false;

        if (ImGui::CollapsingHeader("Point Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::TreeNode("Light", "Point Light %zu", i)) {
                    ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
                    ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                    ImGui::DragFloat("Intensity", &point_lights[i].intensity, 0.1f, 0.0f, 100.0f);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }

            if (ImGui::Button("Add Point Light") && point_lights.size() < 16) {
                point_lights.push_back(PointLight{
                        .position = {0.0f, -1.0f, 0.0f},
                        .color = {1.0f, 1.0f, 1.0f},
                        .intensity = 10.0f
                });

                // CRITICAL: Ждём завершения всех операций GPU перед удалением буфера
                VkDevice &device = veekay::app.vk_device;
                vkDeviceWaitIdle(device);

                // FIXED: Пересоздаём буфер с учётом выравненного заголовка
                delete point_lights_buffer;
                size_t header_size = sizeof(PointLightBufferHeader);
                size_t buffer_size = header_size + point_lights.size() * sizeof(PointLight);
                point_lights_buffer = new veekay::graphics::Buffer(
                        buffer_size,
                        nullptr,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                );
                buffer_resized = true;
            }
        }

        ImGui::End();

        // FIXED: Обновляем дескриптор если буфер был пересоздан
        if (buffer_resized) {
            VkDevice &device = veekay::app.vk_device;

            VkDescriptorBufferInfo buffer_info{
                    .buffer = point_lights_buffer->buffer,
                    .offset = 0,
                    .range = VK_WHOLE_SIZE,
            };

            VkWriteDescriptorSet write_info{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = descriptor_set,
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &buffer_info,
            };

            vkUpdateDescriptorSets(device, 1, &write_info, 0, nullptr);
        }

        // ADDED: Обновляем storage buffer точечных источников
        uint8_t *mapped = static_cast<uint8_t *>(point_lights_buffer->mapped_region);
        PointLightBufferHeader header{
                .count = static_cast<uint32_t>(point_lights.size()),
                ._pad1 = 0,
                ._pad2 = 0,
                ._pad3 = 0
        };
        size_t header_size = sizeof(PointLightBufferHeader);
        std::memcpy(mapped, &header, header_size);
        std::memcpy(mapped + header_size, point_lights.data(),
                    point_lights.size() * sizeof(PointLight));

        // Управление камерой
        ImGuiIO &io = ImGui::GetIO();
        if (!io.WantCaptureMouse && !io.WantCaptureKeyboard) {
            using namespace veekay::input;

            if (mouse::isButtonDown(mouse::Button::left)) {
                auto move_delta = mouse::cursorDelta();
                const float sensitivity = 0.1f;
                camera.yaw += move_delta.x * sensitivity;
                camera.pitch -= move_delta.y * sensitivity;
                camera.pitch = clampf(camera.pitch, -89.0f, 89.0f);
            }

            veekay::vec3 worldUp = {0.0f, 1.0f, 0.0f};
            float cy = cosf(toRadians(camera.yaw));
            float sy = sinf(toRadians(camera.yaw));
            float cp = cosf(toRadians(camera.pitch));
            float sp = sinf(toRadians(camera.pitch));
            veekay::vec3 front = veekay::vec3::normalized({cy * cp, sp, sy * cp});
            veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, worldUp));
            veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(right, front));

            const float speed = 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::w))
                camera.position -= front * speed;

            if (keyboard::isKeyDown(keyboard::Key::s))
                camera.position += front * speed;

            if (keyboard::isKeyDown(keyboard::Key::d))
                camera.position += right * speed;

            if (keyboard::isKeyDown(keyboard::Key::a))
                camera.position -= right * speed;

            if (keyboard::isKeyDown(keyboard::Key::q))
                camera.position -= up * speed;

            if (keyboard::isKeyDown(keyboard::Key::z))
                camera.position += up * speed;
        }

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

        // CHANGED: Добавляем параметры освещения в SceneUniforms
        SceneUniforms scene_uniforms{
                .view_projection = camera.view_projection(aspect_ratio),
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

        *(SceneUniforms *) scene_uniforms_buffer->mapped_region = scene_uniforms;
        uint8_t *base = static_cast<uint8_t *>(model_uniforms_buffer->mapped_region);
        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
            std::memcpy(base + i * g_model_stride, &model_uniforms[i], sizeof(ModelUniforms));
        }
    }


    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        { // NOTE: Start recording rendering commands
            VkCommandBufferBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(cmd, &info);
        }

        { // NOTE: Use current swapchain framebuffer and clear it
            VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

            VkClearValue clear_values[] = {clear_color, clear_depth};

            VkRenderPassBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    .renderPass = veekay::app.vk_render_pass,
                    .framebuffer = framebuffer,
                    .renderArea = {
                            .extent = {
                                    veekay::app.window_width,
                                    veekay::app.window_height
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

            uint32_t offset = uint32_t(i * g_model_stride);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor_set, 1, &offset);

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
