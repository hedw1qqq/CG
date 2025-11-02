#include "veekay/input.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define _USE_MATH_DEFINES

#include <math.h>
#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include "veekay/graphics.hpp"
#include "veekay/veekay.hpp"

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

    // Структура материала с Blinn-Phong параметрами
    struct Material {
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        Material material;
    };

    enum class CameraMode {
        LOOK_AT,
        MODEL_TRANSFORM
    };

    struct Camera {
        float yaw = -90.0f;
        float pitch = 0.0f;
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;
        veekay::vec3 position = {};
        veekay::vec3 rotation = {};
        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        float model_yaw = -90.0f;
        float model_pitch = 0.0f;
        float model_roll = 0.0f;

        CameraMode mode = CameraMode::LOOK_AT;

        struct SavedState {
            veekay::vec3 position;
            float yaw;
            float pitch;
            float roll;
        };
        SavedState look_at_saved_state;
        SavedState model_transform_saved_state;

        void saveCurrentState() {
            if (mode == CameraMode::LOOK_AT) {
                look_at_saved_state.position = position;
                look_at_saved_state.yaw = yaw;
                look_at_saved_state.pitch = pitch;
                look_at_saved_state.roll = 0.0f;
            } else {
                model_transform_saved_state.position = position;
                model_transform_saved_state.yaw = model_yaw;
                model_transform_saved_state.pitch = model_pitch;
                model_transform_saved_state.roll = model_roll;
            }
        }

        void restoreCurrentState() {
            if (mode == CameraMode::LOOK_AT) {
                position = look_at_saved_state.position;
                yaw = look_at_saved_state.yaw;
                pitch = look_at_saved_state.pitch;
            } else {
                position = model_transform_saved_state.position;
                model_yaw = model_transform_saved_state.yaw;
                model_pitch = model_transform_saved_state.pitch;
                model_roll = model_transform_saved_state.roll;
            }
        }

        void switchMode(CameraMode new_mode) {
            if (new_mode != mode) {

                if (mode == CameraMode::LOOK_AT && new_mode == CameraMode::MODEL_TRANSFORM) {
                    model_yaw = yaw;
                    model_pitch = pitch;

                } else if (mode == CameraMode::MODEL_TRANSFORM && new_mode == CameraMode::LOOK_AT) {
                    yaw = model_yaw;
                    pitch = model_pitch;
                }
                mode = new_mode;
            }
        }

        // NOTE: View matrix of camera (inverse of a transform)
        veekay::mat4 view() const;

        veekay::mat4 viewLookAt() const;

        veekay::mat4 viewModelTransform() const;

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

        //  Параметры освещения
        veekay::vec3 ambient_light = {0.2f, 0.2f, 0.2f};
        veekay::vec3 sun_direction = veekay::vec3::normalized({0.3f, 1.0f, 0.5f});
        veekay::vec3 sun_color = {1.0f, 1.0f, 0.9f};

        // Точечные источники света
        std::vector<PointLight> point_lights;
        veekay::graphics::Buffer *point_lights_buffer = nullptr;

        // Прожекторные источники света
        std::vector<SpotLight> spot_lights;
        veekay::graphics::Buffer *spot_lights_buffer = nullptr;

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

    veekay::mat4 mat4_inverse_rigid(const veekay::mat4 &m) {
        veekay::mat4 result = veekay::mat4::identity();

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result[j][i] = m[i][j];
            }
        }
        //  -R^T * t
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

    veekay::mat4 Camera::viewLookAt() const {

        veekay::vec3 worldUp = {0.0f, 1.0f, 0.0f};
        float cy = cosf(toRadians(yaw));
        float sy = sinf(toRadians(yaw));
        float cp = cosf(toRadians(pitch));
        float sp = sinf(toRadians(pitch));
        veekay::vec3 front = veekay::vec3::normalized(veekay::vec3{cy * cp, sp, sy * cp});
        veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, worldUp));
        veekay::vec3 up = veekay::vec3::normalized(veekay::vec3::cross(right, front));

        // Матрица вида (lookAt): R^T * T^{-1}
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

    // MODEL TRANSFORM режим
    veekay::mat4 Camera::viewModelTransform() const {
        auto t = veekay::mat4::translation(position);
        auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(model_yaw));
        auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(model_pitch));
        auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(model_roll));
        veekay::mat4 model = t * ry * rx * rz;
        return mat4_inverse_rigid(model);
    }

    veekay::mat4 Camera::view() const {
        if (mode == CameraMode::LOOK_AT) {
            return viewLookAt();
        } else {
            return viewModelTransform();
        }
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
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
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
                            .binding = 0,  // NOTE: First vertex buffer
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
                // Добавляем storage buffer в пул дескрипторов
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
                if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

            // NOTE: Descriptor set layout specification
            {
                //  binding для storage buffer с точечными и прожекторными источниками света
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
                };
                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                        .pBindings = bindings,
                };
                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
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
            if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
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
            if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical_device, &props);
        g_ubo_align = uint32_t(props.limits.minUniformBufferOffsetAlignment);
        g_model_stride = uint32_t(align_up(sizeof(ModelUniforms), g_ubo_align));

        scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // выделить буфер под модели с учётом выровненного шага
        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * g_model_stride,
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        // Создание storage buffer для точечных источников света
        {
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

            size_t header_size = sizeof(PointLightBufferHeader);
            size_t buffer_size = header_size + point_lights.size() * sizeof(PointLight);
            point_lights_buffer = new veekay::graphics::Buffer(
                    buffer_size,
                    nullptr,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            );

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

        // Создание storage buffer для прожекторных источников света
        {
            spot_lights.push_back(SpotLight{
                    .position = {0.0f, 2.0f, 0.0f},
                    .direction = veekay::vec3::normalized({0.0f, -1.0f, 0.0f}),
                    .cutoff_angle = std::cos(toRadians(12.5f)),
                    .color = {1.0f, 1.0f, 1.0f},
                    .outer_cutoff_angle = std::cos(toRadians(17.5f)),
                    .intensity = 50.0f
            });
            spot_lights.push_back(SpotLight{
                    .position = {-3.0f, 1.5f, -2.0f},
                    .direction = veekay::vec3::normalized({1.0f, -0.5f, 0.5f}),
                    .cutoff_angle = std::cos(toRadians(15.0f)),
                    .color = {1.0f, 0.5f, 0.0f},
                    .outer_cutoff_angle = std::cos(toRadians(20.0f)),
                    .intensity = 40.0f
            });

            spot_lights.push_back(SpotLight{
                    .position = {3.0f, 1.5f, 1.0f},
                    .direction = veekay::vec3::normalized({-1.0f, -0.8f, -0.3f}),
                    .cutoff_angle = std::cos(toRadians(10.0f)),
                    .color = {0.3f, 0.5f, 1.0f},
                    .outer_cutoff_angle = std::cos(toRadians(15.0f)),
                    .intensity = 35.0f
            });

            size_t spot_header_size = sizeof(SpotLightBufferHeader);
            size_t spot_buffer_size = spot_header_size + spot_lights.size() * sizeof(SpotLight);
            spot_lights_buffer = new veekay::graphics::Buffer(
                    spot_buffer_size,
                    nullptr,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            );

            uint8_t *spot_mapped = static_cast<uint8_t *>(spot_lights_buffer->mapped_region);
            SpotLightBufferHeader spot_header{
                    .count = static_cast<uint32_t>(spot_lights.size()),
                    ._pad1 = 0,
                    ._pad2 = 0,
                    ._pad3 = 0
            };
            std::memcpy(spot_mapped, &spot_header, spot_header_size);
            std::memcpy(spot_mapped + spot_header_size, spot_lights.data(),
                        spot_lights.size() * sizeof(SpotLight));
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
                    {
                            .buffer = spot_lights_buffer->buffer,
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

                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
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

        {
            // NOTE: Plane mesh initialization
            // (v0)------(v1)
            //  |      \     |
            //  |        `--,|
            //  |           \|
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

        {
            // NOTE: Cube mesh initialization
            std::vector<Vertex> vertices = {
                    // Back face
                    {{-0.5f, -0.5f, -0.5f}, {0.0f,  0.0f,  -1.0f}, {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, -0.5f}, {0.0f,  0.0f,  -1.0f}, {1.0f, 0.0f}},
                    {{+0.5f, +0.5f, -0.5f}, {0.0f,  0.0f,  -1.0f}, {1.0f, 1.0f}},
                    {{-0.5f, +0.5f, -0.5f}, {0.0f,  0.0f,  -1.0f}, {0.0f, 1.0f}},
                    // Right face
                    {{+0.5f, -0.5f, -0.5f}, {1.0f,  0.0f,  0.0f},  {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, +0.5f}, {1.0f,  0.0f,  0.0f},  {1.0f, 0.0f}},
                    {{+0.5f, +0.5f, +0.5f}, {1.0f,  0.0f,  0.0f},  {1.0f, 1.0f}},
                    {{+0.5f, +0.5f, -0.5f}, {1.0f,  0.0f,  0.0f},  {0.0f, 1.0f}},
                    // Front face
                    {{+0.5f, -0.5f, +0.5f}, {0.0f,  0.0f,  1.0f},  {0.0f, 0.0f}},
                    {{-0.5f, -0.5f, +0.5f}, {0.0f,  0.0f,  1.0f},  {1.0f, 0.0f}},
                    {{-0.5f, +0.5f, +0.5f}, {0.0f,  0.0f,  1.0f},  {1.0f, 1.0f}},
                    {{+0.5f, +0.5f, +0.5f}, {0.0f,  0.0f,  1.0f},  {0.0f, 1.0f}},
                    // Left face
                    {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f,  0.0f},  {0.0f, 0.0f}},
                    {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f,  0.0f},  {1.0f, 0.0f}},
                    {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f,  0.0f},  {1.0f, 1.0f}},
                    {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f,  0.0f},  {0.0f, 1.0f}},
                    // Bottom face
                    {{-0.5f, -0.5f, +0.5f}, {0.0f,  -1.0f, 0.0f},  {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, +0.5f}, {0.0f,  -1.0f, 0.0f},  {1.0f, 0.0f}},
                    {{+0.5f, -0.5f, -0.5f}, {0.0f,  -1.0f, 0.0f},  {1.0f, 1.0f}},
                    {{-0.5f, -0.5f, -0.5f}, {0.0f,  -1.0f, 0.0f},  {0.0f, 1.0f}},
                    // Top face
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
        delete point_lights_buffer;
        delete spot_lights_buffer;
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
        ImGui::Begin("Scene Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        // СЕКЦИЯ КАМЕРЫ
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Current Mode:");
            const char *modes[] = {"Look-At", "Model Transform"};
            int current_mode = static_cast<int>(camera.mode);
            if (ImGui::Combo("##mode", &current_mode, modes, 2)) {
                camera.switchMode(static_cast<CameraMode>(current_mode));
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "State Management:");
            if (ImGui::Button("Save Current State", ImVec2(-1, 0))) {
                camera.saveCurrentState();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Save position and angles for %s mode",
                                  camera.mode == CameraMode::LOOK_AT ? "Look-At" : "Model Transform");
            }
            if (ImGui::Button("Restore Saved State", ImVec2(-1, 0))) {
                camera.restoreCurrentState();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Restore saved state for %s mode",
                                  camera.mode == CameraMode::LOOK_AT ? "Look-At" : "Model Transform");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("Position:");
            ImGui::Text("  X: %.2f  Y: %.2f  Z: %.2f",
                        camera.position.x, camera.position.y, camera.position.z);

            ImGui::Spacing();

            if (camera.mode == CameraMode::LOOK_AT) {
                ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "✓ Look-At Mode");
                ImGui::Text("  Yaw:   %.1f°", camera.yaw);
                ImGui::Text("  Pitch: %.1f°", camera.pitch);
            } else {
                ImGui::TextColored(ImVec4(0.3f, 0.6f, 1.0f, 1.0f), "✓ Model Transform Mode");
                ImGui::Text("  Yaw:   %.1f°", camera.model_yaw);
                ImGui::Text("  Pitch: %.1f°", camera.model_pitch);
                ImGui::SliderFloat("Roll", &camera.model_roll, -180.0f, 180.0f, "%.1f°");
            }
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
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::TreeNode("##light", "Point Light %zu", i)) {
                    ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
                    ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                    ImGui::DragFloat("Intensity", &point_lights[i].intensity, 0.1f, 0.0f, 100.0f);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }

        //  СЕКЦИЯ ПРОЖЕКТОРНЫХ ИСТОЧНИКОВ
        if (ImGui::CollapsingHeader("Spot Lights")) {
            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i) + 1000); // смещение ID
                if (ImGui::TreeNode("##spotlight", "Spot Light %zu", i)) {
                    ImGui::DragFloat3("Position", &spot_lights[i].position.x, 0.1f);
                    ImGui::DragFloat3("Direction", &spot_lights[i].direction.x, 0.01f, -1.0f, 1.0f);
                    spot_lights[i].direction = veekay::vec3::normalized(spot_lights[i].direction);

                    // Углы в градусах для UI
                    float inner_deg = std::acos(spot_lights[i].cutoff_angle) * 180.0f / M_PI;
                    float outer_deg = std::acos(spot_lights[i].outer_cutoff_angle) * 180.0f / M_PI;
                    if (ImGui::SliderFloat("Inner Angle", &inner_deg, 5.0f, 45.0f)) {
                        spot_lights[i].cutoff_angle = std::cos(toRadians(inner_deg));
                    }
                    if (ImGui::SliderFloat("Outer Angle", &outer_deg, 10.0f, 60.0f)) {
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

        // Обновление буферов точечных источников каждый кадр
        {
            uint8_t *mapped = static_cast<uint8_t *>(point_lights_buffer->mapped_region);
            PointLightBufferHeader header{
                    .count = static_cast<uint32_t>(point_lights.size()),
                    ._pad1 = 0, ._pad2 = 0, ._pad3 = 0
            };
            size_t header_size = sizeof(PointLightBufferHeader);
            std::memcpy(mapped, &header, header_size);
            std::memcpy(mapped + header_size, point_lights.data(),
                        point_lights.size() * sizeof(PointLight));
        }

        {
            uint8_t *spot_mapped = static_cast<uint8_t *>(spot_lights_buffer->mapped_region);
            SpotLightBufferHeader spot_header{
                    .count = static_cast<uint32_t>(spot_lights.size()),
                    ._pad1 = 0, ._pad2 = 0, ._pad3 = 0
            };
            size_t spot_header_size = sizeof(SpotLightBufferHeader);
            std::memcpy(spot_mapped, &spot_header, spot_header_size);
            std::memcpy(spot_mapped + spot_header_size, spot_lights.data(),
                        spot_lights.size() * sizeof(SpotLight));
        }

        // УПРАВЛЕНИЕ КАМЕРОЙ
        ImGuiIO &io = ImGui::GetIO();
        if (!io.WantCaptureMouse && !io.WantCaptureKeyboard) {

            using namespace veekay::input;
            if (mouse::isButtonDown(mouse::Button::left)) {
                auto move_delta = mouse::cursorDelta();
                const float sensitivity = 0.1f;
                if (camera.mode == CameraMode::LOOK_AT) {
                    camera.yaw += move_delta.x * sensitivity;
                    camera.pitch -= move_delta.y * sensitivity;
                    camera.pitch = clampf(camera.pitch, -89.0f, 89.0f);
                } else {
                    camera.model_yaw += move_delta.x * sensitivity;
                    camera.model_pitch -= move_delta.y * sensitivity;
                    camera.model_pitch = clampf(camera.model_pitch, -89.0f, 89.0f);
                }
            }

            // Вычисляем направления камеры
            veekay::vec3 front, right, up;
            veekay::vec3 worldUp = {0.0f, 1.0f, 0.0f};

            if (camera.mode == CameraMode::LOOK_AT) {
                float cy = cosf(toRadians(camera.yaw));
                float sy = sinf(toRadians(camera.yaw));
                float cp = cosf(toRadians(camera.pitch));
                float sp = sinf(toRadians(camera.pitch));
                front = veekay::vec3::normalized({cy * cp, sp, sy * cp});
                right = veekay::vec3::normalized(veekay::vec3::cross(front, worldUp));
                up = veekay::vec3::normalized(veekay::vec3::cross(right, front));
            } else {
                auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(camera.model_yaw));
                auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(camera.model_pitch));
                auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(camera.model_roll));
                auto rotation = ry * rx * rz;

                right = veekay::vec3::normalized(veekay::vec3{
                        rotation[0].x,
                        rotation[0].y,
                        rotation[0].z
                });

                up = veekay::vec3::normalized(veekay::vec3{
                        rotation[1].x,
                        rotation[1].y,
                        rotation[1].z
                });

                front = veekay::vec3::normalized(veekay::vec3{
                        -rotation[2].x,
                        -rotation[2].y,
                        -rotation[2].z
                });
            }

            const float speed = 0.1f;
            if (camera.mode == CameraMode::LOOK_AT) {

                if (keyboard::isKeyDown(keyboard::Key::w))
                    camera.position -= front * speed;
                if (keyboard::isKeyDown(keyboard::Key::s))
                    camera.position += front * speed;
                if (keyboard::isKeyDown(keyboard::Key::a))
                    camera.position -= right * speed;
                if (keyboard::isKeyDown(keyboard::Key::d))
                    camera.position += right * speed;
                if (keyboard::isKeyDown(keyboard::Key::q))
                    camera.position -= up * speed;
                if (keyboard::isKeyDown(keyboard::Key::z))
                    camera.position += up * speed;
            } else {
                if (keyboard::isKeyDown(keyboard::Key::w))
                    camera.position -= front * speed;
                if (keyboard::isKeyDown(keyboard::Key::s))
                    camera.position += front * speed;
                if (keyboard::isKeyDown(keyboard::Key::a))
                    camera.position -= right * speed;
                if (keyboard::isKeyDown(keyboard::Key::d))
                    camera.position += right * speed;
                if (keyboard::isKeyDown(keyboard::Key::q))
                    camera.position -= up * speed;
                if (keyboard::isKeyDown(keyboard::Key::z))
                    camera.position += up * speed;
            }
        }

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

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
