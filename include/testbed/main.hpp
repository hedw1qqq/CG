#ifndef TESTBED_MAIN_H
#define TESTBED_MAIN_H


#include <veekay/veekay.hpp>
#include <veekay/graphics.hpp>
#include <imgui.h>
#include <lodepng.h>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>

// Константы
constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 16;
constexpr uint32_t max_spot_lights = 16;
constexpr uint32_t max_textures = 1024;
constexpr uint32_t shadow_map_size = 2048;

// Структура вершины
struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
};

// Структура преобразования
struct Transform {
    veekay::vec3 position = {};
    veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
    veekay::vec3 rotation = {};

    veekay::mat4 matrix() const;
};

// Структура сетки
struct Mesh {
    veekay::graphics::Buffer* vertex_buffer = nullptr;
    veekay::graphics::Buffer* index_buffer = nullptr;
    uint32_t indices = 0;
};

// Структура точечного источника света
struct PointLight {
    veekay::vec3 position;
    veekay::vec3 color;
    float radius;

    PointLight() = default;
    PointLight(veekay::vec3 pos, veekay::vec3 col, float rad) : position(pos), color(col), radius(rad) {}
};

// Структура прожектора
struct SpotLight {
    veekay::vec3 position;
    veekay::vec3 color;
    veekay::vec3 direction;
    float radius;
    float angle;        // внутренний угол (в радианах)
    float outer_angle;  // внешний угол (в радианах)

    SpotLight() = default;
    SpotLight(veekay::vec3 pos, veekay::vec3 col, veekay::vec3 dir, float rad, float ang, float out_ang)
            : position(pos), color(col), direction(dir), radius(rad), angle(ang), outer_angle(out_ang) {}
};

// Структура материала
struct Material {
    veekay::vec3 albedo_color = {1.0f, 1.0f, 1.0f};
    veekay::vec3 specular_color = {1.0f, 1.0f, 1.0f};
    float shininess = 100.0f;

    veekay::graphics::Texture* texture = nullptr;
    veekay::graphics::Texture* specular_texture = nullptr;
    veekay::graphics::Texture* emissive_texture = nullptr;
    VkSampler sampler = VK_NULL_HANDLE;

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

    // Создает дескрипторный набор для материала
    void createMaterialDescriptorSet(VkDescriptorPool pool, VkDescriptorSetLayout* layout) {
        VkDevice device = veekay::app.vk_device;

        VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = pool,
                .descriptorSetCount = 1,
                .pSetLayouts = layout,
        };

        if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to create material descriptor set\n";
            return;
        }

        VkDescriptorImageInfo albedo_image_info{
                .sampler = sampler,
                .imageView = texture->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkDescriptorImageInfo specular_image_info{
                .sampler = sampler,
                .imageView = specular_texture->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkDescriptorImageInfo emissive_image_info{
                .sampler = sampler,
                .imageView = emissive_texture->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkWriteDescriptorSet write_infos[3] = {
                {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = descriptor_set,
                        .dstBinding = 0,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = &albedo_image_info,
                },
                {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = descriptor_set,
                        .dstBinding = 1,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = &specular_image_info,
                },
                {
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .dstSet = descriptor_set,
                        .dstBinding = 2,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .pImageInfo = &emissive_image_info,
                }
        };

        vkUpdateDescriptorSets(device, 3, write_infos, 0, nullptr);
    }
};

// Структура модели
struct Model {
    Mesh mesh;
    Transform transform;
    std::shared_ptr<Material> material;
};

// Структура камеры
struct Camera {
    veekay::vec3 position = {0.0f, 0.0f, 0.0f};
    veekay::vec3 rotation = {0.0f, 0.0f, 0.0f}; // в радианах
    float fov = 60.0f;
    float near_plane = 0.1f;
    float far_plane = 100.0f;

    veekay::mat4 view() const;
    veekay::mat4 look_at(veekay::vec3 target) const;
    veekay::mat4 view_projection(float aspect_ratio, const veekay::mat4& view_matrix) const;
};

// Структура для теневой карты
struct ShadowMap {
    VkImage depth_image = VK_NULL_HANDLE;
    VkDeviceMemory depth_image_memory = VK_NULL_HANDLE;
    VkImageView depth_image_view = VK_NULL_HANDLE;
    VkFormat depth_image_format = VK_FORMAT_UNDEFINED;
    VkSampler sampler = VK_NULL_HANDLE;
    VkShaderModule vertex_shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    veekay::graphics::Buffer* uniform_buffer = nullptr;
    veekay::mat4 matrix;
};

// Структуры uniform буферов
struct MaterialUniforms {
    veekay::vec3 albedo_color;
    veekay::vec3 specular_color;
    float shininess;
};

struct ModelUniforms {
    veekay::mat4 model;
    MaterialUniforms material;
};

struct SceneUniforms {
    veekay::mat4 view_projection;
    veekay::mat4 shadow_projection;
    veekay::mat4 spot_shadow_projection;
    veekay::vec3 view_position;

    veekay::vec3 ambient_light_intensity;

    veekay::vec3 sun_light_direction;
    veekay::vec3 sun_light_color;

    uint32_t point_lights_count;
    uint32_t spot_lights_count;

    float shadow_bias;
    float pcf_size;
    veekay::vec2 shadow_map_size;

    float _pad0, _pad1;
};

// Глобальные экземпляры теневых карт
extern ShadowMap shadow;
extern ShadowMap spotShadow;

// Вспомогательные функции
float toRadians(float degrees);

VkShaderModule loadShaderModule(const char* path);

veekay::graphics::Texture* loadTexture(VkCommandBuffer cmd, const std::string& path) {
    if (path.empty() || !std::filesystem::exists(path)) {
        return nullptr;
    }

    std::vector<unsigned char> pixels;
    unsigned int width, height;
    unsigned int error = lodepng::decode(pixels, width, height, path);

    if (error != 0) {
        std::cerr << "Failed to load texture: " << lodepng_error_text(error) << std::endl;
        return nullptr;
    }

    return new veekay::graphics::Texture(cmd, width, height,
                                         VK_FORMAT_R8G8B8A8_UNORM, pixels.data());
}

VkSampler createTextureSampler() {
    VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = 16.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_ALWAYS,
            .minLod = 0.0f,
            .maxLod = VK_LOD_CLAMP_NONE,
            .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
    };

    VkSampler sampler;
    if (vkCreateSampler(veekay::app.vk_device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create texture sampler\n";
        return VK_NULL_HANDLE;
    }

    return sampler;
}

std::shared_ptr<Material> createTextureMaterial(VkCommandBuffer cmd, const std::string& base_name,
                                                const std::string& albedo_path,
                                                const std::string& specular_path = "",
                                                const std::string& emissive_path = "");

std::shared_ptr<Material> createColorMaterial(const veekay::vec3& albedo, const veekay::vec3& specular,
                                              float shininess, const std::string& name = "");

std::shared_ptr<Material> getMaterial(const std::string& name);

// Функции жизненного цикла приложения
// Определения inline функций
inline float toRadians(float degrees) {
    return degrees * static_cast<float>(M_PI) / 180.0f;
}

inline veekay::mat4 Transform::matrix() const {
    auto t = veekay::mat4::translation(position) *
             veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z) *
             veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y) *
             veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x) *
             veekay::mat4::scaling(scale);
    return t;
}

inline veekay::mat4 Camera::look_at(veekay::vec3 target) const {
    const veekay::vec3 forward = veekay::vec3::normalized(position - target);
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

inline veekay::mat4 Camera::view() const {
    auto t = veekay::mat4::translation(-position);
    auto r = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x) *
             veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y) *
             veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
    r = veekay::mat4::transpose(r);
    return t * r;
}

inline veekay::mat4 Camera::view_projection(float aspect_ratio, const veekay::mat4& view_matrix) const {
    auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
    return view_matrix * projection;
}

#endif // TESTBED_MAIN_H