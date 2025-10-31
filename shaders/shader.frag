#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

// Структура точечного источника света
struct PointLight {
    vec3 position;
    float _pad0;
    vec3 color;
    float intensity;
};

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    float _pad0;
    vec3 ambient_light_intensity;
    float _pad1;
    vec3 sun_light_direction;
    float _pad2;
    vec3 sun_light_color;
    float _pad3;
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad4;
    vec3 specular_color;
    float shininess;
};

layout(binding = 2, std430) readonly buffer PointLights {
    uint point_light_count;
    uint _padA;
    uint _padB;
    uint _padC;
    PointLight point_lights[]; // начинается с offset 16
};

vec3 calculateDirectionalLight(vec3 normal, vec3 view_dir) {
    // Направление света (инвертируем, так как храним направление ОТ источника)
    vec3 light_dir = normalize(-sun_light_direction);

    // Диффузная компонента
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * albedo_color * sun_light_color;

    // Specular компонента (Блинн-Фонг)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = spec * specular_color * sun_light_color;

    return diffuse + specular;
}

vec3 calculatePointLight(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 light_dir = normalize(light.position - frag_pos);

    // Расстояние до источника света
    float distance = length(light.position - frag_pos);

    // Затухание по закону обратных квадратов
    float attenuation = light.intensity / (distance * distance);

    // Диффузная компонента
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * albedo_color * light.color;

    // Specular компонента (Блинн-Фонг)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = spec * specular_color * light.color;

    return (diffuse + specular) * attenuation;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(view_position - f_position);

    // Рассеянный свет (ambient)
    vec3 ambient = ambient_light_intensity * albedo_color;

    // Направленный свет (sun)
    vec3 directional = calculateDirectionalLight(normal, view_dir);

    // Точечные источники света
    vec3 point_lighting = vec3(0.0);
    for (uint i = 0; i < point_light_count; ++i) {
        point_lighting += calculatePointLight(point_lights[i], normal, f_position, view_dir);
    }

    vec3 color = ambient + directional + point_lighting;

    color = pow(color, vec3(1.0/2.2));

    final_color = vec4(color, 1.0);
}