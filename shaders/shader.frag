#version 450

layout(location = 0) in vec3 f_position;
layout(location = 1) in vec3 f_normal;
layout(location = 2) in vec2 f_uv;
layout(location = 3) in vec4 f_shadow_position;

layout(location = 0) out vec4 final_color;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position; float _pad0;
    vec3 ambient_light_intensity; float _pad1;
    vec3 sun_light_direction; float _pad2;
    vec3 sun_light_color; float _pad3;
    uint point_lights_count;
    uint spot_lights_count;
    float _pad4; float _pad5;
    mat4 shadow_projection;
};

layout(set = 0, binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color; float _pad6;
    vec3 specular_color; float _pad7;
    float shininess; float _pad8; float _pad9; float _pad10;
};

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float _pad0;
};

struct SpotLight {
    vec3 position;
    float radius;
    vec3 direction;
    float angle;
    vec3 color;
    float _pad0;
};

layout(set = 0, binding = 2, std430) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(set = 0, binding = 3, std430) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

// Правильное объявление теневого сэмплера
layout(set = 0, binding = 4) uniform sampler2DShadow shadow_texture;

layout(set = 1, binding = 0) uniform sampler2D albedo_texture;
layout(set = 1, binding = 1) uniform sampler2D specular_texture;
layout(set = 1, binding = 2) uniform sampler2D emissive_texture;

float calculateShadow(vec4 shadow_position) {
    vec3 proj_coords = shadow_position.xyz / shadow_position.w;

    // Преобразование из [-1, 1] в [0, 1]
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    // Проверка границ shadow map
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
    proj_coords.y < 0.0 || proj_coords.y > 1.0 ||
    proj_coords.z > 1.0) {
        return 1.0;
    }

    float current_depth = proj_coords.z;

    // Динамический bias
    vec3 light_dir = normalize(-sun_light_direction);
    float bias = max(0.005 * (1.0 - dot(normalize(f_normal), light_dir)), 0.001);

    // PCF фильтрация
    float shadow = 0.0;
    // textureSize возвращает ivec2, берем обратное значение для размера текселя
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_texture, 0));

    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            // Для sampler2DShadow третий компонент координаты - это глубина для сравнения (Dref)
            // texture() автоматически выполняет сравнение и возвращает float [0..1]
            shadow += texture(shadow_texture, vec3(proj_coords.xy + vec2(x, y) * texel_size, current_depth - bias));
        }
    }

    shadow /= 9.0;

    return shadow;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(view_position - f_position);

    vec3 albedo_tex = texture(albedo_texture, f_uv).rgb;
    float specular_mask = texture(specular_texture, f_uv).r;
    vec3 emissive_tex = texture(emissive_texture, f_uv).rgb;

    vec3 albedo = albedo_color * albedo_tex;
    vec3 emissive = emissive_tex * 0.1;

    float shadow = calculateShadow(f_shadow_position);

    vec3 total_light = ambient_light_intensity * albedo + emissive;

    // Sun lighting
    vec3 sun_light_dir = normalize(-sun_light_direction);
    float sun_diffuse = max(dot(normal, sun_light_dir), 0.0);

    vec3 sun_half_dir = normalize(sun_light_dir + view_dir);
    float sun_specular = pow(max(dot(normal, sun_half_dir), 0.0), shininess);

    // Применяем тень к диффузной и спекулярной составляющей солнца
    total_light += shadow * (sun_diffuse * albedo + sun_specular * specular_color * specular_mask) * sun_light_color;

    // Point lights
    for (uint i = 0; i < point_lights_count; ++i) {
        PointLight light = point_lights[i];
        vec3 light_dir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);

        if (distance > light.radius) continue;

        float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
        float radius_attenuation = 1.0 - smoothstep(light.radius * 0.7, light.radius, distance);
        attenuation *= radius_attenuation;

        float diffuse = max(dot(normal, light_dir), 0.0);
        vec3 diffuse_color = diffuse * light.color * albedo;

        vec3 half_dir = normalize(light_dir + view_dir);
        float specular = pow(max(dot(normal, half_dir), 0.0), shininess);
        vec3 specular_result = specular * light.color * specular_color * specular_mask;

        total_light += (diffuse_color + specular_result) * attenuation;
    }

    // Spot lights
    for (uint i = 0; i < spot_lights_count; ++i) {
        SpotLight light = spot_lights[i];
        vec3 light_dir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);

        if (distance > light.radius) continue;

        vec3 spot_dir = normalize(-light.direction);
        float cos_theta = dot(light_dir, spot_dir);

        if (cos_theta < light.angle) continue;

        float attenuation = 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance);
        float outer_angle = light.angle * 0.8;
        float spot_factor = clamp((cos_theta - outer_angle) / (light.angle - outer_angle), 0.0, 1.0);
        float radius_attenuation = 1.0 - smoothstep(light.radius * 0.7, light.radius, distance);
        attenuation *= spot_factor * radius_attenuation;

        float diffuse = max(dot(normal, light_dir), 0.0);
        vec3 diffuse_color = diffuse * light.color * albedo;

        vec3 half_dir = normalize(light_dir + view_dir);
        float specular = pow(max(dot(normal, half_dir), 0.0), shininess);
        vec3 specular_result = specular * light.color * specular_color * specular_mask;

        total_light += (diffuse_color + specular_result) * attenuation;
    }

    final_color = vec4(total_light, 1.0);
}
