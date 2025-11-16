#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

struct PointLight {
    vec3 position;
    float _pad0;
    vec3 color;
    float intensity;
};

struct SpotLight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float cutoff_angle;
    vec3 color;
    float outer_cutoff_angle;
    float intensity;
    float _pad1;
    float _pad2;
    float _pad3;
};

// set = 0: UBO/SSBO, как и раньше, только явно указан set
layout (set = 0, binding = 0, std140) uniform SceneUniforms {
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

layout (set = 0, binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad4;
    vec3 specular_color;
    float shininess;
};

layout (set = 0, binding = 2, std430) readonly buffer PointLights {
    uint point_light_count;
    uint _padA;
    uint _padB;
    uint _padC;
    PointLight point_lights[];
};

layout (set = 0, binding = 3, std430) readonly buffer SpotLights {
    uint spot_light_count;
    uint _padD;
    uint _padE;
    uint _padF;
    SpotLight spot_lights[];
};

// set = 1: материалы
layout (set = 1, binding = 0) uniform sampler2D albedo_texture;
layout (set = 1, binding = 1) uniform sampler2D specular_texture;
layout (set = 1, binding = 2) uniform sampler2D emissive_texture;

// Нетривиальное сэмплирование: вихревая модуляция UV + мульти-сэмплинг (крест)
vec3 sampleAlbedoFancy(vec2 uv) {

    vec2 c = uv * 2.0 - 1.0;
    float r = length(c) + 1e-5;
    float ang = atan(c.y, c.x) + 0.35 * sin(9.0 * r);
    vec2 swirl = vec2(cos(ang), sin(ang)) * r;
    vec2 uv1 = (swirl + 1.0) * 0.5;

    vec2 o = vec2(1.0/1024.0, 1.0/1024.0);
    vec3 c0 = texture(albedo_texture, uv1).rgb;
    vec3 c1 = texture(albedo_texture, uv1 + vec2( o.x, 0)).rgb;
    vec3 c2 = texture(albedo_texture, uv1 + vec2(-o.x, 0)).rgb;
    vec3 c3 = texture(albedo_texture, uv1 + vec2(0,  o.y)).rgb;
    vec3 c4 = texture(albedo_texture, uv1 + vec2(0, -o.y)).rgb;

    return (c0*0.6 + (c1+c2+c3+c4)*0.1);
}

vec3 calculateDirectionalLight(vec3 albedo, vec3 normal, vec3 view_dir, float spec_mask) {

    vec3 light_dir = normalize(-sun_light_direction);

    // Диффузная компонента
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * albedo * sun_light_color;

    // Specular компонента (Блинн-Фонг)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = spec * specular_color * spec_mask * sun_light_color;

    return diffuse + specular;
}

vec3 calculatePointLight(PointLight light, vec3 albedo, vec3 normal, vec3 frag_pos, vec3 view_dir, float spec_mask) {

    vec3 light_dir = normalize(light.position - frag_pos);
    float distance = length(light.position - frag_pos);

    // Затухание по закону обратных квадратов
    float attenuation = light.intensity / (distance * distance);

    // Диффузная компонента
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * albedo * light.color;

    // Specular компонента (Блинн-Фонг)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = spec * specular_color * spec_mask * light.color;

    return (diffuse + specular) * attenuation;
}

vec3 calculateSpotLight(SpotLight light, vec3 albedo, vec3 normal, vec3 frag_pos, vec3 view_dir, float spec_mask) {
    vec3 light_dir = normalize(light.position - frag_pos);
    float distance = length(light.position - frag_pos);

    // Затухание по расстоянию
    float attenuation = light.intensity / (distance * distance);

    // Угол между направлением прожектора и вектором к фрагменту
    float theta = dot(light_dir, normalize(-light.direction));

    // Плавное затухание по углу
    float epsilon = light.cutoff_angle - light.outer_cutoff_angle;
    float intensity = clamp((theta - light.outer_cutoff_angle) / epsilon, 0.0, 1.0);

    // Диффузная компонента
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * albedo * light.color;

    // Specular компонента (Блинн-Фонг)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), shininess);
    vec3 specular = spec * specular_color * spec_mask * light.color;
    return (diffuse + specular) * attenuation * intensity;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(view_position - f_position);

    // albedo берём с нетривиальным сэмплингом
    vec3 tex = sampleAlbedoFancy(f_uv);
    vec3 albedo = albedo_color * tex;

    // Маска зеркальности из текстуры (R-канал)
    float spec_mask = texture(specular_texture, f_uv).r;

    vec3 ambient = ambient_light_intensity * albedo;

    // sun
    vec3 directional = calculateDirectionalLight(albedo, normal, view_dir, spec_mask);

    vec3 point_lighting = vec3(0.0);
    for (uint i = 0; i < point_light_count; ++i) {
        point_lighting += calculatePointLight(point_lights[i], albedo, normal, f_position, view_dir, spec_mask);
    }

    vec3 spot_lighting = vec3(0.0);
    for (uint i = 0; i < spot_light_count; ++i) {
        spot_lighting += calculateSpotLight(spot_lights[i], albedo, normal, f_position, view_dir, spec_mask);
    }

    // игнорируем затенениея для emissive
    vec3 emissive = texture(emissive_texture, f_uv).rgb * 0.1;

    vec3 color = ambient + directional + point_lighting + spot_lighting + emissive;
    color = pow(color, vec3(1.0/2.2)); // Gamma correction
    final_color = vec4(color, 1.0);
}
