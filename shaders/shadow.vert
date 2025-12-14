#version 450

layout (location = 0) in vec3 v_position;

// Binding 0: Shadow Matrix (View * Projection)
layout (binding = 0, std140) uniform ShadowUniforms {
    mat4 shadow_projection;
};

// Binding 1: Model Matrix (Dynamic)
layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
};

void main() {
    gl_Position = shadow_projection * model * vec4(v_position, 1.0f);
}