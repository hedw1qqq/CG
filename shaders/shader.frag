#version 450

layout (location = 0) in vec3 frag_color;

// NOTE: Pixel color
layout (location = 0) out vec4 final_color;

// NOTE: Must match declaration order of a C struct
layout (push_constant, std430) uniform ShaderConstants {
    mat4 projection;
    mat4 transform;
    //vec3 color;
};

void main() {
    final_color = vec4(frag_color, 1.0f);
}