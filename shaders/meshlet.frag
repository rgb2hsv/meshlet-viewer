#version 460
#if MESH_PIPELINE
#extension GL_EXT_mesh_shader : require
#endif
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inWorldPos;
#if MESH_PIPELINE
layout(location = 2) perprimitiveEXT flat in uint inMeshletTriangle;
#else
layout(location = 2) flat in uint inMeshletTriangle;
#endif

layout(location = 0) out vec4 outColor;

void main()
{
    Globals g = pc.globals;
    uint meshlet = inMeshletTriangle >> TRI_BITS;
    uint triangle = inMeshletTriangle & ((1u << TRI_BITS) - 1u);
    bool dimmed = g.highlight >= 0 && meshlet != uint(g.highlight);

    if (pc.pass == 1u)
    {
        outColor = vec4(vec3(dimmed ? 0.02 : 0.0), 1.0);
        return;
    }

    vec3 n = normalize(inNormal);
    if (!gl_FrontFacing)
        n = -n;
    vec3 viewDir = normalize(g.cameraPos.xyz - inWorldPos);
    // key light attached to the camera + a little headlight fill
    float key = max(dot(n, g.lightDir.xyz), 0.0);
    float fill = max(dot(n, viewDir), 0.0);
    float light = 0.12 + 0.68 * key + 0.25 * fill;

    vec3 color;
    if (g.mode == MODE_MESHLET)
        color = hashColor(meshlet) * light;
    else if (g.mode == MODE_TRIANGLE)
        color = hashColor(meshlet * 512u + triangle) * light;
    else if (g.mode == MODE_NORMAL)
        color = n * 0.5 + 0.5;
    else
        color = vec3(0.8) * light;

    if (dimmed)
        color *= 0.15;

    outColor = vec4(color, 1.0);
}
