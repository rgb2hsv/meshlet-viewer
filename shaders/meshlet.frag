#version 460
#if MESH_PIPELINE
#extension GL_EXT_mesh_shader : require
#endif
// The vertex path passes barycentrics as a varying; the mesh path needs the extension.
#if MESH_PIPELINE && BARYCENTRIC
#extension GL_EXT_fragment_shader_barycentric : require
#endif
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inWorldPos;
#if MESH_PIPELINE
layout(location = 2) perprimitiveEXT flat in uint inMeshletTriangle;
layout(location = 3) perprimitiveEXT flat in vec3 inAltitudes;
#else
layout(location = 2) flat in uint inMeshletTriangle;
layout(location = 3) flat in vec3 inAltitudes;
layout(location = 4) in vec3 inBarycentric;
#endif

layout(location = 0) out vec4 outColor;

// Blender matcaps (sRGB textures, sampled as linear); array size set at pipeline creation
layout(constant_id = 0) const int MATCAP_COUNT = 1;
layout(set = 0, binding = 0) uniform sampler2D matcaps[MATCAP_COUNT];

// One color per DAG level, cycling every 8 levels so adjacent levels always differ.
vec3 lodColor(uint level)
{
    const vec3 palette[8] = vec3[8](vec3(0.20, 0.45, 0.95), vec3(0.15, 0.75, 0.85), vec3(0.25, 0.80, 0.40), vec3(0.70, 0.85, 0.25),
        vec3(0.98, 0.80, 0.20), vec3(0.98, 0.55, 0.15), vec3(0.92, 0.25, 0.20), vec3(0.75, 0.25, 0.70));
    return palette[level % 8u];
}

vec3 linearToSrgb(vec3 c)
{
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

// Matcap lookup with the view-space normal. To avoid stretching under perspective,
// the normal is expressed in an orthonormal basis built around the direction to the
// eye instead of the camera axes (same idea as Blender's viewport matcaps).
vec3 matcapColor(Globals g, vec3 n, vec3 toEye)
{
    vec3 N = vec3(dot(n, g.viewRight.xyz), dot(n, g.viewUp.xyz), dot(n, g.viewBack.xyz));
    vec3 I = normalize(vec3(dot(toEye, g.viewRight.xyz), dot(toEye, g.viewUp.xyz), dot(toEye, g.viewBack.xyz)));
    float a = 1.0 / (1.0 + I.z);
    float b = -I.x * I.y * a;
    vec3 tangent = vec3(1.0 - I.x * I.x * a, b, -I.x);
    vec3 bitangent = vec3(b, 1.0 - I.y * I.y * a, -I.y);
    vec2 uv = vec2(dot(tangent, N), dot(bitangent, N)) * 0.496 + 0.5;
    uv.y = 1.0 - uv.y; // image rows are stored top-down
    return linearToSrgb(texture(matcaps[g.matcap], uv).rgb);
}

// Coverage of a wireframe line of fixed width in model units, anti-aliased over
// one pixel. Lines thinner than a pixel fade out instead of aliasing.
float wireCoverage(float width)
{
#if MESH_PIPELINE && !BARYCENTRIC
    return 0.0; // wireframe unsupported (UI disables it)
#elif MESH_PIPELINE
    vec3 barycentric = gl_BaryCoordEXT; // perspective-correct, in primitive vertex order
#else
    vec3 barycentric = inBarycentric;
#endif
#if !MESH_PIPELINE || BARYCENTRIC
    vec3 edgeDistance = barycentric * inAltitudes; // model units, per edge
    vec3 pixel = max(fwidth(edgeDistance), vec3(1e-30));
    float halfWidth = 0.5 * width;
    // overlap of the pixel footprint [d - px/2, d + px/2] with the line [-w/2, w/2] (the line is
    // centered on the edge; each of the two triangles sharing the edge draws its own half)
    vec3 lo = max(edgeDistance - 0.5 * pixel, vec3(-halfWidth));
    vec3 hi = min(edgeDistance + 0.5 * pixel, vec3(halfWidth));
    vec3 coverage = clamp((hi - lo) / pixel, 0.0, 1.0);
    return max(coverage.x, max(coverage.y, coverage.z));
#endif
}

void main()
{
    Globals g = pc.globals;
    uint meshlet = inMeshletTriangle >> TRI_BITS;
    uint triangle = inMeshletTriangle & ((1u << TRI_BITS) - 1u);
    bool dimmed = g.highlight >= 0 && meshlet != uint(g.highlight);

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
    else if (g.mode == MODE_LOD)
        color = lodColor(pc.lods.l[meshlet].level) * light;
    else if (g.matcap >= 0)
        color = matcapColor(g, n, viewDir);
    else
        color = vec3(0.8) * light;

    if ((g.flags & FLAG_WIREFRAME) != 0u)
    {
        // grayscale lines whose shade follows the cluster's LOD level; the shades cycle in an
        // order where adjacent levels always contrast, and stay dark enough for light matcaps
        const float shades[4] = float[4](0.02, 0.38, 0.18, 0.55);
        vec3 wireColor = vec3(shades[pc.lods.l[meshlet].level % 4u]);
        color = mix(color, wireColor, wireCoverage(g.wireWidth));
    }

    if (dimmed)
        color *= 0.15;

    outColor = vec4(color, 1.0);
}
