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
layout(location = 4) perprimitiveEXT flat in uint inEdgeMask;
#else
layout(location = 2) flat in uint inMeshletTriangle;
layout(location = 3) flat in vec3 inAltitudes;
layout(location = 4) in vec3 inBarycentric;
layout(location = 5) flat in uint inEdgeMask;
#endif

layout(location = 0) out vec4 outColor;

// Blender matcaps (sRGB textures, sampled as linear); array size set at pipeline creation
layout(constant_id = 0) const int MATCAP_COUNT = 1;
layout(set = 0, binding = 0) uniform sampler2D matcaps[MATCAP_COUNT];

// Color gradient over the LOD levels: full detail (level 0) blue -> cyan -> green -> yellow ->
// orange for the coarsest level.
vec3 lodGradient(uint level, uint levelCount)
{
    const vec3 stops[5] = vec3[5](vec3(0.20, 0.45, 0.95), vec3(0.15, 0.75, 0.85), vec3(0.30, 0.80, 0.40), vec3(0.95, 0.85, 0.25),
        vec3(0.95, 0.45, 0.20));
    float t = levelCount > 1u ? float(level) / float(levelCount - 1u) : 0.0;
    float x = clamp(t, 0.0, 1.0) * 4.0;
    int i = min(int(x), 3);
    return mix(stops[i], stops[i + 1], x - float(i));
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

// Pixel coverage of lines of fixed width in model units, centered on triangle edges and
// anti-aliased over one pixel (lines thinner than a pixel fade out instead of aliasing).
// Returns (triangle wireframe, meshlet outline) coverage; the outline only uses the edges on
// the meshlet's perimeter. Each of the two triangles sharing an edge draws its own half.
vec2 lineCoverage(float wireWidth, float outlineWidth)
{
#if MESH_PIPELINE && !BARYCENTRIC
    return vec2(0.0); // needs barycentrics (UI disables the wireframe)
#else
#if MESH_PIPELINE
    vec3 barycentric = gl_BaryCoordEXT; // perspective-correct, in primitive vertex order
#else
    vec3 barycentric = inBarycentric;
#endif
    vec3 edgeDistance = barycentric * inAltitudes; // model units, per edge
    vec3 pixel = max(fwidth(edgeDistance), vec3(1e-30));

    // overlap of the pixel footprint [d - px/2, d + px/2] with the line [-w/2, w/2]
    vec3 lo = edgeDistance - 0.5 * pixel, hi = edgeDistance + 0.5 * pixel;
    vec3 wire = clamp((min(hi, vec3(0.5 * wireWidth)) - max(lo, vec3(-0.5 * wireWidth))) / pixel, 0.0, 1.0);
    vec3 outline = clamp((min(hi, vec3(0.5 * outlineWidth)) - max(lo, vec3(-0.5 * outlineWidth))) / pixel, 0.0, 1.0);
    outline *= vec3((inEdgeMask & 1u) != 0u, (inEdgeMask & 2u) != 0u, (inEdgeMask & 4u) != 0u);

    return vec2(max(wire.x, max(wire.y, wire.z)), max(outline.x, max(outline.y, outline.z)));
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
        color = lodGradient(pc.lods.l[meshlet].level, g.lodLevelCount) * light;
    else if (g.mode == MODE_TRIANGLE)
        color = hashColor(meshlet * 512u + triangle) * light;
    else if (g.mode == MODE_NORMAL)
        color = n * 0.5 + 0.5;
    else if (g.matcap >= 0)
        color = matcapColor(g, n, viewDir);
    else
        color = vec3(0.8) * light;

    if ((g.flags & FLAG_WIREFRAME) != 0u)
    {
        // black: thinner, slightly lighter triangle wireframe, full meshlet outlines on top
        vec2 lines = lineCoverage(g.wireWidth, g.outlineWidth);
        color = mix(color, vec3(0.0), lines.x * 0.75);
        color = mix(color, vec3(0.0), lines.y);
    }

    if (dimmed)
        color *= 0.15;

    outColor = vec4(color, 1.0);
}
