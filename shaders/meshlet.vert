#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Fallback for GPUs without VK_EXT_mesh_shader: non-indexed draw that pulls
// vertices from the meshlet-ordered index list (no culling). Every triangle has
// its own three vertices, so barycentrics can be passed as a plain varying.
layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outWorldPos;
layout(location = 2) flat out uint outMeshletTriangle;
layout(location = 3) flat out vec3 outAltitudes;
layout(location = 4) out vec3 outBarycentric;

vec3 position(uint corner)
{
    Vertex v = pc.vertices.v[pc.fallbackIndices.d[corner]];
    return vec3(v.px, v.py, v.pz);
}

void main()
{
    uint corner = uint(gl_VertexIndex) % 3u;
    uint first = uint(gl_VertexIndex) - corner;

    Vertex v = pc.vertices.v[pc.fallbackIndices.d[gl_VertexIndex]];
    vec3 p = vec3(v.px, v.py, v.pz);
    gl_Position = pc.globals.viewProj * vec4(p, 1.0);
    outNormal = vec3(v.nx, v.ny, v.nz);
    outWorldPos = p;
    outMeshletTriangle = pc.fallbackTriangles.d[gl_VertexIndex / 3];
    outBarycentric = vec3(corner == 0u, corner == 1u, corner == 2u);
    outAltitudes = (pc.globals.flags & FLAG_WIREFRAME) != 0u
        ? triangleAltitudes(position(first), position(first + 1u), position(first + 2u))
        : vec3(0.0);
}
