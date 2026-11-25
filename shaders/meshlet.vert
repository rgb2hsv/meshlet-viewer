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
layout(location = 5) flat out uint outEdgeMask;
layout(location = 6) out vec2 outUv;

vec3 position(uint corner)
{
    Vertex v = pc.vertices.v[pc.fallbackIndices.d[corner]];
    return vec3(v.px, v.py, v.pz);
}

void main()
{
    uint corner = uint(gl_VertexIndex) % 3u;
    uint first = uint(gl_VertexIndex) - corner;

    uint vi = pc.fallbackIndices.d[gl_VertexIndex];
    Vertex v = pc.vertices.v[vi];
    outUv = (pc.globals.flags & FLAG_UV_GRID) != 0u ? pc.texcoords.t[vi] : vec2(0.0);
    vec3 p = vec3(v.px, v.py, v.pz);
    gl_Position = pc.globals.viewProj * vec4(p, 1.0);
    outNormal = vec3(v.nx, v.ny, v.nz);
    outWorldPos = p;
    outBarycentric = vec3(corner == 0u, corner == 1u, corner == 2u);
    if ((pc.globals.flags & FLAG_PREVIEW) != 0u)
    {
        // raw mesh while meshlets build: no meshlet ids or edge masks exist yet
        outMeshletTriangle = 0u;
        outEdgeMask = 0u;
        outAltitudes = vec3(0.0);
        return;
    }
    outMeshletTriangle = pc.fallbackTriangles.d[gl_VertexIndex / 3];
    uint triangle = uint(gl_VertexIndex) / 3u; // fallback: edge masks are stored per drawn triangle
    outEdgeMask = (pc.edgeMasks.d[triangle >> 2] >> ((triangle & 3u) * 8u)) & 7u;
    outAltitudes = (pc.globals.flags & FLAG_WIREFRAME) != 0u
        ? triangleAltitudes(position(first), position(first + 1u), position(first + 2u))
        : vec3(0.0);
}
