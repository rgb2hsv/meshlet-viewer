#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

// Fallback for GPUs without VK_EXT_mesh_shader: non-indexed draw that pulls
// vertices from the meshlet-ordered index list (no culling).
layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outWorldPos;
layout(location = 2) flat out uint outMeshletTriangle;

void main()
{
    Vertex v = pc.vertices.v[pc.fallbackIndices.d[gl_VertexIndex]];
    vec3 p = vec3(v.px, v.py, v.pz);
    gl_Position = pc.globals.viewProj * vec4(p, 1.0);
    outNormal = vec3(v.nx, v.ny, v.nz);
    outWorldPos = p;
    outMeshletTriangle = pc.fallbackTriangles.d[gl_VertexIndex / 3];
}
