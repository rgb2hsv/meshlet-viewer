// Shared declarations. All data is accessed through buffer device addresses
// passed in push constants, so there are no descriptor sets.
#extension GL_EXT_buffer_reference : require

#define TASK_GROUP_SIZE 32
#define MESH_GROUP_SIZE 32
#define MAX_MESHLET_VERTICES 256
#define MAX_MESHLET_TRIANGLES 256

// Packs meshlet index + meshlet-local triangle index (triangle < 512)
#define TRI_BITS 9u

#define FLAG_FRUSTUM_CULL 1u
#define FLAG_CONE_CULL 2u
#define FLAG_WIREFRAME 4u

#define MODE_MESHLET 0u
#define MODE_TRIANGLE 1u
#define MODE_NORMAL 2u
#define MODE_SHADED 3u

struct Vertex
{
    float px, py, pz;
    float nx, ny, nz;
};

struct Meshlet
{
    uint vertexOffset;
    uint triangleOffset; // in bytes (3 per triangle)
    uint vertexCount;
    uint triangleCount;
};

struct Bounds
{
    vec4 centerRadius;
    vec4 coneAxisCutoff;
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer VertexBuffer { Vertex v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer MeshletBuffer { Meshlet m[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer BoundsBuffer { Bounds b[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer UintBuffer { uint d[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer StatsBuffer
{
    uint visibleMeshlets;
    uint visibleTriangles;
};

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Globals
{
    mat4 viewProj;
    vec4 frustum[4];  // left, right, bottom, top (world space, normalized)
    vec4 cameraPos;   // xyz
    vec4 lightDir;    // xyz, world space, towards the light
    uint meshletCount;
    uint mode;
    int highlight;    // -1 = none
    uint flags;
    float wireWidth;  // wireframe line width in model units
    int matcap;       // index into matcaps[], -1 = plain lighting
    float pad0, pad1;
    vec4 viewRight;   // camera basis in world space (view-space x, y, z)
    vec4 viewUp;
    vec4 viewBack;
};

layout(push_constant) uniform PushConstants
{
    Globals globals;
    VertexBuffer vertices;
    MeshletBuffer meshlets;
    UintBuffer meshletVertices;
    UintBuffer meshletTriangles;  // bytes packed 4 per uint
    BoundsBuffer bounds;
    UintBuffer fallbackIndices;   // vertex path: global vertex index per corner
    UintBuffer fallbackTriangles; // vertex path: packed meshlet/triangle id per triangle
    StatsBuffer stats;
} pc;

// Altitudes of triangle (a, b, c): distance from each vertex to its opposite edge.
// barycentric[i] * altitude[i] is then the in-plane distance to edge i, in model units.
vec3 triangleAltitudes(vec3 a, vec3 b, vec3 c)
{
    float doubleArea = length(cross(b - a, c - a));
    vec3 edgeLengths = vec3(length(c - b), length(a - c), length(b - a)); // edge opposite a, b, c
    return doubleArea / max(edgeLengths, vec3(1e-30));
}

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

vec3 hashColor(uint x)
{
    uint h = hash(x);
    return vec3(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u)) / 255.0 * 0.75 + 0.25;
}
