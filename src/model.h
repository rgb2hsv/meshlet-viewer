#pragma once

#include <meshoptimizer.h>

#include <cstdint>
#include <string>
#include <vector>

struct Vertex
{
    float px, py, pz;
    float nx, ny, nz;
};

struct Model
{
    std::string path;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    float boundsMin[3] = {0, 0, 0};
    float boundsMax[3] = {0, 0, 0};
    double loadSeconds = 0;
};

// Loads .ply (ascii, binary little/big endian) or .obj; computes normals when the file has none.
// Optionally reorders the index buffer for vertex cache locality (helps meshlet builders).
bool loadModel(const std::string& path, Model& model, std::string& error, bool optimizeVertexCache = true);

enum class MeshletAlgorithm : int
{
    Greedy = 0,  // meshopt_buildMeshlets
    Flex = 1,    // meshopt_buildMeshletsFlex
    Spatial = 2, // meshopt_buildMeshletsSpatial
    Scan = 3,    // meshopt_buildMeshletsScan
};

struct MeshletSettings
{
    MeshletAlgorithm algorithm = MeshletAlgorithm::Greedy;
    int maxVertices = 64;
    int maxTriangles = 124;
    int minTriangles = 32; // Flex/Spatial
    float coneWeight = 0.25f;
    float splitFactor = 2.0f; // Flex
    float fillWeight = 0.5f;  // Spatial
    bool optimize = true;     // meshopt_optimizeMeshlet
};

// GPU-friendly bounds: sphere + normal cone (std430 layout, 32 bytes)
struct MeshletBounds
{
    float center[3];
    float radius;
    float coneAxis[3];
    float coneCutoff;
};

struct MeshletData
{
    std::vector<meshopt_Meshlet> meshlets;
    std::vector<uint32_t> vertices;  // meshlet-local -> global vertex index
    std::vector<uint8_t> triangles;  // 3 local indices per triangle, padded to a multiple of 4 bytes
    std::vector<MeshletBounds> bounds;

    double buildSeconds = 0;
    size_t triangleCount = 0;
    float avgVertices = 0, avgTriangles = 0;
    float vertexFill = 0, triangleFill = 0; // average utilisation of max sizes
    float avgConeAngleDeg = 0; // average normal-cone spread of cullable meshlets
    size_t coneCullable = 0; // meshlets whose cone cutoff < 1 (can be backface culled)
};

bool buildMeshlets(const Model& model, const MeshletSettings& settings, MeshletData& out, std::string& error);
