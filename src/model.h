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
    float avgEdgeLength = 0; // model units; used to pick a default wireframe width
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

    // Cluster LOD (meshoptimizer demo/clusterlod.h): builds a DAG of progressively
    // simplified clusters instead of a single meshlet level. Uses maxVertices,
    // min/maxTriangles; the clusterizer is flex (splitFactor) or spatial (fillWeight).
    bool lod = true;
    bool lodSpatial = false;        // clodConfig::cluster_spatial
    float lodSimplifyRatio = 0.5f;  // target triangle reduction per level
    int lodPartitionSize = 16;      // clusters per group that get simplified together
};

// Per-cluster LOD selection data (std430 layout, 48 bytes). A cluster is drawn when the
// projected coarse error is above the threshold and the projected own error is at or below it.
struct MeshletLod
{
    float selfCenter[3];   // bounds of this cluster's own simplification error
    float selfRadius;
    float coarseCenter[3]; // bounds of the coarser clusters that would replace this one
    float coarseRadius;
    float selfError;       // 0 for full-detail clusters
    float coarseError;     // FLT_MAX when nothing coarser exists (terminal)
    uint32_t level;        // DAG depth of the cluster's group
    uint32_t pad;
};

// GPU-friendly bounds: sphere + normal cone (std430 layout, 32 bytes)
struct MeshletBounds
{
    float center[3];
    float radius;
    float coneAxis[3];
    float coneCutoff;
};

// BVH node over cluster groups (clodBuildHierarchy), std430 layout (32 bytes). Bounds are the
// conservative merge of the subtree's group error spheres; error is the subtree's maximum.
struct LodNode
{
    float center[3];
    float radius;
    float error;          // FLT_MAX when the subtree contains a terminal group
    int32_t group;        // leaf: group index; internal: -1
    uint32_t childOffset; // internal: children are nodes[childOffset, childOffset + childCount)
    uint32_t childCount;
};

struct LodGroupRange
{
    uint32_t firstCluster; // a group's clusters are contiguous in MeshletData::meshlets
    uint32_t clusterCount;
};

struct MeshletData
{
    std::vector<meshopt_Meshlet> meshlets;
    std::vector<uint32_t> vertices;  // meshlet-local -> global vertex index
    std::vector<uint8_t> triangles;  // 3 local indices per triangle, padded to a multiple of 4 bytes
    std::vector<MeshletBounds> bounds;
    std::vector<MeshletLod> lods; // one per meshlet; level 0 + "always selected" without LOD
    // per triangle (index triangle_offset / 3 + t): bit i set when the edge opposite local vertex i
    // lies on the meshlet's perimeter (used by no other triangle of the meshlet); padded to 4 bytes
    std::vector<uint8_t> edgeMasks;

    double buildSeconds = 0;
    size_t triangleCount = 0;     // all levels
    size_t baseTriangleCount = 0; // level 0
    bool lod = false;
    std::vector<size_t> levelClusters, levelTriangles; // per DAG level
    size_t coarsestTriangles = 0; // triangles in the coarsest possible cut (terminal clusters)

    // LOD selection BVH (only with lod): one tree per DAG level, roots are nodes[0, rootCount)
    std::vector<LodNode> lodNodes;
    std::vector<LodGroupRange> lodGroups;
    uint32_t lodRootCount = 0;
    std::vector<uint32_t> lodNodesPerDepth; // node count at each BVH tree depth (dispatch sizes)
    float avgVertices = 0, avgTriangles = 0; // level 0
    float vertexFill = 0, triangleFill = 0; // average utilisation of max sizes
    float avgConeAngleDeg = 0; // average normal-cone spread of cullable meshlets
    size_t coneCullable = 0; // level-0 meshlets whose cone cutoff < 1 (can be backface culled)
    float coneCullableFraction = 0;
};

bool buildMeshlets(const Model& model, const MeshletSettings& settings, MeshletData& out, std::string& error);
