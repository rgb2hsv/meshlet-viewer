#pragma once

// LOD hierarchies built by meshlet-optimizer (external/meshlet-optimizer): CVT charts of the mesh, then
// levels of clustered, flattened and remeshed regions, written as a .mlod file. The viewer runs its
// command line tool as a child process on a temporary PLY of the mesh and loads the .mlod it writes.

#include "model.h"

#include <cstdint>
#include <functional>
#include <string>

// The meshlet-optimizer settings the viewer exposes (defaults match the tool's own).
struct MeshletOptimizerParams
{
    // base charts
    int chartTriangles = 256;  // --chart-triangles: largest base chart
    uint32_t cvtSeed = 1;      // --cvt-seed
    bool regularize = true;    // --no-regularize
    // hierarchy
    int maxClique = 4;         // --max-clique: most regions merged into one per level
    int stopAtRegions = 2;     // --stop-at-regions
    int maxDepth = 12;         // --max-depth
    uint32_t seed = 3429841;   // --seed: region clustering
    bool keepDisks = true;     // --no-keep-disks
    bool straightenSeams = true; // --no-straighten-seams
    // remeshing
    float interiorFraction = 0.35f; // --interior-fraction: interior vertices kept per region and level
    int splitAboveFaces = 256;      // --split-above-faces (0: never)
    bool project = true;            // --no-project: snap new vertices onto the input surface
    // region borders
    bool simplifyBorders = true; // --no-simplify-borders
    bool relaxBorders = true;    // --no-relax-borders
    // run
    int threads = 0; // -j (0: all cores)
};

// Path of the meshlet-optimizer executable built with the viewer (empty when it was not built).
std::string defaultMeshletOptimizerPath();

// Writes the referenced, position-welded triangles of model to a temporary PLY, runs exe on it and loads
// the .mlod it writes into outModel / outSettings / outData (outModel.path = model.path). progress gets
// each line the tool prints. Returns false with error set on failure.
bool runMeshletOptimizer(const std::string& exe, const Model& model, const MeshletOptimizerParams& params, Model& outModel,
    MeshletSettings& outSettings, MeshletData& outData, std::string& error,
    const std::function<void(const std::string& line)>& progress = nullptr);

// The command line options for params (without input and output), for logs and --help text.
std::string meshletOptimizerArgs(const MeshletOptimizerParams& params);
