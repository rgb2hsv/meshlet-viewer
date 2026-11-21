#pragma once

#include "model.h"

#include <string>

// Meshlet LOD files (.mlod): the model's mesh, the settings it was built with and the full
// MeshletData (clusters, bounds, LOD DAG data and selection BVH) in one cereal portable binary
// archive, so a later run can skip loading the source model and building the hierarchy.
// Portable binary stores the writer's endianness and swaps on load, so files move between machines.

inline constexpr const char* kMeshletFileExtension = ".mlod";

bool isMeshletFile(const std::string& path);

bool saveMeshletFile(const std::string& path, const Model& model, const MeshletSettings& settings, const MeshletData& data,
    std::string& error);

// Fills model (path = the .mlod file), the settings the hierarchy was built with, and data.
bool loadMeshletFile(const std::string& path, Model& model, MeshletSettings& settings, MeshletData& data, std::string& error);
