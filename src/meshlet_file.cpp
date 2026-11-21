#include "meshlet_file.h"

#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace
{

constexpr uint32_t kMagic = 0x444f4c4d; // "MLOD" in little endian
constexpr uint32_t kVersion = 1;

using OutputArchive = cereal::PortableBinaryOutputArchive;
using InputArchive = cereal::PortableBinaryInputArchive;

// The GPU structs are written as raw arrays of 32-bit words (one bulk write per array instead of one
// call per field); portable binary byte-swaps each word when the reader's endianness differs.
// This needs every field to be a 4-byte scalar, which these layouts are (std430, no padding gaps).
static_assert(sizeof(Vertex) == 24 && sizeof(meshopt_Meshlet) == 16 && sizeof(MeshletBounds) == 32 && sizeof(MeshletLod) == 48 &&
                  sizeof(LodNode) == 32 && sizeof(LodGroupRange) == 8,
    "meshlet file layout: update kVersion and the serialization when these structs change");

template <class Word, class T>
void array(OutputArchive& ar, std::vector<T>& v, uint64_t)
{
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % sizeof(Word) == 0);
    ar(uint64_t(v.size()));
    if (!v.empty())
        ar(cereal::binary_data(reinterpret_cast<const Word*>(v.data()), v.size() * sizeof(T)));
}

template <class Word, class T>
void array(InputArchive& ar, std::vector<T>& v, uint64_t maxBytes)
{
    static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % sizeof(Word) == 0);
    uint64_t count = 0;
    ar(count);
    if (count > maxBytes / sizeof(T)) // corrupt count: don't try to allocate it
        throw std::runtime_error("array larger than the file");
    v.resize(size_t(count));
    if (count)
        ar(cereal::binary_data(reinterpret_cast<Word*>(v.data()), v.size() * sizeof(T)));
}

// size_t is stored as uint64 so 32- and 64-bit builds read the same files
void sizeField(OutputArchive& ar, size_t& s) { ar(uint64_t(s)); }
void sizeField(InputArchive& ar, size_t& s)
{
    uint64_t v = 0;
    ar(v);
    s = size_t(v);
}

void sizeArray(OutputArchive& ar, std::vector<size_t>& v, uint64_t maxBytes)
{
    std::vector<uint64_t> wide(v.begin(), v.end());
    array<uint64_t>(ar, wide, maxBytes);
}
void sizeArray(InputArchive& ar, std::vector<size_t>& v, uint64_t maxBytes)
{
    std::vector<uint64_t> wide;
    array<uint64_t>(ar, wide, maxBytes);
    v.resize(wide.size());
    std::transform(wide.begin(), wide.end(), v.begin(), [](uint64_t x) { return size_t(x); });
}

// One function for both directions (cereal style); maxBytes bounds array sizes when loading.
template <class Archive>
void archive(Archive& ar, Model& m, MeshletSettings& s, MeshletData& d, uint64_t maxBytes)
{
    int32_t algorithm = int32_t(s.algorithm);
    ar(algorithm, s.maxVertices, s.maxTriangles, s.minTriangles, s.coneWeight, s.splitFactor, s.fillWeight, s.optimize);
    ar(s.lod, s.lodSpatial, s.lodSimplifyRatio, s.lodPartitionSize);
    s.algorithm = MeshletAlgorithm(algorithm);

    array<uint32_t>(ar, m.vertices, maxBytes);
    array<uint32_t>(ar, m.indices, maxBytes);
    ar(m.boundsMin, m.boundsMax, m.center, m.radius, m.avgEdgeLength);

    array<uint32_t>(ar, d.meshlets, maxBytes);
    array<uint32_t>(ar, d.vertices, maxBytes);
    array<uint8_t>(ar, d.triangles, maxBytes);
    array<uint32_t>(ar, d.bounds, maxBytes);
    array<uint32_t>(ar, d.lods, maxBytes);
    array<uint8_t>(ar, d.edgeMasks, maxBytes);

    ar(d.buildSeconds, d.lod);
    sizeField(ar, d.triangleCount);
    sizeField(ar, d.baseTriangleCount);
    sizeArray(ar, d.levelClusters, maxBytes);
    sizeArray(ar, d.levelTriangles, maxBytes);
    sizeField(ar, d.coarsestTriangles);

    array<uint32_t>(ar, d.lodNodes, maxBytes);
    array<uint32_t>(ar, d.lodGroups, maxBytes);
    ar(d.lodRootCount);
    array<uint32_t>(ar, d.lodNodesPerDepth, maxBytes);

    ar(d.avgVertices, d.avgTriangles, d.vertexFill, d.triangleFill, d.avgConeAngleDeg);
    sizeField(ar, d.coneCullable);
    ar(d.coneCullableFraction);
}

// Cheap consistency checks so a damaged file fails to load instead of crashing the renderer
bool validate(const Model& m, const MeshletData& d, std::string& error)
{
    const size_t vertexCount = m.vertices.size();
    if (m.indices.size() % 3 != 0 || std::any_of(m.indices.begin(), m.indices.end(), [&](uint32_t i) { return i >= vertexCount; }))
        error = "bad index buffer";
    else if (d.meshlets.empty() || d.levelClusters.empty())
        error = "no meshlets";
    else if (d.bounds.size() != d.meshlets.size() || d.lods.size() != d.meshlets.size())
        error = "per-meshlet arrays have different sizes";
    else if (std::any_of(d.vertices.begin(), d.vertices.end(), [&](uint32_t i) { return i >= vertexCount; }))
        error = "meshlet vertex out of range";
    else
        for (const meshopt_Meshlet& ml : d.meshlets)
            if (uint64_t(ml.vertex_offset) + ml.vertex_count > d.vertices.size() ||
                uint64_t(ml.triangle_offset) + ml.triangle_count * 3ull > d.triangles.size() ||
                ml.triangle_offset / 3 + uint64_t(ml.triangle_count) > d.edgeMasks.size())
            {
                error = "meshlet range out of bounds";
                break;
            }
    return error.empty();
}

} // namespace

bool isMeshletFile(const std::string& path)
{
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return ext == kMeshletFileExtension;
}

bool saveMeshletFile(const std::string& path, const Model& model, const MeshletSettings& settings, const MeshletData& data,
    std::string& error)
{
    // write next to the target and rename, so a failed save never leaves a truncated file behind
    const std::filesystem::path target(path), temp(path + ".tmp");
    try
    {
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out)
            {
                error = "cannot open " + temp.string() + " for writing";
                return false;
            }
            OutputArchive ar(out);
            uint32_t magic = kMagic, version = kVersion;
            ar(magic, version);
            // the archive only reads through these references
            archive(ar, const_cast<Model&>(model), const_cast<MeshletSettings&>(settings), const_cast<MeshletData&>(data), 0);
            out.flush();
            if (!out)
            {
                error = "write failed: " + temp.string();
                return false;
            }
        }
        std::filesystem::rename(temp, target);
    }
    catch (const std::exception& e)
    {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        error = "saving " + path + ": " + e.what();
        return false;
    }
    return true;
}

bool loadMeshletFile(const std::string& path, Model& model, MeshletSettings& settings, MeshletData& data, std::string& error)
{
    const auto start = std::chrono::steady_clock::now();
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "cannot open " + path;
        return false;
    }
    in.seekg(0, std::ios::end);
    const uint64_t fileBytes = uint64_t(in.tellg());
    in.seekg(0, std::ios::beg);

    model = Model();
    data = MeshletData();
    try
    {
        InputArchive ar(in);
        uint32_t magic = 0, version = 0;
        ar(magic, version);
        if (magic != kMagic)
        {
            error = path + " is not a meshlet LOD file";
            return false;
        }
        if (version != kVersion)
        {
            error = path + ": unsupported meshlet file version " + std::to_string(version) + " (expected " + std::to_string(kVersion) + ")";
            return false;
        }
        archive(ar, model, settings, data, fileBytes);
    }
    catch (const std::exception& e)
    {
        error = path + ": truncated or corrupt meshlet file (" + e.what() + ")";
        return false;
    }
    if (!validate(model, data, error))
    {
        error = path + ": " + error;
        return false;
    }
    model.path = path;
    model.loadSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return true;
}
