#include "model.h"

#define CLUSTERLOD_IMPLEMENTATION
#include "../external/meshoptimizer/demo/clusterlod.h"

#define FAST_OBJ_IMPLEMENTATION
#include "../external/meshoptimizer/extern/fast_obj.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace
{

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string lowerExtension(const std::string& path)
{
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos)
        return {};
    std::string ext = path.substr(dot + 1);
    for (char& c : ext)
        c = char(std::tolower(static_cast<unsigned char>(c)));
    return ext;
}

bool readFile(const std::string& path, std::vector<char>& data, std::string& error)
{
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file)
    {
        error = "cannot open " + path;
        return false;
    }
    std::fseek(file, 0, SEEK_END);
    long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    data.resize(size_t(size));
    size_t read = size > 0 ? std::fread(data.data(), 1, size_t(size), file) : 0;
    std::fclose(file);
    if (read != size_t(size))
    {
        error = "failed to read " + path;
        return false;
    }
    return true;
}

// Runs fn(begin, end) over [0, count) on all hardware threads.
template <typename Fn>
void parallelFor(size_t count, Fn fn)
{
    unsigned threadCount = std::max(1u, std::thread::hardware_concurrency());
    if (count < 4096 || threadCount == 1)
    {
        fn(size_t(0), count);
        return;
    }
    std::vector<std::thread> threads;
    size_t chunk = (count + threadCount - 1) / threadCount;
    for (unsigned t = 0; t < threadCount; ++t)
    {
        size_t begin = t * chunk, end = std::min(count, begin + chunk);
        if (begin >= end)
            break;
        threads.emplace_back([=] { fn(begin, end); });
    }
    for (std::thread& thread : threads)
        thread.join();
}

// ---------------------------------------------------------------------------
// PLY
// ---------------------------------------------------------------------------

enum class PlyType : uint8_t
{
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64,
    Invalid
};

PlyType parsePlyType(const char* name)
{
    struct Entry
    {
        const char* name;
        PlyType type;
    };
    static const Entry entries[] = {
        {"char", PlyType::Int8}, {"int8", PlyType::Int8}, {"uchar", PlyType::UInt8}, {"uint8", PlyType::UInt8},
        {"short", PlyType::Int16}, {"int16", PlyType::Int16}, {"ushort", PlyType::UInt16}, {"uint16", PlyType::UInt16},
        {"int", PlyType::Int32}, {"int32", PlyType::Int32}, {"uint", PlyType::UInt32}, {"uint32", PlyType::UInt32},
        {"float", PlyType::Float32}, {"float32", PlyType::Float32}, {"double", PlyType::Float64}, {"float64", PlyType::Float64},
    };
    for (const Entry& e : entries)
        if (std::strcmp(e.name, name) == 0)
            return e.type;
    return PlyType::Invalid;
}

size_t plyTypeSize(PlyType type)
{
    switch (type)
    {
    case PlyType::Int8:
    case PlyType::UInt8:
        return 1;
    case PlyType::Int16:
    case PlyType::UInt16:
        return 2;
    case PlyType::Int32:
    case PlyType::UInt32:
    case PlyType::Float32:
        return 4;
    case PlyType::Float64:
        return 8;
    default:
        return 0;
    }
}

struct PlyProperty
{
    std::string name;
    PlyType type = PlyType::Invalid;
    bool isList = false;
    PlyType countType = PlyType::Invalid;
};

struct PlyElement
{
    std::string name;
    size_t count = 0;
    std::vector<PlyProperty> properties;
};

enum class PlyFormat
{
    Ascii,
    BinaryLE,
    BinaryBE
};

template <typename T>
T loadUnaligned(const char* p, bool swap)
{
    T value;
    if (!swap)
    {
        std::memcpy(&value, p, sizeof(T));
        return value;
    }
    char bytes[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); ++i)
        bytes[i] = p[sizeof(T) - 1 - i];
    std::memcpy(&value, bytes, sizeof(T));
    return value;
}

double readBinary(const char* p, PlyType type, bool swap)
{
    switch (type)
    {
    case PlyType::Int8:
        return double(int8_t(*p));
    case PlyType::UInt8:
        return double(uint8_t(*p));
    case PlyType::Int16:
        return double(loadUnaligned<int16_t>(p, swap));
    case PlyType::UInt16:
        return double(loadUnaligned<uint16_t>(p, swap));
    case PlyType::Int32:
        return double(loadUnaligned<int32_t>(p, swap));
    case PlyType::UInt32:
        return double(loadUnaligned<uint32_t>(p, swap));
    case PlyType::Float32:
        return double(loadUnaligned<float>(p, swap));
    case PlyType::Float64:
        return loadUnaligned<double>(p, swap);
    default:
        return 0;
    }
}

uint32_t readBinaryIndex(const char* p, PlyType type, bool swap)
{
    switch (type)
    {
    case PlyType::Int8:
    case PlyType::UInt8:
        return uint8_t(*p);
    case PlyType::Int16:
    case PlyType::UInt16:
        return loadUnaligned<uint16_t>(p, swap);
    case PlyType::Int32:
    case PlyType::UInt32:
        return loadUnaligned<uint32_t>(p, swap);
    default:
        return uint32_t(readBinary(p, type, swap));
    }
}

bool loadPly(const std::string& path, Model& model, bool& hasNormals, std::string& error)
{
    std::vector<char> data;
    if (!readFile(path, data, error))
        return false;
    data.push_back('\0'); // terminator for ascii parsing

    const char* cursor = data.data();
    const char* end = data.data() + data.size() - 1;

    auto nextLine = [&](std::string& line) -> bool {
        if (cursor >= end)
            return false;
        const char* eol = static_cast<const char*>(std::memchr(cursor, '\n', size_t(end - cursor)));
        if (!eol)
            eol = end;
        line.assign(cursor, eol);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        cursor = eol < end ? eol + 1 : end;
        return true;
    };

    std::string line;
    if (!nextLine(line) || line != "ply")
    {
        error = "not a PLY file";
        return false;
    }

    PlyFormat format = PlyFormat::Ascii;
    std::vector<PlyElement> elements;
    bool headerDone = false;

    while (nextLine(line))
    {
        char word[64] = {}, a[64] = {}, b[64] = {}, c[64] = {}, d[64] = {};
        int n = std::sscanf(line.c_str(), "%63s %63s %63s %63s %63s", word, a, b, c, d);
        if (n <= 0)
            continue;
        if (!std::strcmp(word, "format"))
        {
            if (!std::strcmp(a, "ascii"))
                format = PlyFormat::Ascii;
            else if (!std::strcmp(a, "binary_little_endian"))
                format = PlyFormat::BinaryLE;
            else if (!std::strcmp(a, "binary_big_endian"))
                format = PlyFormat::BinaryBE;
            else
            {
                error = std::string("unknown PLY format ") + a;
                return false;
            }
        }
        else if (!std::strcmp(word, "element") && n >= 3)
        {
            PlyElement element;
            element.name = a;
            element.count = size_t(std::strtoull(b, nullptr, 10));
            elements.push_back(element);
        }
        else if (!std::strcmp(word, "property") && !elements.empty())
        {
            PlyProperty prop;
            if (!std::strcmp(a, "list") && n >= 5)
            {
                prop.isList = true;
                prop.countType = parsePlyType(b);
                prop.type = parsePlyType(c);
                prop.name = d;
            }
            else if (n >= 3)
            {
                prop.type = parsePlyType(a);
                prop.name = b;
            }
            if (prop.type == PlyType::Invalid || (prop.isList && prop.countType == PlyType::Invalid))
            {
                error = "unsupported PLY property: " + line;
                return false;
            }
            elements.back().properties.push_back(prop);
        }
        else if (!std::strcmp(word, "end_header"))
        {
            headerDone = true;
            break;
        }
    }

    if (!headerDone)
    {
        error = "PLY header has no end_header";
        return false;
    }

    hasNormals = false;
    const bool swap = format == PlyFormat::BinaryBE; // assumes a little-endian host

    for (const PlyElement& element : elements)
    {
        const bool isVertex = element.name == "vertex";
        const bool isFace = element.name == "face";

        // property slots we care about
        int pos[3] = {-1, -1, -1}, nrm[3] = {-1, -1, -1}, faceList = -1;
        for (size_t i = 0; i < element.properties.size(); ++i)
        {
            const std::string& name = element.properties[i].name;
            if (isVertex)
            {
                if (name == "x") pos[0] = int(i);
                if (name == "y") pos[1] = int(i);
                if (name == "z") pos[2] = int(i);
                if (name == "nx") nrm[0] = int(i);
                if (name == "ny") nrm[1] = int(i);
                if (name == "nz") nrm[2] = int(i);
            }
            if (isFace && element.properties[i].isList && (name == "vertex_indices" || name == "vertex_index"))
                faceList = int(i);
        }

        if (isVertex)
        {
            if (pos[0] < 0 || pos[1] < 0 || pos[2] < 0)
            {
                error = "PLY vertex element has no x/y/z";
                return false;
            }
            hasNormals = nrm[0] >= 0 && nrm[1] >= 0 && nrm[2] >= 0;
            model.vertices.resize(element.count);
        }
        if (isFace)
        {
            if (faceList < 0)
            {
                error = "PLY face element has no vertex_indices list";
                return false;
            }
            model.indices.reserve(element.count * 3);
        }

        if (format == PlyFormat::Ascii)
        {
            std::vector<double> values;
            for (size_t row = 0; row < element.count; ++row)
            {
                if (cursor >= end)
                {
                    error = "PLY ascii data truncated in element " + element.name;
                    return false;
                }
                values.clear();
                uint32_t face[256];
                uint32_t faceCount = 0;
                for (size_t p = 0; p < element.properties.size(); ++p)
                {
                    const PlyProperty& prop = element.properties[p];
                    char* next = nullptr;
                    if (prop.isList)
                    {
                        unsigned long count = std::strtoul(cursor, &next, 10);
                        cursor = next;
                        for (unsigned long k = 0; k < count; ++k)
                        {
                            unsigned long index = std::strtoul(cursor, &next, 10);
                            cursor = next;
                            if (int(p) == faceList && faceCount < 256)
                                face[faceCount++] = uint32_t(index);
                        }
                        values.push_back(0);
                    }
                    else
                    {
                        values.push_back(std::strtod(cursor, &next));
                        cursor = next;
                    }
                }
                // skip to end of line
                while (cursor < end && *cursor != '\n')
                    ++cursor;
                if (cursor < end)
                    ++cursor;

                if (isVertex)
                {
                    Vertex& v = model.vertices[row];
                    v.px = float(values[size_t(pos[0])]);
                    v.py = float(values[size_t(pos[1])]);
                    v.pz = float(values[size_t(pos[2])]);
                    if (hasNormals)
                    {
                        v.nx = float(values[size_t(nrm[0])]);
                        v.ny = float(values[size_t(nrm[1])]);
                        v.nz = float(values[size_t(nrm[2])]);
                    }
                }
                if (isFace)
                    for (uint32_t k = 2; k < faceCount; ++k)
                        model.indices.insert(model.indices.end(), {face[0], face[k - 1], face[k]});
            }
            continue;
        }

        // binary
        bool fixedSize = true;
        size_t rowSize = 0;
        std::vector<size_t> offsets;
        for (const PlyProperty& prop : element.properties)
        {
            offsets.push_back(rowSize);
            if (prop.isList)
                fixedSize = false;
            rowSize += plyTypeSize(prop.type);
        }

        if (fixedSize)
        {
            if (size_t(end - cursor) < rowSize * element.count)
            {
                error = "PLY binary data truncated in element " + element.name;
                return false;
            }
            if (isVertex)
            {
                const char* base = cursor;
                const auto& props = element.properties;
                parallelFor(element.count, [&](size_t begin, size_t endRow) {
                    for (size_t row = begin; row < endRow; ++row)
                    {
                        const char* r = base + row * rowSize;
                        Vertex& v = model.vertices[row];
                        v.px = float(readBinary(r + offsets[size_t(pos[0])], props[size_t(pos[0])].type, swap));
                        v.py = float(readBinary(r + offsets[size_t(pos[1])], props[size_t(pos[1])].type, swap));
                        v.pz = float(readBinary(r + offsets[size_t(pos[2])], props[size_t(pos[2])].type, swap));
                        if (hasNormals)
                        {
                            v.nx = float(readBinary(r + offsets[size_t(nrm[0])], props[size_t(nrm[0])].type, swap));
                            v.ny = float(readBinary(r + offsets[size_t(nrm[1])], props[size_t(nrm[1])].type, swap));
                            v.nz = float(readBinary(r + offsets[size_t(nrm[2])], props[size_t(nrm[2])].type, swap));
                        }
                    }
                });
            }
            cursor += rowSize * element.count;
            continue;
        }

        // variable-size rows (lists)
        for (size_t row = 0; row < element.count; ++row)
        {
            for (size_t p = 0; p < element.properties.size(); ++p)
            {
                const PlyProperty& prop = element.properties[p];
                if (!prop.isList)
                {
                    cursor += plyTypeSize(prop.type);
                    continue;
                }
                size_t countSize = plyTypeSize(prop.countType), itemSize = plyTypeSize(prop.type);
                if (size_t(end - cursor) < countSize)
                {
                    error = "PLY binary data truncated in element " + element.name;
                    return false;
                }
                uint32_t count = readBinaryIndex(cursor, prop.countType, swap);
                cursor += countSize;
                if (size_t(end - cursor) < itemSize * count)
                {
                    error = "PLY binary data truncated in element " + element.name;
                    return false;
                }
                if (int(p) == faceList && count >= 3)
                {
                    uint32_t first = readBinaryIndex(cursor, prop.type, swap);
                    uint32_t prev = readBinaryIndex(cursor + itemSize, prop.type, swap);
                    for (uint32_t k = 2; k < count; ++k)
                    {
                        uint32_t cur = readBinaryIndex(cursor + itemSize * k, prop.type, swap);
                        model.indices.insert(model.indices.end(), {first, prev, cur});
                        prev = cur;
                    }
                }
                cursor += itemSize * count;
            }
        }
    }

    if (model.vertices.empty() || model.indices.empty())
    {
        error = "PLY has no triangles (point clouds are not supported)";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// OBJ
// ---------------------------------------------------------------------------

bool loadObj(const std::string& path, Model& model, bool& hasNormals, std::string& error)
{
    fastObjMesh* obj = fast_obj_read(path.c_str());
    if (!obj)
    {
        error = "cannot read " + path;
        return false;
    }

    hasNormals = obj->normal_count > 1;

    std::vector<Vertex> corners;
    size_t indexOffset = 0;
    for (unsigned int f = 0; f < obj->face_count; ++f)
    {
        unsigned int faceVerts = obj->face_vertices[f];
        auto corner = [&](unsigned int k) {
            fastObjIndex idx = obj->indices[indexOffset + k];
            Vertex v = {};
            v.px = obj->positions[idx.p * 3 + 0];
            v.py = obj->positions[idx.p * 3 + 1];
            v.pz = obj->positions[idx.p * 3 + 2];
            if (hasNormals)
            {
                v.nx = obj->normals[idx.n * 3 + 0];
                v.ny = obj->normals[idx.n * 3 + 1];
                v.nz = obj->normals[idx.n * 3 + 2];
            }
            return v;
        };
        for (unsigned int k = 2; k < faceVerts; ++k)
        {
            corners.push_back(corner(0));
            corners.push_back(corner(k - 1));
            corners.push_back(corner(k));
        }
        indexOffset += faceVerts;
    }
    fast_obj_destroy(obj);

    if (corners.empty())
    {
        error = "OBJ has no faces";
        return false;
    }

    std::vector<unsigned int> remap(corners.size());
    size_t vertexCount = meshopt_generateVertexRemap(remap.data(), nullptr, corners.size(), corners.data(), corners.size(), sizeof(Vertex));
    model.vertices.resize(vertexCount);
    model.indices.resize(corners.size());
    meshopt_remapVertexBuffer(model.vertices.data(), corners.data(), corners.size(), sizeof(Vertex), remap.data());
    meshopt_remapIndexBuffer(model.indices.data(), nullptr, corners.size(), remap.data());
    return true;
}

void computeNormals(Model& model)
{
    for (Vertex& v : model.vertices)
        v.nx = v.ny = v.nz = 0;

    for (size_t i = 0; i < model.indices.size(); i += 3)
    {
        Vertex& a = model.vertices[model.indices[i + 0]];
        Vertex& b = model.vertices[model.indices[i + 1]];
        Vertex& c = model.vertices[model.indices[i + 2]];
        float e1[3] = {b.px - a.px, b.py - a.py, b.pz - a.pz};
        float e2[3] = {c.px - a.px, c.py - a.py, c.pz - a.pz};
        float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        for (Vertex* v : {&a, &b, &c})
        {
            v->nx += n[0];
            v->ny += n[1];
            v->nz += n[2];
        }
    }

    parallelFor(model.vertices.size(), [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            Vertex& v = model.vertices[i];
            float len = std::sqrt(v.nx * v.nx + v.ny * v.ny + v.nz * v.nz);
            if (len > 0)
            {
                v.nx /= len;
                v.ny /= len;
                v.nz /= len;
            }
            else
                v.nz = 1;
        }
    });
}

bool buildSingleLevel(const Model& model, const MeshletSettings& s, size_t maxVertices, size_t minTriangles, size_t maxTriangles,
    MeshletData& out, std::string& error);

// Builds a cluster LOD DAG with clusterlod.h; appends every cluster of every level as a meshlet.
bool buildLodClusters(const Model& model, const MeshletSettings& s, size_t maxVertices, size_t minTriangles, size_t maxTriangles,
    MeshletData& out, std::string& error)
{
    maxTriangles = std::clamp<size_t>(maxTriangles, 4, 256); // clodDefaultConfig limits
    clodConfig config = clodDefaultConfig(maxTriangles);
    config.max_vertices = maxVertices;
    config.min_triangles = std::min(minTriangles, maxTriangles);
    config.cluster_spatial = s.lodSpatial;
    config.cluster_split_factor = s.splitFactor;
    config.cluster_fill_weight = s.fillWeight;
    config.partition_size = size_t(std::max(2, s.lodPartitionSize));
    config.simplify_ratio = std::clamp(s.lodSimplifyRatio, 0.05f, 0.95f);
    config.optimize_clusters = s.optimize;

    // normals as simplification attributes (also required by permissive simplification)
    const float attributeWeights[3] = {0.5f, 0.5f, 0.5f};
    clodMesh mesh = {};
    mesh.indices = model.indices.data();
    mesh.index_count = model.indices.size();
    mesh.vertex_count = model.vertices.size();
    mesh.vertex_positions = &model.vertices[0].px;
    mesh.vertex_positions_stride = sizeof(Vertex);
    mesh.vertex_attributes = &model.vertices[0].nx;
    mesh.vertex_attributes_stride = sizeof(Vertex);
    mesh.attribute_weights = attributeWeights;
    mesh.attribute_count = 3;

    std::vector<clodGroup> groups;
    clodBuild(config, mesh, [&](clodGroup group, const clodCluster* clusters, size_t clusterCount) -> int {
        const int groupIndex = int(groups.size());
        groups.push_back(group);
        out.lodGroups.push_back({uint32_t(out.meshlets.size()), uint32_t(clusterCount)});

        for (size_t i = 0; i < clusterCount; ++i)
        {
            const clodCluster& cluster = clusters[i];

            meshopt_Meshlet m = {};
            m.vertex_offset = unsigned(out.vertices.size());
            m.triangle_offset = unsigned(out.triangles.size());
            m.triangle_count = unsigned(cluster.index_count / 3);
            out.vertices.resize(out.vertices.size() + cluster.vertex_count);
            out.triangles.resize(out.triangles.size() + cluster.index_count);
            m.vertex_count = unsigned(clodLocalIndices(&out.vertices[m.vertex_offset], &out.triangles[m.triangle_offset], cluster.indices,
                cluster.index_count));
            out.meshlets.push_back(m);

            MeshletLod lod = {};
            if (cluster.refined >= 0)
            {
                const clodBounds& self = groups[size_t(cluster.refined)].simplified;
                std::memcpy(lod.selfCenter, self.center, sizeof(lod.selfCenter));
                lod.selfRadius = self.radius;
                lod.selfError = self.error;
            }
            std::memcpy(lod.coarseCenter, group.simplified.center, sizeof(lod.coarseCenter));
            lod.coarseRadius = group.simplified.radius;
            lod.coarseError = group.simplified.error; // FLT_MAX for terminal groups
            lod.level = uint32_t(group.depth);
            out.lods.push_back(lod);
        }
        return groupIndex;
    });

    if (out.meshlets.empty())
    {
        error = "cluster LOD builder produced no clusters";
        return false;
    }
    out.triangles.resize((out.triangles.size() + 3) & ~size_t(3));

    // BVH over the groups for GPU LOD selection: a forest with one 8-wide tree per DAG level
    const size_t nodeWidth = 8;
    int maxDepth = 0;
    for (const clodGroup& group : groups)
        maxDepth = std::max(maxDepth, group.depth);
    const size_t levelCount = size_t(maxDepth) + 1;

    std::vector<clodNode> nodes(clodBuildHierarchyBound(groups.size(), nodeWidth, levelCount));
    nodes.resize(clodBuildHierarchy(nodes.data(), groups.data(), groups.size(), nodeWidth, levelCount));

    out.lodNodes.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        const clodNode& n = nodes[i];
        LodNode& g = out.lodNodes[i];
        std::memcpy(g.center, n.bounds.center, sizeof(g.center));
        g.radius = n.bounds.radius;
        g.error = n.bounds.error;
        g.group = n.group;
        g.childOffset = n.group < 0 ? n.child_offset : 0;
        g.childCount = n.group < 0 ? n.child_count : 0;
    }
    out.lodRootCount = uint32_t(levelCount);

    // count nodes per tree depth (breadth-first from the roots); the GPU traverses one depth per pass
    std::vector<uint32_t> frontier, next;
    for (uint32_t root = 0; root < out.lodRootCount; ++root)
        frontier.push_back(root);
    while (!frontier.empty())
    {
        out.lodNodesPerDepth.push_back(uint32_t(frontier.size()));
        next.clear();
        for (uint32_t index : frontier)
        {
            const LodNode& n = out.lodNodes[index];
            for (uint32_t k = 0; k < n.childCount; ++k)
                next.push_back(n.childOffset + k);
        }
        frontier.swap(next);
    }
    return true;
}

} // namespace

bool loadModel(const std::string& path, Model& model, std::string& error, bool optimizeVertexCache)
{
    Clock::time_point start = Clock::now();
    model = Model();
    model.path = path;

    bool hasNormals = false;
    std::string ext = lowerExtension(path);
    bool ok = false;
    if (ext == "ply")
        ok = loadPly(path, model, hasNormals, error);
    else if (ext == "obj")
        ok = loadObj(path, model, hasNormals, error);
    else
        error = "unsupported file extension (expected .ply, .obj or .mlod): " + path;
    if (!ok)
        return false;

    const uint32_t vertexCount = uint32_t(model.vertices.size());
    for (uint32_t index : model.indices)
        if (index >= vertexCount)
        {
            error = "index out of range: " + std::to_string(index) + " >= " + std::to_string(vertexCount);
            return false;
        }

    if (!hasNormals)
        computeNormals(model);

    if (optimizeVertexCache)
        meshopt_optimizeVertexCache(model.indices.data(), model.indices.data(), model.indices.size(), model.vertices.size());

    for (int k = 0; k < 3; ++k)
    {
        model.boundsMin[k] = INFINITY;
        model.boundsMax[k] = -INFINITY;
    }
    // bounds over vertices the triangles use (files can carry stray unreferenced points)
    std::vector<uint8_t> referenced(model.vertices.size(), 0);
    for (uint32_t index : model.indices)
        referenced[index] = 1;
    for (size_t i = 0; i < model.vertices.size(); ++i)
    {
        if (!referenced[i])
            continue;
        const Vertex& v = model.vertices[i];
        const float p[3] = {v.px, v.py, v.pz};
        for (int k = 0; k < 3; ++k)
        {
            model.boundsMin[k] = std::min(model.boundsMin[k], p[k]);
            model.boundsMax[k] = std::max(model.boundsMax[k], p[k]);
        }
    }
    if (model.indices.empty())
        for (int k = 0; k < 3; ++k)
            model.boundsMin[k] = model.boundsMax[k] = 0;

    // object center: area-weighted centroid of the triangles (falls back to the box center)
    {
        double sum[3] = {0, 0, 0}, totalArea = 0;
        for (size_t t = 0; t + 2 < model.indices.size(); t += 3)
        {
            const Vertex& a = model.vertices[model.indices[t + 0]];
            const Vertex& b = model.vertices[model.indices[t + 1]];
            const Vertex& c = model.vertices[model.indices[t + 2]];
            const double e1[3] = {double(b.px) - a.px, double(b.py) - a.py, double(b.pz) - a.pz};
            const double e2[3] = {double(c.px) - a.px, double(c.py) - a.py, double(c.pz) - a.pz};
            const double cx = e1[1] * e2[2] - e1[2] * e2[1];
            const double cy = e1[2] * e2[0] - e1[0] * e2[2];
            const double cz = e1[0] * e2[1] - e1[1] * e2[0];
            const double area = 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
            sum[0] += area * (double(a.px) + b.px + c.px) / 3.0;
            sum[1] += area * (double(a.py) + b.py + c.py) / 3.0;
            sum[2] += area * (double(a.pz) + b.pz + c.pz) / 3.0;
            totalArea += area;
        }
        for (int k = 0; k < 3; ++k)
            model.center[k] = totalArea > 0 ? float(sum[k] / totalArea) : (model.boundsMin[k] + model.boundsMax[k]) * 0.5f;

        double r2 = 0;
        for (size_t i = 0; i < model.vertices.size(); ++i)
        {
            if (!referenced[i])
                continue;
            const Vertex& v = model.vertices[i];
            const double d[3] = {double(v.px) - model.center[0], double(v.py) - model.center[1], double(v.pz) - model.center[2]};
            r2 = std::max(r2, d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        }
        model.radius = float(std::sqrt(r2));
    }

    // average edge length over (up to) ~1M evenly sampled triangles
    {
        const size_t triangleCount = model.indices.size() / 3;
        const size_t step = std::max<size_t>(1, triangleCount / (1u << 20));
        double total = 0;
        size_t edges = 0;
        for (size_t t = 0; t < triangleCount; t += step)
            for (int k = 0; k < 3; ++k)
            {
                const Vertex& a = model.vertices[model.indices[t * 3 + size_t(k)]];
                const Vertex& b = model.vertices[model.indices[t * 3 + size_t((k + 1) % 3)]];
                float dx = a.px - b.px, dy = a.py - b.py, dz = a.pz - b.pz;
                total += std::sqrt(double(dx * dx + dy * dy + dz * dz));
                edges++;
            }
        model.avgEdgeLength = edges ? float(total / double(edges)) : 0.f;
    }

    model.loadSeconds = secondsSince(start);
    return true;
}

bool buildMeshlets(const Model& model, const MeshletSettings& s, MeshletData& out, std::string& error)
{
    Clock::time_point start = Clock::now();
    out = MeshletData();

    const size_t maxVertices = size_t(std::clamp(s.maxVertices, 3, 256));
    const size_t maxTriangles = size_t(std::clamp(s.maxTriangles, 1, 512));
    const size_t minTriangles = size_t(std::clamp(s.minTriangles, 1, int(maxTriangles)));
    const size_t indexCount = model.indices.size();
    const size_t vertexCount = model.vertices.size();
    const float* positions = &model.vertices[0].px;
    const size_t stride = sizeof(Vertex);

    if (indexCount == 0)
    {
        error = "model has no triangles";
        return false;
    }

    if (s.lod)
    {
        if (!buildLodClusters(model, s, maxVertices, minTriangles, maxTriangles, out, error))
            return false;
    }
    else if (!buildSingleLevel(model, s, maxVertices, minTriangles, maxTriangles, out, error))
        return false;

    const size_t count = out.meshlets.size();
    const bool optimizeHere = s.optimize && !s.lod; // clodBuild already optimized its clusters

    out.bounds.resize(count);
    out.edgeMasks.assign(((out.triangles.size() + 2) / 3 + 3) & ~size_t(3), 0);
    parallelFor(count, [&](size_t begin, size_t end) {
        std::vector<uint32_t> edges; // (min << 8 | max) local vertex pairs
        for (size_t i = begin; i < end; ++i)
        {
            const meshopt_Meshlet& m = out.meshlets[i];
            if (optimizeHere)
                meshopt_optimizeMeshlet(&out.vertices[m.vertex_offset], &out.triangles[m.triangle_offset], m.triangle_count, m.vertex_count);

            // perimeter edges: local edges used by exactly one triangle of this meshlet
            const uint8_t* tri = &out.triangles[m.triangle_offset];
            edges.clear();
            for (uint32_t t = 0; t < m.triangle_count * 3; t += 3)
                for (uint32_t k = 0; k < 3; ++k)
                {
                    uint32_t a = tri[t + (k + 1) % 3], b = tri[t + (k + 2) % 3];
                    edges.push_back(std::min(a, b) << 8 | std::max(a, b));
                }
            std::vector<uint32_t> sorted = edges;
            std::sort(sorted.begin(), sorted.end());
            for (uint32_t t = 0; t < m.triangle_count; ++t)
            {
                uint8_t mask = 0;
                for (uint32_t k = 0; k < 3; ++k)
                {
                    auto range = std::equal_range(sorted.begin(), sorted.end(), edges[t * 3 + k]);
                    if (range.second - range.first == 1)
                        mask |= uint8_t(1u << k); // edge opposite local vertex k
                }
                out.edgeMasks[m.triangle_offset / 3 + t] = mask;
            }

            meshopt_Bounds b = meshopt_computeMeshletBounds(&out.vertices[m.vertex_offset], &out.triangles[m.triangle_offset],
                m.triangle_count, positions, vertexCount, stride);
            MeshletBounds& gb = out.bounds[i];
            std::memcpy(gb.center, b.center, sizeof(gb.center));
            gb.radius = b.radius;
            std::memcpy(gb.coneAxis, b.cone_axis, sizeof(gb.coneAxis));
            gb.coneCutoff = b.cone_cutoff;
        }
    });

    // statistics: sizes and cones over level 0 (comparable with and without LOD), counts per level
    size_t baseClusters = 0, baseVertices = 0;
    double coneAngles = 0;
    for (size_t i = 0; i < count; ++i)
    {
        const meshopt_Meshlet& m = out.meshlets[i];
        const MeshletLod& lod = out.lods[i];
        out.triangleCount += m.triangle_count;
        if (lod.level >= out.levelClusters.size())
        {
            out.levelClusters.resize(lod.level + 1);
            out.levelTriangles.resize(lod.level + 1);
        }
        out.levelClusters[lod.level]++;
        out.levelTriangles[lod.level] += m.triangle_count;
        if (lod.coarseError == FLT_MAX)
            out.coarsestTriangles += m.triangle_count;
        if (lod.level != 0)
            continue;

        baseClusters++;
        baseVertices += m.vertex_count;
        out.baseTriangleCount += m.triangle_count;
        float cutoff = out.bounds[i].coneCutoff;
        if (cutoff < 1.f)
        {
            out.coneCullable++;
            coneAngles += 2.0 * std::asin(double(std::clamp(cutoff, 0.f, 1.f))) * 180.0 / M_PI; // cutoff = sin(half-angle)
        }
    }
    out.lod = s.lod;
    out.avgVertices = float(double(baseVertices) / double(baseClusters));
    out.avgTriangles = float(double(out.baseTriangleCount) / double(baseClusters));
    out.vertexFill = out.avgVertices / float(maxVertices);
    out.triangleFill = out.avgTriangles / float(maxTriangles);
    out.avgConeAngleDeg = out.coneCullable ? float(coneAngles / double(out.coneCullable)) : 0.f;
    out.coneCullableFraction = float(double(out.coneCullable) / double(baseClusters));
    out.buildSeconds = secondsSince(start);
    return true;
}

namespace
{

// Single meshlet level with one of the meshopt_buildMeshlets* builders.
bool buildSingleLevel(const Model& model, const MeshletSettings& s, size_t maxVertices, size_t minTriangles, size_t maxTriangles,
    MeshletData& out, std::string& error)
{
    const size_t indexCount = model.indices.size();
    const size_t vertexCount = model.vertices.size();
    const float* positions = &model.vertices[0].px;
    const size_t stride = sizeof(Vertex);

    const bool usesMin = s.algorithm == MeshletAlgorithm::Flex || s.algorithm == MeshletAlgorithm::Spatial;
    size_t bound = meshopt_buildMeshletsBound(indexCount, maxVertices, usesMin ? minTriangles : maxTriangles);

    out.meshlets.resize(bound);
    out.vertices.resize(indexCount);
    out.triangles.resize(indexCount);

    size_t count = 0;
    switch (s.algorithm)
    {
    case MeshletAlgorithm::Greedy:
        count = meshopt_buildMeshlets(out.meshlets.data(), out.vertices.data(), out.triangles.data(), model.indices.data(), indexCount,
            positions, vertexCount, stride, maxVertices, maxTriangles, s.coneWeight);
        break;
    case MeshletAlgorithm::Flex:
        count = meshopt_buildMeshletsFlex(out.meshlets.data(), out.vertices.data(), out.triangles.data(), model.indices.data(), indexCount,
            positions, vertexCount, stride, maxVertices, minTriangles, maxTriangles, s.coneWeight, s.splitFactor);
        break;
    case MeshletAlgorithm::Spatial:
        count = meshopt_buildMeshletsSpatial(out.meshlets.data(), out.vertices.data(), out.triangles.data(), model.indices.data(), indexCount,
            positions, vertexCount, stride, maxVertices, minTriangles, maxTriangles, s.fillWeight);
        break;
    case MeshletAlgorithm::Scan:
        count = meshopt_buildMeshletsScan(out.meshlets.data(), out.vertices.data(), out.triangles.data(), model.indices.data(), indexCount,
            vertexCount, maxVertices, maxTriangles);
        break;
    }

    if (count == 0)
    {
        error = "meshlet builder produced no meshlets";
        return false;
    }

    out.meshlets.resize(count);
    const meshopt_Meshlet& last = out.meshlets.back();
    out.vertices.resize(last.vertex_offset + last.vertex_count);
    out.triangles.resize((last.triangle_offset + last.triangle_count * 3 + 3) & ~size_t(3));

    MeshletLod always = {}; // level 0, nothing coarser: always selected
    always.coarseError = FLT_MAX;
    out.lods.assign(count, always);
    return true;
}

} // namespace
