#include "meshlet_optimizer.h"

#include "meshlet_file.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#if defined(_WIN32)
#define popen _popen
#define pclose _pclose
#endif

namespace
{

// Quotes one argument for the shell popen() runs (cmd.exe on Windows, sh elsewhere)
std::string quoteArg(const std::string& arg)
{
#if defined(_WIN32)
    return "\"" + arg + "\""; // paths cannot contain '"' on Windows
#else
    std::string quoted = "'";
    for (char c : arg)
        quoted += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return quoted + "'";
#endif
}

// Binary little-endian PLY of the referenced triangles, vertices welded by position (OBJ corners,
// or the coarser levels' vertices of a mesh loaded from a .mlod, are not part of the surface).
// Texture coordinates go per face corner ("texcoord" lists, as MeshLab writes them), so texture
// seams survive the welding.
bool writeSurfacePly(const std::filesystem::path& file, const Model& model, std::string& error)
{
    const size_t indexCount = model.indices.size();
    const bool textured = !model.texcoords.empty() && model.texcoords.size() == model.vertices.size();
    std::vector<unsigned int> remap(model.vertices.size());
    const meshopt_Stream position = {&model.vertices[0].px, sizeof(float) * 3, sizeof(Vertex)};
    const size_t vertexCount = meshopt_generateVertexRemapMulti(
        remap.data(), model.indices.data(), indexCount, model.vertices.size(), &position, 1);

    std::vector<float> positions(vertexCount * 3);
    for (size_t v = 0; v < model.vertices.size(); ++v)
        if (remap[v] != ~0u)
        {
            positions[remap[v] * 3 + 0] = model.vertices[v].px;
            positions[remap[v] * 3 + 1] = model.vertices[v].py;
            positions[remap[v] * 3 + 2] = model.vertices[v].pz;
        }

    std::ofstream out(file, std::ios::binary);
    if (!out)
    {
        error = "cannot write " + file.string();
        return false;
    }
    out << "ply\nformat binary_little_endian 1.0\n"
        << "element vertex " << vertexCount << "\nproperty float x\nproperty float y\nproperty float z\n"
        << "element face " << indexCount / 3 << "\nproperty list uchar int vertex_indices\n"
        << (textured ? "property list uchar float texcoord\n" : "") << "end_header\n";
    out.write(reinterpret_cast<const char*>(positions.data()), std::streamsize(positions.size() * sizeof(float)));
    const size_t faceSize = textured ? 13 + 25 : 13; // + count byte and six floats
    std::vector<char> faces(indexCount / 3 * faceSize);
    for (size_t t = 0; t < indexCount / 3; ++t)
    {
        char* f = &faces[t * faceSize];
        f[0] = 3;
        for (int k = 0; k < 3; ++k)
        {
            const int32_t index = int32_t(remap[model.indices[t * 3 + size_t(k)]]);
            std::memcpy(f + 1 + 4 * k, &index, 4);
        }
        if (!textured)
            continue;
        f[13] = 6;
        for (int k = 0; k < 3; ++k)
        {
            const Texcoord& uv = model.texcoords[model.indices[t * 3 + size_t(k)]];
            std::memcpy(f + 14 + 8 * k, &uv.u, 4);
            std::memcpy(f + 18 + 8 * k, &uv.v, 4);
        }
    }
    out.write(faces.data(), std::streamsize(faces.size()));
    if (!out)
    {
        error = "cannot write " + file.string();
        return false;
    }
    return true;
}

// Removes a temporary file when it goes out of scope
struct TempFile
{
    std::filesystem::path path;
    ~TempFile()
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

std::filesystem::path tempPath(const char* extension)
{
    static std::atomic<unsigned> counter{0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string name = "meshlet-viewer-" + std::to_string(stamp) + "-" + std::to_string(counter++) + extension;
    return std::filesystem::temp_directory_path() / name;
}

} // namespace

std::string defaultMeshletOptimizerPath()
{
#ifdef MESHLET_OPTIMIZER_EXE
    return MESHLET_OPTIMIZER_EXE;
#else
    return std::string();
#endif
}

std::string meshletOptimizerArgs(const MeshletOptimizerParams& p)
{
    const MeshletOptimizerParams d;
    std::string args;
    auto add = [&](const std::string& arg) { args += (args.empty() ? "" : " ") + arg; };
    auto value = [&](const char* name, auto v, auto def) {
        if (v != def)
            add(std::string(name) + " " + std::to_string(v));
    };
    auto flag = [&](const char* name, bool on) {
        if (!on)
            add(name);
    };
    value("--chart-triangles", p.chartTriangles, d.chartTriangles);
    value("--cvt-seed", p.cvtSeed, d.cvtSeed);
    flag("--no-regularize", p.regularize);
    value("--max-clique", p.maxClique, d.maxClique);
    value("--stop-at-regions", p.stopAtRegions, d.stopAtRegions);
    value("--max-depth", p.maxDepth, d.maxDepth);
    value("--seed", p.seed, d.seed);
    flag("--no-keep-disks", p.keepDisks);
    flag("--no-straighten-seams", p.straightenSeams);
    if (p.interiorFraction != d.interiorFraction)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "--interior-fraction %.4g", double(p.interiorFraction));
        add(buf);
    }
    value("--split-above-faces", p.splitAboveFaces, d.splitAboveFaces);
    flag("--no-project", p.project);
    flag("--no-simplify-borders", p.simplifyBorders);
    flag("--no-relax-borders", p.relaxBorders);
    value("-j", p.threads, d.threads);
    return args;
}

bool runMeshletOptimizer(const std::string& exe, const Model& model, const MeshletOptimizerParams& params, Model& outModel,
    MeshletSettings& outSettings, MeshletData& outData, std::string& error, const std::function<void(const std::string&)>& progress)
{
    const auto start = std::chrono::steady_clock::now();
    std::error_code ec;
    if (exe.empty() || !std::filesystem::exists(exe, ec))
    {
        error = exe.empty() ? "meshlet-optimizer was not built with this viewer (external/meshlet-optimizer missing?)"
                            : "meshlet-optimizer not found: " + exe;
        return false;
    }
    if (model.indices.empty())
    {
        error = "meshlet-optimizer: the model has no triangles";
        return false;
    }

    TempFile input{tempPath(".ply")}, output{tempPath(kMeshletFileExtension)};
    if (!writeSurfacePly(input.path, model, error))
        return false;

    std::string command = quoteArg(exe) + " " + quoteArg(input.path.string()) + " -o " + quoteArg(output.path.string());
    const std::string args = meshletOptimizerArgs(params);
    if (!args.empty())
        command += " " + args;
    std::printf("meshlet-optimizer %s\n", args.c_str());
    std::fflush(stdout);
    command += " 2>&1";
#if defined(_WIN32)
    command = "\"" + command + "\""; // cmd /c strips the outer quotes
#endif

    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe)
    {
        error = "cannot run " + exe;
        return false;
    }
    std::string lastLine, line;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe))
    {
        line += buf;
        if (!line.empty() && line.back() == '\n')
        {
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
                line.pop_back();
            std::printf("  %s\n", line.c_str());
            std::fflush(stdout);
            if (progress)
                progress(line);
            lastLine = line; // the last line, used for errors
            line.clear();
        }
    }
    const int status = pclose(pipe);
    if (status != 0)
    {
        error = "meshlet-optimizer failed (exit " + std::to_string(status) + ")" + (lastLine.empty() ? "" : ": " + lastLine);
        return false;
    }

    if (!loadMeshletFile(output.path.string(), outModel, outSettings, outData, error))
    {
        error = "meshlet-optimizer output: " + error;
        return false;
    }
    outModel.path = model.path;
    outData.buildSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return true;
}
