#include "matcap.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<Matcap> loadMatcaps(std::string& error)
{
    std::vector<Matcap> result;

    std::string directory;
    for (const std::string& candidate : {std::string("assets/matcap"), std::string(MESHLETS_ASSET_DIR) + "/matcap"})
        if (fs::is_directory(candidate))
        {
            directory = candidate;
            break;
        }
    if (directory.empty())
    {
        error = "assets/matcap not found";
        return result;
    }

    std::vector<fs::path> files;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory))
        if (entry.is_regular_file() && entry.path().extension() == ".png" && entry.path().stem() != "brush_thumbnail_preview")
            files.push_back(entry.path());
    std::sort(files.begin(), files.end());

    for (const fs::path& file : files)
    {
        int width = 0, height = 0, channels = 0;
        stbi_uc* data = stbi_load(file.string().c_str(), &width, &height, &channels, 4);
        if (!data)
        {
            error += (error.empty() ? "" : "\n") + file.string() + ": " + stbi_failure_reason();
            continue;
        }
        Matcap matcap;
        matcap.name = file.stem().string();
        matcap.width = width;
        matcap.height = height;
        matcap.pixels.assign(data, data + size_t(width) * size_t(height) * 4);
        stbi_image_free(data);
        result.push_back(std::move(matcap));
    }
    return result;
}
