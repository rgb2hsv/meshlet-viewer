#pragma once

#include <cstdint>
#include <string>
#include <vector>

// A matcap ("material capture") image from Blender's datafiles/studiolights/matcap
// (CC0). The PNGs in assets/matcap were converted from Blender's EXRs as
// diffuse + specular (the look of a white material), sRGB-encoded, 8 bits per channel.
struct Matcap
{
    std::string name; // file stem, e.g. "clay_studio"
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels; // RGBA8, sRGB-encoded, top row first
};

// Finds assets/matcap (working directory first, then the source tree) and loads every
// .png in it, sorted by name. Blender's internal brush preview matcap is skipped.
std::vector<Matcap> loadMatcaps(std::string& error);
