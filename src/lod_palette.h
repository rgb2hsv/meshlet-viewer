#pragma once

#include <array>
#include <cstdint>

// Colors of the LOD levels for the "Meshlet (LOD gradient)" mode, designed in CIELAB (D65):
// lightness L* falls in equal steps from the finest level (lightest) to the coarsest, so every
// level is equally far from its neighbours in perceived luminance; hue turns a full 300 degrees from yellow through
// green, cyan, blue and magenta to red, and chroma is the largest that stays inside sRGB (capped, so the
// colors stay soft). L* stays well above black so the black wireframe and outlines read on every level.
namespace lod_palette
{
constexpr float kLightest = 92.f; // L* of level 0
constexpr float kDarkest = 62.f;  // L* of the coarsest level

// sRGB-encoded color in [0, 1] of level `level` out of `levelCount`
std::array<float, 3> levelColor(uint32_t level, uint32_t levelCount);
} // namespace lod_palette
