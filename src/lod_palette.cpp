#include "lod_palette.h"

#include <algorithm>
#include <cmath>

namespace lod_palette
{
namespace
{

constexpr float kHueStart = 100.f; // degrees in LCh(ab): yellow
constexpr float kHueEnd = 400.f;   // past blue and magenta round to red (40 deg)
constexpr float kMaxChroma = 55.f;

// CIELAB (D65 white) -> linear sRGB; components may fall outside [0, 1] (out of gamut)
std::array<float, 3> labToLinearSrgb(float L, float a, float b)
{
    auto finv = [](float t) {
        constexpr float d = 6.f / 29.f;
        return t > d ? t * t * t : 3.f * d * d * (t - 4.f / 29.f);
    };
    const float fy = (L + 16.f) / 116.f;
    const float X = 0.95047f * finv(fy + a / 500.f);
    const float Y = finv(fy);
    const float Z = 1.08883f * finv(fy - b / 200.f);
    return {3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z, -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z,
        0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z};
}

bool inGamut(const std::array<float, 3>& rgb)
{
    for (float c : rgb)
        if (c < -1e-4f || c > 1.f + 1e-4f)
            return false;
    return true;
}

float encodeSrgb(float c)
{
    c = std::clamp(c, 0.f, 1.f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}

} // namespace

std::array<float, 3> levelColor(uint32_t level, uint32_t levelCount)
{
    const float t = levelCount > 1 ? float(std::min(level, levelCount - 1)) / float(levelCount - 1) : 0.f;
    const float L = kLightest + (kDarkest - kLightest) * t;
    const float hue = (kHueStart + (kHueEnd - kHueStart) * t) * float(M_PI / 180.0);

    // largest chroma (up to the cap) at this lightness and hue that sRGB can show: L* is kept
    // exactly, so the even luminance steps survive gamut mapping
    float lo = 0.f, hi = kMaxChroma;
    if (!inGamut(labToLinearSrgb(L, hi * std::cos(hue), hi * std::sin(hue))))
        for (int i = 0; i < 24; ++i)
        {
            const float mid = 0.5f * (lo + hi);
            (inGamut(labToLinearSrgb(L, mid * std::cos(hue), mid * std::sin(hue))) ? lo : hi) = mid;
        }
    else
        lo = hi;
    const std::array<float, 3> rgb = labToLinearSrgb(L, lo * std::cos(hue), lo * std::sin(hue));
    return {encodeSrgb(rgb[0]), encodeSrgb(rgb[1]), encodeSrgb(rgb[2])};
}

} // namespace lod_palette
