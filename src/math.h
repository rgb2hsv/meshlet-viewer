#pragma once

#include <cmath>

struct vec3
{
    float x = 0, y = 0, z = 0;
};

inline vec3 operator+(vec3 a, vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline vec3 operator-(vec3 a, vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline vec3 operator*(vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline vec3 cross(vec3 a, vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(vec3 a) { return std::sqrt(dot(a, a)); }
inline vec3 normalize(vec3 a)
{
    float len = length(a);
    return len > 0 ? a * (1.f / len) : a;
}

// Column-major 4x4 matrix (matches GLSL): m[col * 4 + row]
struct mat4
{
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    float& at(int row, int col) { return m[col * 4 + row]; }
    float at(int row, int col) const { return m[col * 4 + row]; }
};

inline mat4 operator*(const mat4& a, const mat4& b)
{
    mat4 r;
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
        {
            float sum = 0;
            for (int k = 0; k < 4; ++k)
                sum += a.at(row, k) * b.at(k, col);
            r.at(row, col) = sum;
        }
    return r;
}

// Right-handed view matrix
inline mat4 lookAt(vec3 eye, vec3 target, vec3 up)
{
    vec3 f = normalize(target - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    mat4 r;
    r.at(0, 0) = s.x, r.at(0, 1) = s.y, r.at(0, 2) = s.z, r.at(0, 3) = -dot(s, eye);
    r.at(1, 0) = u.x, r.at(1, 1) = u.y, r.at(1, 2) = u.z, r.at(1, 3) = -dot(u, eye);
    r.at(2, 0) = -f.x, r.at(2, 1) = -f.y, r.at(2, 2) = -f.z, r.at(2, 3) = dot(f, eye);
    return r;
}

// Vulkan clip space (y down), reversed-Z with an infinite far plane: near -> 1, infinity -> 0
inline mat4 perspectiveReverseZ(float fovY, float aspect, float zNear)
{
    float f = 1.f / std::tan(fovY * 0.5f);
    mat4 r;
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 0;
    r.at(0, 0) = f / aspect;
    r.at(1, 1) = -f;
    r.at(2, 3) = zNear;
    r.at(3, 2) = -1.f;
    return r;
}
