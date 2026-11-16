#pragma once

// Vector math uses Eigen (external/eigen). Eigen matrices are column-major by default,
// matching GLSL, so a Matrix4f can be copied to the GPU as is (matrix.data()).
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

// Right-handed view matrix
inline Eigen::Matrix4f lookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& target, const Eigen::Vector3f& up)
{
    const Eigen::Vector3f f = (target - eye).normalized();
    const Eigen::Vector3f s = f.cross(up).normalized();
    const Eigen::Vector3f u = s.cross(f);

    Eigen::Matrix4f r = Eigen::Matrix4f::Identity();
    r.block<1, 3>(0, 0) = s.transpose();
    r.block<1, 3>(1, 0) = u.transpose();
    r.block<1, 3>(2, 0) = -f.transpose();
    r(0, 3) = -s.dot(eye);
    r(1, 3) = -u.dot(eye);
    r(2, 3) = f.dot(eye);
    return r;
}

// Vulkan clip space (y down), reversed-Z with an infinite far plane: near -> 1, infinity -> 0
inline Eigen::Matrix4f perspectiveReverseZ(float fovY, float aspect, float zNear)
{
    const float f = 1.f / std::tan(fovY * 0.5f);
    Eigen::Matrix4f r = Eigen::Matrix4f::Zero();
    r(0, 0) = f / aspect;
    r(1, 1) = -f;
    r(2, 3) = zNear;
    r(3, 2) = -1.f;
    return r;
}
