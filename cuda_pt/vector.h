#pragma once
#include "common.h"
#include <cmath>

struct Vec3
{
    float x, y, z;
    DEVICE HOST Vec3() : Vec3(0, 0, 0) {}
    DEVICE HOST Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    DEVICE HOST Vec3(const Vec3&) = default;
    DEVICE HOST Vec3& operator =(const Vec3 &v) {
        x = v.x; y = v.y; z = v.z;
        return *this;
    }
    DEVICE HOST Vec3 operator + (const Vec3 &v) const {
        return{ x + v.x, y + v.y, z + v.z };
    }
    DEVICE HOST Vec3 operator - (const Vec3 &v) const {
        return{ x - v.x, y - v.y, z - v.z };
    }
    DEVICE HOST Vec3 operator * (float r) const {
        return{ x * r, y * r, z * r };
    }
    DEVICE HOST Vec3 operator / (float r) const {
        return{ x / r, y / r, z / r };
    }

    DEVICE HOST float dot(const Vec3 &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    DEVICE HOST float sqNorm() const {
        return this->dot(*this);
    }

    DEVICE HOST float norm() const {
        return std::sqrt(sqNorm());
    }


    DEVICE HOST Vec3 cross(const Vec3 &v) const {
        return{
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        };
    }

    DEVICE HOST Vec3 normalized() const {
        return *this / norm();
    }
    
};

inline DEVICE HOST Vec3 operator*(float r, const Vec3 &v) {
    return v * r;
}