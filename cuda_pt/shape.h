#pragma once
#include "common.h"
#include "vector.h"
#include <algorithm>
#include <cmath>
struct BBox
{
    Vec3 mins, maxs;
    DEVICE HOST BBox() = default;
    DEVICE HOST BBox(const Vec3 mins, const Vec3 maxs) : mins(mins), maxs(maxs) {}
    DEVICE HOST BBox(const BBox &) = default;
    DEVICE HOST BBox &operator=(const BBox &bbox) {
        mins = bbox.mins;
        maxs = bbox.maxs;
        return *this;
    }
    
    BBox Union(const BBox &another) const {
        BBox box;
        box.mins.x = std::min(mins.x, another.mins.x);
        box.mins.y = std::min(mins.y, another.mins.y);
        box.mins.z = std::min(mins.z, another.mins.z);

        box.maxs.x = std::max(maxs.x, another.maxs.x);
        box.maxs.y = std::max(maxs.y, another.maxs.y);
        box.maxs.z = std::max(maxs.z, another.maxs.z);
        return box;
    }
    BBox operator +(const BBox &another) const {
        return Union(another);
    }
};

struct Ray {
    Vec3 start;
    Vec3 n_dir;

    DEVICE HOST Ray() : start(0, 0, 0), n_dir(1, 0, 0) {}
    DEVICE HOST Ray(const Vec3 &start, const Vec3 &n_dir) : start(start), n_dir(n_dir) {}
    DEVICE HOST static Ray FromTo(const Vec3 &from, const Vec3 &to) {
        return Ray(from, (to - from).normalized());
    }
    DEVICE HOST Vec3 atDistance(float d) const
    {
        return start + n_dir * d;
    }
};

enum class ShapeType {
    Sphere,
    Triangle,
    NonType
};


struct Sphere {
    static const ShapeType type = ShapeType::Sphere;
    Vec3 center;
    float radius;

    DEVICE HOST Sphere() = default;
    DEVICE HOST Sphere(const Vec3 &center, float radius) : center(center), radius(radius) {}
    DEVICE HOST Sphere(const Sphere &) = default;
    DEVICE HOST Sphere &operator=(const Sphere &s) {
        center = s.center;
        radius = s.radius;
        return *this;
    }

    DEVICE HOST float firstIntersection(const Ray &ray) const {
        Vec3 H = ray.start + ray.n_dir.dot(center - ray.start) * ray.n_dir;
        //レイの直線と交点の距離の二乗
        float sqr_d = (H - center).sqNorm();
        if (sqr_d > radius * radius) {
            return -1;
        }

        //Hと「球と直線と交点」との距離
        float L = std::sqrt(radius * radius - sqr_d);

        //レイ原点からHまでの(符号付き)距離
        float OH = ray.n_dir.dot(H - ray.start);

        if (OH + L < 0) {
            return{};
        }
        float distance = (OH - L > 0) ? (OH - L) : (OH + L);
        return distance;
    }

    DEVICE HOST Vec3 getNormal(const Vec3 &pos) const {
        return (pos - center).normalized();
    }

    BBox getBBox() const {
        Vec3 diag(radius, radius, radius);
        return { center - diag, center + diag };
    }
    
};


struct Triangle {
    static const ShapeType type = ShapeType::Triangle;
private:
    Vec3 a, b, c;
    Vec3 normal;
public:
    DEVICE HOST Triangle() = default;
    DEVICE HOST Triangle(const Vec3 &a, const Vec3 &b, const Vec3 &c) : a(a), b(b), c(c) {
        normal = (a - b).cross(a - c).normalized();
    }
    DEVICE HOST Triangle(const Triangle &) = default;
    DEVICE HOST Triangle &operator=(const Triangle &t) {
        a = t.a;
        b = t.b;
        c = t.c;
        normal = t.normal;
        return *this;
    }

    DEVICE HOST float firstIntersection(const Ray &ray) const {
        float distance_h = normal.dot(a - ray.start);
        float rate = normal.dot(ray.n_dir);
        if (std::abs(rate) < 1e-6) {
            return -1;
        }
        float distance = distance_h / rate;
        if (distance > 0) {
            Vec3 hitpoint = ray.atDistance(distance);
            Vec3 c1 = (b - a).cross(hitpoint - a);
            Vec3 c2 = (c - b).cross(hitpoint - b);
            Vec3 c3 = (a - c).cross(hitpoint - c);
            if (c1.dot(c2) > 0 && c2.dot(c3) > 0) {
                return distance;
            }
            else {
                return -1;
            }
        }
        return distance;
    }

    DEVICE HOST Vec3 getNormal(const Vec3 &) const {
        return normal;
    }

    BBox getBBox() const {
        Vec3 mins, maxs;
        mins.x = std::min(a.x, std::min(b.x, c.x));
        mins.y = std::min(a.y, std::min(b.y, c.y));
        mins.z = std::min(a.z, std::min(b.z, c.z));
        maxs.x = std::max(a.x, std::max(b.x, c.x));
        maxs.y = std::max(a.y, std::max(b.y, c.y));
        maxs.z = std::max(a.z, std::max(b.z, c.z));

        return{ mins, maxs };
    }
};


struct Shape {
    DEVICE HOST Shape() : shape{0} {};
    DEVICE HOST Shape(const Shape &) = default;
    DEVICE HOST Shape(const Sphere &sphere) : type(ShapeType::Sphere), shape{0}  
    {
        shape.sphere = sphere;
    }
    DEVICE HOST Shape(const Triangle &triangle) : type(ShapeType::Triangle), shape{0}
    { 
        shape.triangle = triangle;
    }

    ShapeType type = ShapeType::NonType;
    union {
        int dummy;
        Sphere sphere;
        Triangle triangle;
    } shape;

    DEVICE HOST float firstIntersecttion(const Ray &ray) const {
        switch (type) {
        case ShapeType::Sphere:
            return shape.sphere.firstIntersection(ray);
        case ShapeType::Triangle:
            return shape.triangle.firstIntersection(ray);
        case ShapeType::NonType:
        default:
            return -1;
        }
    }

    DEVICE HOST Vec3 getNormal(const Vec3 &pos) const {
        switch (type) {
        case ShapeType::Sphere:
            return shape.sphere.getNormal(pos);
        case ShapeType::Triangle:
            return shape.triangle.getNormal(pos);
        case ShapeType::NonType:
        default:
            return{ 1, 0, 0 };
        }
    }
};

struct Optical {
    Vec3 reflection = { 0, 0, 0 };
    Vec3 emission = { 0, 0, 0 };
    DEVICE HOST static Optical Emission(float x, float y, float z) {
        Optical o;
        o.reflection = { 0, 0, 0 };
        o.emission = { x, y, z };
        return o;
    }
    DEVICE HOST static Optical Reflection(float x, float y, float z) {
        Optical o;
        o.reflection = { x, y, z };
        o.emission = { 0, 0, 0 };
        return o;
    }
};

struct Object {
    Object(const Shape &shape, const Optical &optical) : shape(shape), optical(optical) {}
    Object() = default;
    Object(const Object &object) = default;
    Shape shape;
    Optical optical;
};