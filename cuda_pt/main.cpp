#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include "common.h"
#include "vector.h"
#include "shape.h"
#include "kernel.h"
#include <cmath>
#include <algorithm>

#include <random>

int main()
{
    Shape shape;
    shape.type = ShapeType::Sphere;
    shape.shape.sphere = Sphere({ 0, 0, 0 }, 100);
    std::cout << shape.shape.sphere.radius << std::endl;

    RenderSettings rs;
    rs.viewfrom = { 0, 0, 100 };
    rs.viewat = { 0, 0, 0 };
    rs.upvec = { 0, 1, 0 };
    rs.screenwidth = 100;
    rs.WIDTH = 1000;
    rs.HEIGHT = 1000;
    rs.PATH = 1000;
    std::vector<Object> objs;
    
    //objs.emplace_back(Shape(Triangle({ -10, 0, 0 }, { 10, 10, 0 }, { 10, -10, 0 })), Optical::Emission(0, 80, 0.5));
    //objs.emplace_back(Shape(Sphere({ -20, -20, 0 }, 19.0f)), Optical::Reflection(0.8, 0.8, 0));
    //objs.emplace_back(Shape(Sphere({ -20, 20, 0 }, 19.0f)), Optical::Reflection(0, 0.8, 0.8));
    //objs.emplace_back(Shape(Sphere({ 20, -20, 0 }, 19.0f)), Optical::Reflection(0.8, 0, 0.8));
    //objs.emplace_back(Shape(Sphere({ 20, 20, 0 }, 20.0f)), Optical::Emission(10, 10, 10));
    objs.emplace_back(Shape(Sphere({0, 0, 0}, 1e6f)), Optical::Emission(0.01, 0.01, 0.01));

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for (int i = 0; i < 50; i++) {
        float x = -30 + 60 * u01(mt);
        float y = -30 + 60 * u01(mt);
        float z = -10 + 20 * u01(mt);
        float radius = 5 + 5 * u01(mt);
        float r = u01(mt);
        float g = u01(mt);
        float b = u01(mt);
        bool emitting = u01(mt) < 0.25;
        Optical  o = emitting ? Optical::Emission(r * 2, g * 2, b * 2) : Optical::Reflection(r, g, b);
        objs.emplace_back(Shape(Sphere({x, y, z}, radius)), o);
    }
    
    std::vector<Vec3> vec = kernelmain(rs, objs, true);
    std::cout << "done" << std::endl;
    std::ofstream ofs("test.ppm");
    ofs << "P3" << std::endl;
    ofs << rs.WIDTH << ' ' << rs.HEIGHT << std::endl;
    ofs << 255 << std::endl;
    for (const auto &v : vec) {
        int r = (int)(255 * v.x);
        int g = (int)(255 * v.y);
        int b = (int)(255 * v.z);
        ofs << std::min(255, r) << ' ' << std::min(255, g) << ' ' << std::min(255, b) << ' ' << std::endl;
    }
    return 0;
}