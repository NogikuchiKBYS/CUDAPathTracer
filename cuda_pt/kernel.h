#pragma once
#include "common.h"
#include "vector.h"
#include "shape.h"
#include <vector>

struct RenderSettings {
    Vec3 viewfrom, viewat, upvec;
    float screenwidth;
    size_t WIDTH, HEIGHT;
    size_t PATH;
};
std::vector<Vec3> kernelmain(const RenderSettings &rs, const std::vector<Object> &objs, bool usegpu);