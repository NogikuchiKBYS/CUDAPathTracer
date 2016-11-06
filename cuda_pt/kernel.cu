
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cmath>
#include "kernel.h"
#include "shape.h"
#include <thrust/random.h>

constexpr float PI = 3.141592653589793238462643383279502884197169399375105820974944f;

struct RenderInfo : RenderSettings {
    float pixelsize;
    Vec3 center_dir;
    Vec3 screeny;
    Vec3 screenx;


    DEVICE HOST RenderInfo(const RenderSettings &rs) : RenderSettings (rs) {
        pixelsize = rs.screenwidth / rs.WIDTH;
        center_dir = (rs.viewat - rs.viewfrom).normalized();
        screeny = (rs.upvec - center_dir * center_dir.dot(rs.upvec)).normalized();
        screenx = center_dir.cross(screeny);
    }
};

__device__ __host__ Vec3 uniformHemisphere(const Vec3 &nZ, thrust::ranlux24 &rgen)
{
    Vec3 x1 = nZ.cross(Vec3(1, 0, 0));
    Vec3 x2 = nZ.cross(Vec3(0, 1, 0));
    Vec3 nX = (x1.sqNorm() > x2.sqNorm()) ? (x1.normalized()) : (x2.normalized());
    Vec3 nY = nZ.cross(nX);

    thrust::uniform_real_distribution<float> dist_phi(-PI, PI);
    float phi = dist_phi(rgen);

    thrust::uniform_real_distribution<float> dist_costheta(0.0, 1.0);
    float costheta = dist_costheta(rgen);
    float theta = std::acos(costheta);

    return nX * (std::cos(phi) * std::sin(theta)) + nY * (std::sin(phi) * std::sin(theta)) + nZ * costheta;
}

__device__ __host__ Vec3 pixelproc(RenderInfo ri, size_t ixx, size_t ixy, const Object *objs, size_t objs_n, size_t npath, thrust::ranlux24 &rgen_g)
{
    //copy radom generator from global memory
    thrust::ranlux24 rgen = rgen_g;

    Sphere sphere({ 0, 10, 0 }, 5.0);
    thrust::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    float dx = ri.pixelsize * (ixx - ri.WIDTH / 2.0f + dist(rgen)) ;
    float dy = ri.pixelsize * (ri.HEIGHT / 2.0f - ixy + dist(rgen));
    Vec3 pos = ri.viewat + dx * ri.screenx + dy * ri.screeny;

    Ray ray = Ray::FromTo(ri.viewfrom, pos);

    Vec3 ret = { 0, 0, 0 };
    Vec3 contrib = { 1, 1, 1 };
    for (int depth = 0; depth < 10; depth++) {
        bool intersect = false;
        size_t objid = 0;
        float nearest = 0;
        for (size_t i = 0; i < objs_n; i++) {
            float dist = objs[i].shape.firstIntersecttion(ray);
            if (dist > 0) {
                if (!intersect || dist < nearest) {
                    intersect = true;
                    objid = i;
                    nearest = dist;
                }
            }
        }

        if (intersect) {
            Vec3 hitpos = ray.atDistance(nearest);
            const auto &emission = objs[objid].optical.emission;
            const auto &reflection = objs[objid].optical.reflection;
            ret.x += emission.x * contrib.x;
            ret.y += emission.y * contrib.y;
            ret.z += emission.z * contrib.z;
            contrib.x *= reflection.x;
            contrib.y *= reflection.y;
            contrib.z *= reflection.z;


            Vec3 norm = objs[objid].shape.getNormal(hitpos);
            if (norm.dot(ray.n_dir) > 0) {
                norm = -1 * norm;
            }
            ray.n_dir = uniformHemisphere(norm, rgen);
            float direction_prob_density = 1 / (4 * PI);
            float brdf_coeff = 1 / (4 * PI);
            contrib = contrib * brdf_coeff / direction_prob_density;
            //ray.n_dir = ray.n_dir - 2 * norm * ray.n_dir.dot(norm);
            ray.start = hitpos + 1e-3f * ray.n_dir;

        }
        else {
            break;
        }
        if (contrib.norm() < 1e-6f) {
            break;
        }
    }
    //restore random generator to global memory
    rgen_g = rgen;
    return ret;
}

__global__ void initialize_rgens(RenderInfo ri, thrust::ranlux24 *rgens)
{
    size_t ixx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t ixy = blockDim.y * blockIdx.y + threadIdx.y;

    size_t ix = ixy * ri.WIDTH + ixx;
    if (ixx < ri.WIDTH && ixy < ri.HEIGHT) {
        rgens[ix] = thrust::ranlux24(ix);
    }
}

__global__ void pixelproc_kernel(RenderInfo ri, Vec3 *map, Object *objs, size_t objs_n, size_t npath,thrust::ranlux24 *rgens)
{
    size_t ixx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t ixy = blockDim.y * blockIdx.y + threadIdx.y;
    
    size_t ix = ixy * ri.WIDTH + ixx;
    if (ixx < ri.WIDTH && ixy < ri.HEIGHT) {
        //thrust::uniform_int_distribution<uint32_t> dist_seed;
        //seed random generator
        //don't use given generator to avoid passing a reference to the global memory
        //thrust::ranlux24 rgen(dist_seed(rgens[ix]));
        map[ix] = map[ix] + pixelproc(ri, ixx, ixy, objs, objs_n, npath, rgens[ix]);
    }
}



std::vector<Vec3> kernelmain(const RenderSettings &rs, const std::vector<Object> &objs, bool usegpu)
{
    RenderInfo ri(rs);
    printf("ex %f %f %f\n", ri.screenx.x, ri.screenx.y, ri.screenx.z);
    printf("ey %f %f %f\n", ri.screeny.x, ri.screeny.y, ri.screeny.z);
    printf("ez %f %f %f\n", -ri.center_dir.x, -ri.center_dir.y, -ri.center_dir.z);
    std::vector<Vec3> v(rs.HEIGHT * rs.WIDTH, { 0, 0, 0 });
    if (usegpu) {
        Vec3 *result;
        cudaMalloc(&result, sizeof(Vec3) * rs.HEIGHT * rs.WIDTH);
        cudaMemset(result, 0, sizeof(Vec3) * rs.HEIGHT * rs.WIDTH);

        Object *o;
        cudaMalloc(&o, sizeof(Object) * objs.size());
        cudaMemcpy(o, objs.data(), sizeof(Object) * objs.size(), cudaMemcpyHostToDevice);

       
        dim3 threadsPerBlock(16, 16);
        dim3 blocks(rs.HEIGHT / 16 + 1, rs.WIDTH / 16 + 1);

        thrust::ranlux24 *rgens_dev;
        cudaMalloc(&rgens_dev, sizeof(thrust::ranlux24) * ri.HEIGHT * ri.WIDTH);
        //seed random generators
        initialize_rgens <<<blocks, threadsPerBlock>>> (ri, rgens_dev);

        for (size_t i = 0; i < ri.PATH; i++) {
            if (i % 10 == 0) {
                printf("%zu / %zu\n", i, ri.PATH);
            }
            pixelproc_kernel<<<blocks, threadsPerBlock>>>(ri, result, o, objs.size(), i, rgens_dev);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(v.data(), result, sizeof(Vec3) * rs.HEIGHT * rs.WIDTH, cudaMemcpyDeviceToHost);
        cudaFree(o);
        cudaFree(result);
        cudaFree(rgens_dev);
    }
    else {
      
        for (size_t ixy = 0; ixy < rs.HEIGHT; ixy++) {
            printf("%zu\n", ixy);
            for (size_t ixx = 0; ixx < rs.WIDTH; ixx++) {
                size_t ix = ixy * rs.WIDTH + ixx;
                thrust::ranlux24 rgen((uint32_t)(ixy * ri.HEIGHT + ixx));
                for (size_t i = 0; i < ri.PATH; i++) {
                    v.at(ix) = v.at(ix) + pixelproc(ri, ixx, ixy, objs.data(), objs.size(), i, rgen);
                }
            }
        }
    }
    for (auto &pixel : v) {
        pixel = pixel / (float)ri.PATH;
    }
    return v;
}