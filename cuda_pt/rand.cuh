//#pragma once
//#include "common.h"
//#include "cuda_runtime.h"
//#if defined (__CUDA_ACRH__)
//#include <curand_kernel.h>
//#else
//#include <random>
//#endif
//
//
//struct RandGen {
//#if defined (__CUDA_ARCH__)
//    cuRandState_t state;
//    DEVICE RandGen(uint64_t seed, uint64_t subsequence, uint64_t offset) {
//        curand_init(seed, subsequence, offset, &state);
//    }
//#else
//    std::mt19937 mt;
//    HOST RandGen() : mt(std::random_device()()) {
//    }
//#endif
//};