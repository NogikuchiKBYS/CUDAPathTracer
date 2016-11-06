#pragma once

#if defined (__CUDACC__)
#define DEVICE __device__
#define HOST __host__
#else
#define DEVICE
#define HOST
#endif
