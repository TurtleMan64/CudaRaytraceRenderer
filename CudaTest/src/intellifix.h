#pragma once

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define tex2D(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif
