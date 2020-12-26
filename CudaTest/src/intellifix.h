#pragma once

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif
