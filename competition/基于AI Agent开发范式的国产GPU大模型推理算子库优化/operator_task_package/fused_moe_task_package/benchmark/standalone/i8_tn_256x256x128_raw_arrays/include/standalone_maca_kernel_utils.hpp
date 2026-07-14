#pragma once

#include <mc_runtime.h>

#define standalone_arrive_gvmcnt(count) __builtin_mxc_arrive(64 + count)

#if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1000 || __MACA_ARCH__ == 1089)
#define STANDALONE_BUILTIN_MMA_16X16X16_I8(a, b, c) __builtin_mxc_mma_16x16x16i8(a, b, c)
#else
#define STANDALONE_BUILTIN_MMA_16X16X16_I8(a, b, c) 0
#endif

#define standalone_cp_async_fenc() asm(";--------------")

#define STANDALONE_LDS(dst, src, ldstype)                                                          \
    standalone_cp_async_fenc();                                                                    \
    *reinterpret_cast<ldstype *>(&(dst)) = *reinterpret_cast<ldstype *>(&(src));                  \
    standalone_cp_async_fenc()
