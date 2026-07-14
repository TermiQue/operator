#pragma once

#include "fused_moe_i8_tn_types.h"

#define FUSED_MOE_CP_ASYNC_FENC() asm(";--------------")

#define FUSED_MOE_LDS(dst, src, type_)                                                              \
    FUSED_MOE_CP_ASYNC_FENC();                                                                      \
    *reinterpret_cast<type_ *>(&(dst)) = *reinterpret_cast<type_ *>(&(src));                        \
    FUSED_MOE_CP_ASYNC_FENC()

#define FUSED_MOE_STS(dst, src, type_)                                                              \
    FUSED_MOE_CP_ASYNC_FENC();                                                                      \
    *reinterpret_cast<type_ *>(&(dst)) = *reinterpret_cast<type_ *>(&(src));                        \
    FUSED_MOE_CP_ASYNC_FENC()

#if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1000 || __MACA_ARCH__ == 1089)
#define FUSED_MOE_BUILTIN_MMA_16X16X16_I8(a, b, c) __builtin_mxc_mma_16x16x16i8(a, b, c)
#else
#define FUSED_MOE_BUILTIN_MMA_16X16X16_I8(a, b, c) 0
#endif
