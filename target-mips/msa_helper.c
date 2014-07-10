/*
 * MIPS SIMD Architecture Module Instruction emulation helpers for QEMU.
 *
 * Copyright (c) 2014 Imagination Technologies
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#include "cpu.h"
#include "helper.h"

#define DF_BYTE   0
#define DF_HALF   1
#define DF_WORD   2
#define DF_DOUBLE 3

static void msa_check_index(CPUMIPSState *env,
        uint32_t df, uint32_t n) {
    switch (df) {
    case DF_BYTE: /* b */
        if (n > MSA_WRLEN / 8 - 1) {
            helper_raise_exception(env, EXCP_RI);
        }
        break;
    case DF_HALF: /* h */
        if (n > MSA_WRLEN / 16 - 1) {
            helper_raise_exception(env, EXCP_RI);
        }
        break;
    case DF_WORD: /* w */
        if (n > MSA_WRLEN / 32 - 1) {
            helper_raise_exception(env, EXCP_RI);
        }
        break;
    case DF_DOUBLE: /* d */
        if (n > MSA_WRLEN / 64 - 1) {
            helper_raise_exception(env, EXCP_RI);
        }
        break;
    default:
        /* shouldn't get here */
        assert(0);
    }
}

/* Data format min and max values */
#define DF_BITS(df) (1 << ((df) + 3))

#define DF_MAX_INT(df)  (int64_t)((1LL << (DF_BITS(df) - 1)) - 1)
#define M_MAX_INT(m)    (int64_t)((1LL << ((m)         - 1)) - 1)

#define DF_MIN_INT(df)  (int64_t)(-(1LL << (DF_BITS(df) - 1)))
#define M_MIN_INT(m)    (int64_t)(-(1LL << ((m)         - 1)))

#define DF_MAX_UINT(df) (uint64_t)(-1ULL >> (64 - DF_BITS(df)))
#define M_MAX_UINT(m)   (uint64_t)(-1ULL >> (64 - (m)))

/* Data format bit position and unsigned values */
#define BIT_POSITION(x, df) ((uint64_t)(x) % DF_BITS(df))

#define UNSIGNED(x, df) ((x) & DF_MAX_UINT(df))
#define SIGNED(x, df)                                                   \
    ((((int64_t)x) << (64 - DF_BITS(df))) >> (64 - DF_BITS(df)))

/* Element-by-element access macros */
#define DF_ELEMENTS(df, wrlen) (wrlen / DF_BITS(df))

#define  B(pwr, i) (((wr_t *)pwr)->b[i])
#define BR(pwr, i) (((wr_t *)pwr)->b[i])
#define BL(pwr, i) (((wr_t *)pwr)->b[i + MSA_WRLEN/16])

#define ALL_B_ELEMENTS(i, wrlen)                \
    do {                                        \
        uint32_t i;                             \
        for (i = wrlen / 8; i--;)

#define  H(pwr, i) (((wr_t *)pwr)->h[i])
#define HR(pwr, i) (((wr_t *)pwr)->h[i])
#define HL(pwr, i) (((wr_t *)pwr)->h[i + MSA_WRLEN/32])

#define ALL_H_ELEMENTS(i, wrlen)                \
    do {                                        \
        uint32_t i;                             \
        for (i = wrlen / 16; i--;)

#define  W(pwr, i) (((wr_t *)pwr)->w[i])
#define WR(pwr, i) (((wr_t *)pwr)->w[i])
#define WL(pwr, i) (((wr_t *)pwr)->w[i + MSA_WRLEN/64])

#define ALL_W_ELEMENTS(i, wrlen)                \
    do {                                        \
        uint32_t i;                             \
        for (i = wrlen / 32; i--;)

#define  D(pwr, i) (((wr_t *)pwr)->d[i])
#define DR(pwr, i) (((wr_t *)pwr)->d[i])
#define DL(pwr, i) (((wr_t *)pwr)->d[i + MSA_WRLEN/128])

#define ALL_D_ELEMENTS(i, wrlen)                \
    do {                                        \
        uint32_t i;                             \
        for (i = wrlen / 64; i--;)

#define Q(pwr, i) (((wr_t *)pwr)->q[i])
#define ALL_Q_ELEMENTS(i, wrlen)                \
    do {                                        \
        uint32_t i;                             \
        for (i = wrlen / 128; i--;)

#define DONE_ALL_ELEMENTS                       \
    } while (0)

static inline void msa_move_v(void *pwd, void *pws)
{
    ALL_D_ELEMENTS(i, MSA_WRLEN) {
        D(pwd, i) = D(pws, i);
    } DONE_ALL_ELEMENTS;
}

static inline uint64_t msa_load_wr_elem_i64(CPUMIPSState *env, int32_t wreg,
        int32_t df, int32_t i)
{
    i %= DF_ELEMENTS(df, MSA_WRLEN);
    msa_check_index(env, (uint32_t)df, (uint32_t)i);

    switch (df) {
    case DF_BYTE: /* b */
        return (uint8_t)env->active_fpu.fpr[wreg].wr.b[i];
    case DF_HALF: /* h */
        return (uint16_t)env->active_fpu.fpr[wreg].wr.h[i];
    case DF_WORD: /* w */
        return (uint32_t)env->active_fpu.fpr[wreg].wr.w[i];
    case DF_DOUBLE: /* d */
        return (uint64_t)env->active_fpu.fpr[wreg].wr.d[i];
    default:
        /* shouldn't get here */
        assert(0);
    }
}

static inline int64_t msa_load_wr_elem_s64(CPUMIPSState *env, int32_t wreg,
        int32_t df, int32_t i)
{
    i %= DF_ELEMENTS(df, MSA_WRLEN);
    msa_check_index(env, (uint32_t)df, (uint32_t)i);

    switch (df) {
    case DF_BYTE: /* b */
        return env->active_fpu.fpr[wreg].wr.b[i];
    case DF_HALF: /* h */
        return env->active_fpu.fpr[wreg].wr.h[i];
    case DF_WORD: /* w */
        return env->active_fpu.fpr[wreg].wr.w[i];
    case DF_DOUBLE: /* d */
        return env->active_fpu.fpr[wreg].wr.d[i];
    default:
        /* shouldn't get here */
        assert(0);
    }
}

static inline void msa_store_wr_elem(CPUMIPSState *env, uint64_t val,
        int32_t wreg, int32_t df, int32_t i)
{
    i %= DF_ELEMENTS(df, MSA_WRLEN);
    msa_check_index(env, (uint32_t)df, (uint32_t)i);

    switch (df) {
    case DF_BYTE: /* b */
        env->active_fpu.fpr[wreg].wr.b[i] = (uint8_t)val;
        break;
    case DF_HALF: /* h */
        env->active_fpu.fpr[wreg].wr.h[i] = (uint16_t)val;
        break;
    case DF_WORD: /* w */
        env->active_fpu.fpr[wreg].wr.w[i] = (uint32_t)val;
        break;
    case DF_DOUBLE: /* d */
        env->active_fpu.fpr[wreg].wr.d[i] = (uint64_t)val;
        break;
    default:
        /* shouldn't get here */
        assert(0);
    }
}

static inline int64_t msa_add_a_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t abs_arg1 = arg1 >= 0 ? arg1 : -arg1;
    uint64_t abs_arg2 = arg2 >= 0 ? arg2 : -arg2;
    return abs_arg1 + abs_arg2;
}

void helper_msa_add_a_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_add_a_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_addv_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = (int64_t) ts + tt;
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_addvi_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t u5)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = (int64_t) ts + u5;
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_subv_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = (int64_t) ts - tt;
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_subvi_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t u5)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = (int64_t) ts - u5;
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_adds_a_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t max_int = (uint64_t)DF_MAX_INT(df);
    uint64_t abs_arg1 = arg1 >= 0 ? arg1 : -arg1;
    uint64_t abs_arg2 = arg2 >= 0 ? arg2 : -arg2;
    if (abs_arg1 > max_int || abs_arg2 > max_int) {
        return (int64_t)max_int;
    } else {
        return (abs_arg1 < max_int - abs_arg2) ? abs_arg1 + abs_arg2 : max_int;
    }
}

void helper_msa_adds_a_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_adds_a_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_adds_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    int64_t max_int = DF_MAX_INT(df);
    int64_t min_int = DF_MIN_INT(df);
    if (arg1 < 0) {
        return (min_int - arg1 < arg2) ? arg1 + arg2 : min_int;
    } else {
        return (arg2 < max_int - arg1) ? arg1 + arg2 : max_int;
    }
}

void helper_msa_adds_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_adds_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline uint64_t msa_adds_u_df(CPUMIPSState *env, uint32_t df,
        uint64_t arg1, uint64_t arg2)
{
    uint64_t max_uint = DF_MAX_UINT(df);
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return (u_arg1 < max_uint - u_arg2) ? u_arg1 + u_arg2 : max_uint;
}

void helper_msa_adds_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_adds_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_subs_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    int64_t max_int = DF_MAX_INT(df);
    int64_t min_int = DF_MIN_INT(df);
    if (arg2 > 0) {
        return (min_int + arg2 < arg1) ? arg1 - arg2 : min_int;
    } else {
        return (arg1 < max_int + arg2) ? arg1 - arg2 : max_int;
    }
}

void helper_msa_subs_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_subs_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_subs_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return (u_arg1 > u_arg2) ? u_arg1 - u_arg2 : 0;
}

void helper_msa_subs_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_subs_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_subsuu_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    int64_t max_int = DF_MAX_INT(df);
    int64_t min_int = DF_MIN_INT(df);
    if (u_arg1 > u_arg2) {
        return u_arg1 - u_arg2 < (uint64_t)max_int ?
            (int64_t)(u_arg1 - u_arg2) :
            max_int;
    } else {
        return u_arg2 - u_arg1 < (uint64_t)(-min_int) ?
            (int64_t)(u_arg1 - u_arg2) :
            min_int;
    }
}

void helper_msa_subsuu_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_subsuu_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_subsus_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t max_uint = DF_MAX_UINT(df);
    if (arg2 >= 0) {
        uint64_t u_arg2 = (uint64_t)arg2;
        return (u_arg1 > u_arg2) ?
            (int64_t)(u_arg1 - u_arg2) :
            0;
    } else {
        uint64_t u_arg2 = (uint64_t)(-arg2);
        return (u_arg1 < max_uint - u_arg2) ?
            (int64_t)(u_arg1 + u_arg2) :
            (int64_t)max_uint;
    }
}

void helper_msa_subsus_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_subsus_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_andi_b(CPUMIPSState *env, uint32_t wd, uint32_t ws,
        uint32_t i8)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    ALL_B_ELEMENTS(i, MSA_WRLEN) {
        B(pwd, i) = B(pws, i) & i8;
    } DONE_ALL_ELEMENTS;
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_ori_b(CPUMIPSState *env, uint32_t wd, uint32_t ws,
        uint32_t i8)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    ALL_B_ELEMENTS(i, MSA_WRLEN) {
        B(pwd, i) = B(pws, i) | i8;
    } DONE_ALL_ELEMENTS;
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_nori_b(CPUMIPSState *env, uint32_t wd, uint32_t ws,
        uint32_t i8)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    ALL_B_ELEMENTS(i, MSA_WRLEN) {
        B(pwd, i) = ~(B(pws, i) | i8);
    } DONE_ALL_ELEMENTS;
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_xori_b(CPUMIPSState *env, uint32_t wd, uint32_t ws,
        uint32_t i8)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    ALL_B_ELEMENTS(i, MSA_WRLEN) {
        B(pwd, i) = B(pws, i) ^ i8;
    } DONE_ALL_ELEMENTS;
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_asub_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    /* signed compare */
    return (arg1 < arg2) ?
        (uint64_t)(arg2 - arg1) : (uint64_t)(arg1 - arg2);
}

void helper_msa_asub_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_asub_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline uint64_t msa_asub_u_df(CPUMIPSState *env, uint32_t df,
        uint64_t arg1, uint64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    /* unsigned compare */
    return (u_arg1 < u_arg2) ?
        (uint64_t)(u_arg2 - u_arg1) : (uint64_t)(u_arg1 - u_arg2);
}

void helper_msa_asub_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_asub_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_ave_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    /* signed shift */
    return (arg1 >> 1) + (arg2 >> 1) + (arg1 & arg2 & 1);
}

void helper_msa_ave_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_ave_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline uint64_t msa_ave_u_df(CPUMIPSState *env, uint32_t df,
        uint64_t arg1, uint64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    /* unsigned shift */
    return (u_arg1 >> 1) + (u_arg2 >> 1) + (u_arg1 & u_arg2 & 1);
}

void helper_msa_ave_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_ave_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_aver_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    /* signed shift */
    return (arg1 >> 1) + (arg2 >> 1) + ((arg1 | arg2) & 1);
}

void helper_msa_aver_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_aver_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline uint64_t msa_aver_u_df(CPUMIPSState *env, uint32_t df,
        uint64_t arg1, uint64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    /* unsigned shift */
    return (u_arg1 >> 1) + (u_arg2 >> 1) + ((u_arg1 | u_arg2) & 1);
}

void helper_msa_aver_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_aver_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_bclr_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 & (~(1LL << b_arg2)), df);
}

void helper_msa_bclr_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_bclr_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_bclri_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_bclr_df(env, df, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_bneg_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 ^ (1LL << b_arg2), df);
}

void helper_msa_bneg_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_bneg_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_bnegi_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_bneg_df(env, df, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_bset_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return UNSIGNED(arg1 | (1LL << b_arg2), df);
}

void helper_msa_bset_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_bset_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_bseti_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_bset_df(env, df, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_binsl_df(CPUMIPSState *env, uint32_t df,
        int64_t dest, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_dest = UNSIGNED(dest, df);
    int32_t sh_d = BIT_POSITION(arg2, df) + 1;
    int32_t sh_a = DF_BITS(df) - sh_d;
    if (sh_d == DF_BITS(df)) {
        return u_arg1;
    } else {
        return UNSIGNED(UNSIGNED(u_dest << sh_d, df) >> sh_d, df) |
               UNSIGNED(UNSIGNED(u_arg1 >> sh_a, df) << sh_a, df);
    }
}

void helper_msa_binsl_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_binsl_df(env, df, td, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_binsli_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_binsl_df(env, df, td, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_binsr_df(CPUMIPSState *env, uint32_t df,
        int64_t dest, int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_dest = UNSIGNED(dest, df);
    int32_t sh_d = BIT_POSITION(arg2, df) + 1;
    int32_t sh_a = DF_BITS(df) - sh_d;
    if (sh_d == DF_BITS(df)) {
        return u_arg1;
    } else {
        return UNSIGNED(UNSIGNED(u_dest >> sh_d, df) << sh_d, df) |
               UNSIGNED(UNSIGNED(u_arg1 << sh_a, df) >> sh_a, df);
    }
}

void helper_msa_binsr_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_binsr_df(env, df, td, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_binsri_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_binsr_df(env, df, td, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

#define BIT_MOVE_IF_NOT_ZERO(dest, arg1, arg2, df) \
            dest = UNSIGNED(((dest & (~arg2)) | (arg1 & arg2)), df)

void helper_msa_bmnzi_b(CPUMIPSState *env, uint32_t wd, uint32_t ws,
        uint32_t i8)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    ALL_B_ELEMENTS(i, MSA_WRLEN) {
        BIT_MOVE_IF_NOT_ZERO(B(pwd, i), B(pws, i), i8, DF_BYTE);
    } DONE_ALL_ELEMENTS;
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

#define BIT_MOVE_IF_ZERO(dest, arg1, arg2, df) \
            dest = UNSIGNED((dest & arg2) | (arg1 & (~arg2)), df)

void helper_msa_bmzi_b(CPUMIPSState *env, uint32_t wd, uint32_t ws,
        uint32_t i8)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    ALL_B_ELEMENTS(i, MSA_WRLEN) {
        BIT_MOVE_IF_ZERO(B(pwd, i), B(pws, i), i8, DF_BYTE);
    } DONE_ALL_ELEMENTS;
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

#define BIT_SELECT(dest, arg1, arg2, df) \
            dest = UNSIGNED((arg1 & (~dest)) | (arg2 & dest), df)

void helper_msa_bseli_b(CPUMIPSState *env, uint32_t wd, uint32_t ws,
        uint32_t i8)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    ALL_B_ELEMENTS(i, MSA_WRLEN) {
        BIT_SELECT(B(pwd, i), B(pws, i), i8, DF_BYTE);
    } DONE_ALL_ELEMENTS;
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_ceq_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    return arg1 == arg2 ? -1 : 0;
}

void helper_msa_ceq_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_ceq_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_ceqi_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t i5)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_ceq_df(env, df, ts, i5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_cle_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return arg1 <= arg2 ? -1 : 0;
}

void helper_msa_cle_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_cle_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_clei_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t s5)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_cle_s_df(env, df, ts, s5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_cle_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 <= u_arg2 ? -1 : 0;
}

void helper_msa_cle_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_cle_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_clei_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t u5)
{
    uint64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        td = msa_cle_u_df(env, df, ts, u5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_clt_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return arg1 < arg2 ? -1 : 0;
}

void helper_msa_clt_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_clt_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_clti_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t s5)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_clt_s_df(env, df, ts, s5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_clt_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 < u_arg2 ? -1 : 0;
}

void helper_msa_clt_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_clt_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_clti_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t u5)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_clt_u_df(env, df, ts, u5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

#define SIGNED_EVEN(a, df) \
        ((((int64_t)(a)) << (64 - DF_BITS(df)/2)) >> (64 - DF_BITS(df)/2))
#define UNSIGNED_EVEN(a, df) \
        ((((uint64_t)(a)) << (64 - DF_BITS(df)/2)) >> (64 - DF_BITS(df)/2))

#define SIGNED_ODD(a, df) \
        ((((int64_t)(a)) << (64 - DF_BITS(df))) >> (64 - DF_BITS(df)/2))
#define UNSIGNED_ODD(a, df) \
        ((((uint64_t)(a)) << (64 - DF_BITS(df))) >> (64 - DF_BITS(df)/2))

#define SIGNED_EXTRACT(e, o, a, df)             \
    int64_t e = SIGNED_EVEN(a, df);             \
    int64_t o = SIGNED_ODD(a, df);

#define UNSIGNED_EXTRACT(e, o, a, df)           \
    int64_t e = UNSIGNED_EVEN(a, df);           \
    int64_t o = UNSIGNED_ODD(a, df);

static inline int64_t msa_hadd_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return SIGNED_ODD(arg1, df) + SIGNED_EVEN(arg2, df);
}

void helper_msa_hadd_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_hadd_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_hadd_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return UNSIGNED_ODD(arg1, df) + UNSIGNED_EVEN(arg2, df);
}

void helper_msa_hadd_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_hadd_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_hsub_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return SIGNED_ODD(arg1, df) - SIGNED_EVEN(arg2, df);
}

void helper_msa_hsub_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_hsub_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_hsub_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return UNSIGNED_ODD(arg1, df) - UNSIGNED_EVEN(arg2, df);
}

void helper_msa_hsub_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_hsub_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_dotp_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_msa_dotp_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_dotp_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_dotp_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_msa_dotp_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_dotp_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_dpadd_s_df(CPUMIPSState *env, uint32_t df,
        int64_t dest, int64_t arg1, int64_t arg2)
{
    SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest + (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_msa_dpadd_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_dpadd_s_df(env, df, td, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_dpadd_u_df(CPUMIPSState *env, uint32_t df,
        int64_t dest, int64_t arg1, int64_t arg2)
{
    UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest + (even_arg1 * even_arg2) + (odd_arg1 * odd_arg2);
}

void helper_msa_dpadd_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_dpadd_u_df(env, df, td, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_dpsub_s_df(CPUMIPSState *env, uint32_t df,
        int64_t dest, int64_t arg1, int64_t arg2)
{
    SIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    SIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest - ((even_arg1 * even_arg2) + (odd_arg1 * odd_arg2));
}

void helper_msa_dpsub_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_dpsub_s_df(env, df, td, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_dpsub_u_df(CPUMIPSState *env, uint32_t df,
        int64_t dest, int64_t arg1, int64_t arg2)
{
    UNSIGNED_EXTRACT(even_arg1, odd_arg1, arg1, df);
    UNSIGNED_EXTRACT(even_arg2, odd_arg2, arg2, df);
    return dest - ((even_arg1 * even_arg2) + (odd_arg1 * odd_arg2));
}

void helper_msa_dpsub_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_dpsub_u_df(env, df, td, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline void msa_ilvev_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, void *pwt)
{
    wr_t wx, *pwx = &wx;
    switch (df) {
    case DF_BYTE:
        /* byte data format */
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            B(pwx, 2*i)   = B(pwt, 2*i);
            B(pwx, 2*i+1) = B(pws, 2*i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        /* half data format */
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            H(pwx, 2*i)   = H(pwt, 2*i);
            H(pwx, 2*i+1) = H(pws, 2*i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        /* word data format */
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            W(pwx, 2*i)   = W(pwt, 2*i);
            W(pwx, 2*i+1) = W(pws, 2*i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        /* double data format */
        ALL_Q_ELEMENTS(i, MSA_WRLEN) {
            D(pwx, 2*i)   = D(pwt, 2*i);
            D(pwx, 2*i+1) = D(pws, 2*i);
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    msa_move_v(pwd, pwx);
}

void helper_msa_ilvev_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    void *pwt = &(env->active_fpu.fpr[wt]);
    msa_ilvev_df(env, df, pwd, pws, pwt);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline void msa_ilvod_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, void *pwt)
{
    wr_t wx, *pwx = &wx;
    switch (df) {
    case DF_BYTE:
        /* byte data format */
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            B(pwx, 2*i)   = B(pwt, 2*i+1);
            B(pwx, 2*i+1) = B(pws, 2*i+1);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        /* half data format */
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            H(pwx, 2*i)   = H(pwt, 2*i+1);
            H(pwx, 2*i+1) = H(pws, 2*i+1);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        /* word data format */
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            W(pwx, 2*i)   = W(pwt, 2*i+1);
            W(pwx, 2*i+1) = W(pws, 2*i+1);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        /* double data format */
        ALL_Q_ELEMENTS(i, MSA_WRLEN) {
            D(pwx, 2*i)   = D(pwt, 2*i+1);
            D(pwx, 2*i+1) = D(pws, 2*i+1);
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    msa_move_v(pwd, pwx);
}

void helper_msa_ilvod_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    void *pwt = &(env->active_fpu.fpr[wt]);
    msa_ilvod_df(env, df, pwd, pws, pwt);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline void msa_ilvl_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, void *pwt)
{
    wr_t wx, *pwx = &wx;
    switch (df) {
    case DF_BYTE:
        /* byte data format */
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            B(pwx, 2*i)   = BL(pwt, i);
            B(pwx, 2*i+1) = BL(pws, i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        /* half data format */
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            H(pwx, 2*i)   = HL(pwt, i);
            H(pwx, 2*i+1) = HL(pws, i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        /* word data format */
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            W(pwx, 2*i)   = WL(pwt, i);
            W(pwx, 2*i+1) = WL(pws, i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        /* double data format */
        ALL_Q_ELEMENTS(i, MSA_WRLEN) {
            D(pwx, 2*i)   = DL(pwt, i);
            D(pwx, 2*i+1) = DL(pws, i);
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    msa_move_v(pwd, pwx);
}

void helper_msa_ilvl_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    void *pwt = &(env->active_fpu.fpr[wt]);
    msa_ilvl_df(env, df, pwd, pws, pwt);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline void msa_ilvr_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, void *pwt)
{
    wr_t wx, *pwx = &wx;
    switch (df) {
    case DF_BYTE:
        /* byte data format */
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            B(pwx, 2*i)   = BR(pwt, i);
            B(pwx, 2*i+1) = BR(pws, i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        /* half data format */
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            H(pwx, 2*i)   = HR(pwt, i);
            H(pwx, 2*i+1) = HR(pws, i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        /* word data format */
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            W(pwx, 2*i)   = WR(pwt, i);
            W(pwx, 2*i+1) = WR(pws, i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        /* double data format */
        ALL_Q_ELEMENTS(i, MSA_WRLEN) {
            D(pwx, 2*i)   = DR(pwt, i);
            D(pwx, 2*i+1) = DR(pws, i);
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    msa_move_v(pwd, pwx);
}

void helper_msa_ilvr_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    void *pwt = &(env->active_fpu.fpr[wt]);
    msa_ilvr_df(env, df, pwd, pws, pwt);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline void msa_pckev_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, void *pwt)
{
    wr_t wx, *pwx = &wx;
    switch (df) {
    case DF_BYTE:
        /* byte data format */
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            BR(pwx, i) = B(pwt, 2*i);
            BL(pwx, i) = B(pws, 2*i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        /* half data format */
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            HR(pwx, i) = H(pwt, 2*i);
            HL(pwx, i) = H(pws, 2*i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        /* word data format */
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            WR(pwx, i) = W(pwt, 2*i);
            WL(pwx, i) = W(pws, 2*i);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        /* double data format */
        ALL_Q_ELEMENTS(i, MSA_WRLEN) {
            DR(pwx, i) = D(pwt, 2*i);
            DL(pwx, i) = D(pws, 2*i);
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    msa_move_v(pwd, pwx);
}

void helper_msa_pckev_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    void *pwt = &(env->active_fpu.fpr[wt]);
    msa_pckev_df(env, df, pwd, pws, pwt);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline void msa_pckod_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, void *pwt)
{
    wr_t wx, *pwx = &wx;
    switch (df) {
    case DF_BYTE:
        /* byte data format */
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            BR(pwx, i) = B(pwt, 2*i+1);
            BL(pwx, i) = B(pws, 2*i+1);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        /* half data format */
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            HR(pwx, i) = H(pwt, 2*i+1);
            HL(pwx, i) = H(pws, 2*i+1);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        /* word data format */
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            WR(pwx, i) = W(pwt, 2*i+1);
            WL(pwx, i) = W(pws, 2*i+1);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        /* double data format */
        ALL_Q_ELEMENTS(i, MSA_WRLEN) {
            DR(pwx, i) = D(pwt, 2*i+1);
            DL(pwx, i) = D(pws, 2*i+1);
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    msa_move_v(pwd, pwx);
}

void helper_msa_pckod_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    void *pwt = &(env->active_fpu.fpr[wt]);
    msa_pckod_df(env, df, pwd, pws, pwt);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline void msa_vshf_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, void *pwt)
{
    uint32_t n = MSA_WRLEN / DF_BITS(df);
    uint32_t k;
    wr_t wx, *pwx = &wx;
    switch (df) {
    case DF_BYTE:
        /* byte data format */
        ALL_B_ELEMENTS(i, MSA_WRLEN) {
            k = (B(pwd, i) & 0x3f) % (2 * n);
            B(pwx, i) =
                (B(pwd, i) & 0xc0) ? 0 : k < n ? B(pwt, k) : B(pws, k - n);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        /* half data format */
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            k = (H(pwd, i) & 0x3f) % (2 * n);
            H(pwx, i) =
                (H(pwd, i) & 0xc0) ? 0 : k < n ? H(pwt, k) : H(pws, k - n);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        /* word data format */
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            k = (W(pwd, i) & 0x3f) % (2 * n);
            W(pwx, i) =
                (W(pwd, i) & 0xc0) ? 0 : k < n ? W(pwt, k) : W(pws, k - n);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        /* double data format */
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            k = (D(pwd, i) & 0x3f) % (2 * n);
            D(pwx, i) =
                (D(pwd, i) & 0xc0) ? 0 : k < n ? D(pwt, k) : D(pws, k - n);
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    msa_move_v(pwd, pwx);
}

void helper_msa_vshf_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    void *pwt = &(env->active_fpu.fpr[wt]);
    msa_vshf_df(env, df, pwd, pws, pwt);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}


#define SHF_POS(i, imm) ((i & 0xfc) + ((imm >> (2 * (i & 0x03))) & 0x03))

static inline void msa_shf_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, uint32_t imm)
{
    wr_t wx, *pwx = &wx;
    switch (df) {
    case DF_BYTE:
      ALL_B_ELEMENTS(i, MSA_WRLEN) {
        B(pwx, i) = B(pws, SHF_POS(i, imm));
      } DONE_ALL_ELEMENTS;
      break;
    case DF_HALF:
      ALL_H_ELEMENTS(i, MSA_WRLEN) {
        H(pwx, i) = H(pws, SHF_POS(i, imm));
      } DONE_ALL_ELEMENTS;
      break;
    case DF_WORD:
      ALL_W_ELEMENTS(i, MSA_WRLEN) {
        W(pwx, i) = W(pws, SHF_POS(i, imm));
      } DONE_ALL_ELEMENTS;
      break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    msa_move_v(pwd, pwx);
}

void helper_msa_shf_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t imm)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    msa_shf_df(env, df, pwd, pws, imm);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_maddv_df(CPUMIPSState *env, uint32_t df,
        int64_t dest, int64_t arg1, int64_t arg2)
{
    return dest + arg1 * arg2;
}

void helper_msa_maddv_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_maddv_df(env, df, td, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_msubv_df(CPUMIPSState *env, uint32_t df,
        int64_t dest, int64_t arg1, int64_t arg2)
{
    return dest - arg1 * arg2;
}

void helper_msa_msubv_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_load_wr_elem_s64(env, wd, df, i);
        td = msa_msubv_df(env, df, td, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_max_a_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t abs_arg1 = arg1 >= 0 ? arg1 : -arg1;
    uint64_t abs_arg2 = arg2 >= 0 ? arg2 : -arg2;
    return abs_arg1 > abs_arg2 ? arg1 : arg2;
}

void helper_msa_max_a_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_max_a_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_max_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return arg1 > arg2 ? arg1 : arg2;
}

void helper_msa_max_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_max_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_maxi_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t s5)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_max_s_df(env, df, ts, s5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_max_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 > u_arg2 ? arg1 : arg2;
}

void helper_msa_max_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_max_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_maxi_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t u5)
{
    uint64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        td = msa_max_u_df(env, df, ts, u5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_min_a_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t abs_arg1 = arg1 >= 0 ? arg1 : -arg1;
    uint64_t abs_arg2 = arg2 >= 0 ? arg2 : -arg2;
    return abs_arg1 < abs_arg2 ? arg1 : arg2;
}

void helper_msa_min_a_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_min_a_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_min_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return arg1 < arg2 ? arg1 : arg2;
}

void helper_msa_min_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_min_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_mini_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t s5)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_min_s_df(env, df, ts, s5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_min_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg1 < u_arg2 ? arg1 : arg2;
}

void helper_msa_min_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_min_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_mini_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, int64_t u5)
{
    uint64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        td = msa_min_u_df(env, df, ts, u5);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline void msa_splat_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, target_ulong rt)
{
    uint32_t n = rt % DF_ELEMENTS(df, MSA_WRLEN);
    msa_check_index(env, df, n);
    switch (df) {
    case DF_BYTE:
        ALL_B_ELEMENTS(i, MSA_WRLEN) {
            B(pwd, i)   = B(pws, n);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            H(pwd, i)   = H(pws, n);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            W(pwd, i)   = W(pws, n);
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            D(pwd, i)   = D(pws, n);
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
}

void helper_msa_splat_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t rt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    msa_splat_df(env, df, pwd, pws, env->active_tc.gpr[rt]);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_ldi_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t s10)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    int64_t s64 = ((int64_t)s10 << 54) >> 54;
    switch (df) {
    case DF_BYTE:
        ALL_B_ELEMENTS(i, MSA_WRLEN) {
            B(pwd, i)   = (int8_t)s10;
        } DONE_ALL_ELEMENTS;
        break;
    case DF_HALF:
        ALL_H_ELEMENTS(i, MSA_WRLEN) {
            H(pwd, i)   = (int16_t)s64;
        } DONE_ALL_ELEMENTS;
        break;
    case DF_WORD:
        ALL_W_ELEMENTS(i, MSA_WRLEN) {
            W(pwd, i)   = (int32_t)s64;
        } DONE_ALL_ELEMENTS;
        break;
    case DF_DOUBLE:
        ALL_D_ELEMENTS(i, MSA_WRLEN) {
            D(pwd, i)   = s64;
        } DONE_ALL_ELEMENTS;
       break;
    default:
        /* shouldn't get here */
        assert(0);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_mulv_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = ts * tt;
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_div_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    if (arg1 == DF_MIN_INT(df) && arg2 == -1) {
        return DF_MIN_INT(df);
    }
    return arg2 ? arg1 / arg2 : 0;
}

void helper_msa_div_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_div_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_div_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg2 ? u_arg1 / u_arg2 : 0;
}

void helper_msa_div_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_div_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_mod_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    if (arg1 == DF_MIN_INT(df) && arg2 == -1) {
        return 0;
    }
    return arg2 ? arg1 % arg2 : 0;
}

void helper_msa_mod_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_mod_s_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_mod_u_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    uint64_t u_arg2 = UNSIGNED(arg2, df);
    return u_arg2 ? u_arg1 % u_arg2 : 0;
}

void helper_msa_mod_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    uint64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        tt = msa_load_wr_elem_i64(env, wt, df, i);
        td = msa_mod_u_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_sat_u_df(CPUMIPSState *env, uint32_t df, int64_t arg,
        uint32_t m)
{
    uint64_t u_arg = UNSIGNED(arg, df);
    return  u_arg < M_MAX_UINT(m+1) ? u_arg :
                                      M_MAX_UINT(m+1);
}

void helper_msa_sat_u_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    uint64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_i64(env, ws, df, i);
        td = msa_sat_u_df(env, df, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_sat_s_df(CPUMIPSState *env, uint32_t df, int64_t arg,
        uint32_t m)
{
    return arg < M_MIN_INT(m+1) ? M_MIN_INT(m+1) :
                                  arg > M_MAX_INT(m+1) ? M_MAX_INT(m+1) :
                                                         arg;
}

void helper_msa_sat_s_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_sat_s_df(env, df, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_sll_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return arg1 << b_arg2;
}

void helper_msa_sll_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_sll_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_slli_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = ts << m;
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_sra_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return arg1 >> b_arg2;
}

void helper_msa_sra_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_sra_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

void helper_msa_srai_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = ts >> m;
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_srl_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    return u_arg1 >> b_arg2;
}

void helper_msa_srl_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_srl_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_srli_df(CPUMIPSState *env, uint32_t df, int64_t arg,
        uint32_t m)
{
    uint64_t u_arg = UNSIGNED(arg, df);
    return u_arg >> m;
}

void helper_msa_srli_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_srli_df(env, df, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_srar_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    if (b_arg2 == 0) {
        return arg1;
    } else {
        int64_t r_bit = (arg1 >> (b_arg2 - 1)) & 1;
        return (arg1 >> b_arg2) + r_bit;
    }
}

void helper_msa_srar_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_srar_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_srari_df(CPUMIPSState *env, uint32_t df, int64_t arg,
        uint32_t m)
{
    if (m == 0) {
        return arg;
    } else {
        int64_t r_bit = (arg >> (m - 1)) & 1;
        return (arg >> m) + r_bit;
    }
}

void helper_msa_srari_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_srari_df(env, df, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_srlr_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    uint64_t u_arg1 = UNSIGNED(arg1, df);
    int32_t b_arg2 = BIT_POSITION(arg2, df);
    if (b_arg2 == 0) {
        return u_arg1;
    } else {
        uint64_t r_bit = (u_arg1 >> (b_arg2 - 1)) & 1;
        return (u_arg1 >> b_arg2) + r_bit;
    }
}

void helper_msa_srlr_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t wt)
{
    int64_t td, ts, tt;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        tt = msa_load_wr_elem_s64(env, wt, df, i);
        td = msa_srlr_df(env, df, ts, tt);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}

static inline int64_t msa_srlri_df(CPUMIPSState *env, uint32_t df, int64_t arg,
        uint32_t m)
{
    uint64_t u_arg = UNSIGNED(arg, df);
    if (m == 0) {
        return u_arg;
    } else {
        uint64_t r_bit = (u_arg >> (m - 1)) & 1;
        return (u_arg >> m) + r_bit;
    }
}

void helper_msa_srlri_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t m)
{
    int64_t td, ts;
    int i;
    int df_bits = 8 * (1 << df);
    for (i = 0; i < MSA_WRLEN / df_bits; i++) {
        ts = msa_load_wr_elem_s64(env, ws, df, i);
        td = msa_srlri_df(env, df, ts, m);
        msa_store_wr_elem(env, td, wd, df, i);
    }
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}
static inline void msa_sld_df(CPUMIPSState *env, uint32_t df, void *pwd,
        void *pws, target_ulong rt)
{
    uint32_t n = rt % DF_ELEMENTS(df, MSA_WRLEN);
    uint8_t v[64];
    uint32_t i, k;
#define CONCATENATE_AND_SLIDE(s, k)             \
    do {                                        \
        for (i = 0; i < s; i++) {               \
            v[i]     = B(pws, s * k + i);       \
            v[i + s] = B(pwd, s * k + i);       \
        }                                       \
        for (i = 0; i < s; i++) {               \
            B(pwd, s * k + i) = v[i + n];       \
        }                                       \
    } while (0)

    msa_check_index(env, df, n);
    switch (df) {
    case DF_BYTE:
        CONCATENATE_AND_SLIDE(MSA_WRLEN/8, 0);
        break;
    case DF_HALF:
        for (k = 0; k < 2; k++) {
            CONCATENATE_AND_SLIDE(MSA_WRLEN/16, k);
        }
        break;
    case DF_WORD:
        for (k = 0; k < 4; k++) {
            CONCATENATE_AND_SLIDE(MSA_WRLEN/32, k);
        }
        break;
    case DF_DOUBLE:
        for (k = 0; k < 8; k++) {
            CONCATENATE_AND_SLIDE(MSA_WRLEN/64, k);
        }
        break;
    default:
        /* shouldn't get here */
        assert(0);
    }
}

void helper_msa_sld_df(CPUMIPSState *env, uint32_t df, uint32_t wd,
        uint32_t ws, uint32_t rt)
{
    void *pwd = &(env->active_fpu.fpr[wd]);
    void *pws = &(env->active_fpu.fpr[ws]);
    msa_sld_df(env, df, pwd, pws, env->active_tc.gpr[rt]);
    if (env->active_msa.msair & MSAIR_WRP_BIT) {
        env->active_msa.msamodify |= (1 << wd);
    }
}
