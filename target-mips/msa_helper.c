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

static inline int64_t msa_bclr_df(CPUMIPSState *env, uint32_t df, int64_t arg1,
        int64_t arg2)
{
    int32_t b_arg2 = BIT_POSITION(arg2, df);

    return UNSIGNED(arg1 & (~(1LL << b_arg2)), df);
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

static inline int64_t msa_max_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return arg1 > arg2 ? arg1 : arg2;
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

static inline int64_t msa_min_s_df(CPUMIPSState *env, uint32_t df,
        int64_t arg1, int64_t arg2)
{
    return arg1 < arg2 ? arg1 : arg2;
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
