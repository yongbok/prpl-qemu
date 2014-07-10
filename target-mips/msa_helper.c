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
