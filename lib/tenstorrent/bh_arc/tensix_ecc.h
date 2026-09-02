/*
 * Copyright (c) 2026 Tenstorrent AI ULC
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TENSIX_ECC_H
#define TENSIX_ECC_H

#include <stdbool.h>
#include <stdint.h>

#include <zephyr/sys/util.h>

/* Per-Tensix RISCV debug registers. One instance per tile, shared by all five RISC cores. */
#define TENSIX_ECC_CTRL   0xFFB121D0
#define TENSIX_ECC_STATUS 0xFFB121D4

/*
 * ECC_STATUS is a windowed register: ECC_CTRL[3:0] selects which view it returns.
 * Any select other than 1-6 returns TENSIX_ECC_LIVENESS_MAGIC.
 */
#define TENSIX_ECC_SEL_COMBINED_CNT 1U   /* localmem parity counters + L1 counters */
#define TENSIX_ECC_SEL_L1_CNT       2U   /* L1 SBE/DBE counters                    */
#define TENSIX_ECC_SEL_LAST_L1_LO   3U   /* last L1 error, banks 0-15              */
#define TENSIX_ECC_SEL_NOC          4U   /* NOC mem parity / header SBE / hdr DBE  */
#define TENSIX_ECC_SEL_IRAM_CNT     5U   /* iram parity counters + ncrisc localmem */
#define TENSIX_ECC_SEL_LAST_L1_HI   6U   /* last L1 error, banks 16-31             */
#define TENSIX_ECC_SEL_LIVENESS     0xFU /* not a view - returns the magic value   */

#define TENSIX_ECC_SEL_FIRST TENSIX_ECC_SEL_COMBINED_CNT
#define TENSIX_ECC_SEL_LAST  TENSIX_ECC_SEL_LAST_L1_HI

#define TENSIX_ECC_LIVENESS_MAGIC 0xDEADBEEFU

/*
 * Field extraction. Widths are Blackhole-specific: L1_BANK_CNT = 32, so the bank one-hot is
 * split across two views, and L1_BANK_ADDR_WIDTH = 12.
 */
#define TENSIX_ECC_L1_DBE_CNT(s)   FIELD_GET(GENMASK(5, 0), (s))   /* sel=2, 6 bits */
#define TENSIX_ECC_L1_DBE_OVF(s)   FIELD_GET(BIT(6), (s))
#define TENSIX_ECC_L1_SBE_CNT(s)   FIELD_GET(GENMASK(14, 7), (s))  /* sel=2, 8 bits */
#define TENSIX_ECC_L1_SBE_OVF(s)   FIELD_GET(BIT(15), (s))
#define TENSIX_ECC_L1_ERR_VALID(s) FIELD_GET(BIT(31), (s))         /* sel=3 / sel=6 */
#define TENSIX_ECC_L1_ERR_BANK(s)  FIELD_GET(GENMASK(27, 12), (s))
#define TENSIX_ECC_L1_ERR_ADDR(s)  FIELD_GET(GENMASK(11, 0), (s))

#define TENSIX_ECC_NOC_MEM_PARITY(s) FIELD_GET(BIT(0), (s))        /* sel=4 */
#define TENSIX_ECC_NOC_HDR_SBE(s)    FIELD_GET(BIT(1), (s))
#define TENSIX_ECC_NOC_HDR_DBE(s)    FIELD_GET(BIT(2), (s))

/*
 * The tile's own NOC 0 NIU registers, reachable over the same NOC2AXI window as the debug
 * registers above but a completely separate block. NiuRegsBase() returns this same address
 * for a Tensix node; it is spelled out here to keep this file self-contained.
 *
 * Note the NIU has its own ECC_CTRL at offset 0x5C that is unrelated to TENSIX_ECC_CTRL.
 */
#define NOC_NIU_CFG_0 0xFFB20100

/*
 * NIU_CFG_0[11:9] gate the NIU's three o_ecc_error outputs, which are what this tile reports
 * in ECC_STATUS sel=4. The NIU error counters at 0xFFB20050/54/58 count either way - these
 * bits only decide whether a non-zero counter is visible to Tensix.
 */
#define NOC_NIU_CFG_0_ECC_MEM_PARITY_INT_EN BIT(9)
#define NOC_NIU_CFG_0_ECC_HDR_SBE_INT_EN    BIT(10)
#define NOC_NIU_CFG_0_ECC_HDR_DBE_INT_EN    BIT(11)
#define NOC_NIU_CFG_0_ECC_INT_EN            GENMASK(11, 9)

/*
 * The NIU's three ECC error counters. Read-only, and NOC_ECC_ERROR_COUNTER_WIDTH is 16 with no
 * saturation - a link erroring continuously wraps back through zero and briefly looks healthy.
 * The counters increment whether or not NIU_CFG_0[11:9] is set.
 */
#define NOC_NIU_NUM_MEM_PARITY_ERR 0xFFB20050
#define NOC_NIU_NUM_HEADER_1B_ERR  0xFFB20054
#define NOC_NIU_NUM_HEADER_2B_ERR  0xFFB20058

/*
 * The NIU's own ECC_CTRL - not TENSIX_ECC_CTRL. Write-only: reads return zero, so this one must
 * never be read-modify-written. Each of the three sources has a clear bit in [2:0] and a force
 * bit in [5:3]; both are decoded off the write strobe, so one write is one pulse, and a force
 * bumps its counter by exactly one.
 */
#define NOC_NIU_ECC_CTRL       0xFFB2005C
#define NOC_NIU_ECC_CTRL_CLEAR GENMASK(2, 0)
#define NOC_NIU_ECC_CTRL_FORCE GENMASK(5, 3)

/* Bit position within either 3-bit field above, and index into the counter array below. */
#define NOC_ECC_MEM_PARITY   0
#define NOC_ECC_HDR_SBE      1
#define NOC_ECC_HDR_DBE      2
#define NOC_ECC_NUM_SOURCES  3
#define NOC_ECC_SOURCE_MASK  GENMASK(2, 0)

/*
 * Parity counters are packed as 4-bit nibbles of {overflow, count[2:0]}, indexed from the
 * bottom of the word. Which nibble holds which core is not uniform across the two views:
 *
 *   localmem brisc, trisc0, trisc1, trisc2 -> sel=1, nibbles 4, 5, 6, 7
 *   localmem ncrisc                        -> sel=5, nibble 0
 *   iram brisc, trisc0, trisc1, trisc2, ncrisc -> sel=5, nibbles 1, 2, 3, 4, 5
 */
#define TENSIX_ECC_PAR_NIBBLE(s, i) FIELD_GET(GENMASK(3, 0), (s) >> (4U * (i)))
#define TENSIX_ECC_PAR_CNT(n)       ((n) & GENMASK(2, 0))
#define TENSIX_ECC_PAR_OVF(n)       (((n) >> 3U) & 1U)

/**
 * @brief Read one view of a Tensix tile's ECC_STATUS register.
 *
 * Writes @p sel into ECC_CTRL[3:0] and reads back ECC_STATUS. The write is a
 * read-modify-write that preserves ECC_CTRL[31:4] (the IRQ enables) and never sets the
 * level-sensitive clear bits [4] and [5], so this is non-destructive to other observers.
 *
 * The caller must ensure Tensix is powered - a NOC read to an unclocked tile can hang the
 * ARC. The caller should also confirm liveness once (@ref TENSIX_ECC_SEL_LIVENESS returns
 * @ref TENSIX_ECC_LIVENESS_MAGIC) before trusting a zero from any other view, since a
 * healthy tile and a failed read both read as zero.
 *
 * @param noc_x NOC 0 X coordinate of the Tensix tile.
 * @param noc_y NOC 0 Y coordinate of the Tensix tile.
 * @param sel   Which view to select, one of TENSIX_ECC_SEL_*.
 *
 * @return The raw ECC_STATUS value for that view.
 */
uint32_t TensixEccReadStatus(uint8_t noc_x, uint8_t noc_y, uint8_t sel);

/**
 * @brief Enable or disable the NOC ECC error outputs of a tile's NIU.
 *
 * Sets or clears NIU_CFG_0[11:9] on the tile's NOC 0 NIU, which gates whether a non-zero NIU
 * ECC error counter shows up in that tile's ECC_STATUS sel=4 (@ref TENSIX_ECC_SEL_NOC). This
 * does not enable ECC checking itself - that is a separate chip-wide two-phase sequence - and
 * it does not affect the NIU counters, which count regardless.
 *
 * The write is a read-modify-write: NIU_CFG_0 also carries TILE_CLK_OFF, the coordinate
 * translation enable and the command buffer enable, so a blind write would take the tile down.
 *
 * Unlike the debug registers, the NIU stays reachable while the tile clock is gated.
 *
 * @param noc_x  NOC 0 X coordinate of the Tensix tile.
 * @param noc_y  NOC 0 Y coordinate of the Tensix tile.
 * @param enable True to open the gate, false to close it.
 *
 * @return 0 on success, -EIO if the value does not read back as written.
 */
int TensixNocEccIntEnable(uint8_t noc_x, uint8_t noc_y, bool enable);

/**
 * @brief Read back which of a tile's NOC ECC error outputs are gated on.
 *
 * A closed gate and a healthy tile both report zero in ECC_STATUS sel=4, so anything that
 * presents those bits has to say which one it is - otherwise a chip nobody enabled reads as
 * permanently clean.
 *
 * @param noc_x NOC 0 X coordinate of the Tensix tile.
 * @param noc_y NOC 0 Y coordinate of the Tensix tile.
 *
 * @return NIU_CFG_0[11:9] shifted down, so it indexes by NOC_ECC_MEM_PARITY / _HDR_SBE /
 *         _HDR_DBE. Zero means no source can report.
 */
uint8_t TensixNocEccIntEnabled(uint8_t noc_x, uint8_t noc_y);

/**
 * @brief Read a tile's three NOC NIU ECC error counters.
 *
 * These count regardless of NIU_CFG_0[11:9], so they are the only way to tell a force that
 * never landed from a gate that never opened. Since the NIU ECC_CTRL is write-only, they are
 * also the only confirmation that @ref TensixNocEccForce did anything.
 *
 * @param noc_x NOC 0 X coordinate of the Tensix tile.
 * @param noc_y NOC 0 Y coordinate of the Tensix tile.
 * @param out   Receives the counts, indexed by NOC_ECC_MEM_PARITY / _HDR_SBE / _HDR_DBE.
 */
void TensixNocEccReadCounters(uint8_t noc_x, uint8_t noc_y, uint32_t out[NOC_ECC_NUM_SOURCES]);

/**
 * @brief Force one ECC error per selected source on a tile's NOC NIU.
 *
 * Each selected source's counter increments by one. The force path is unconditional in RTL -
 * it does not depend on ECC checking being enabled, so this works on a part where the chip-wide
 * two-phase ECC enable sequence has never been run.
 *
 * @param noc_x NOC 0 X coordinate of the Tensix tile.
 * @param noc_y NOC 0 Y coordinate of the Tensix tile.
 * @param which Bitmask of NOC_ECC_MEM_PARITY / _HDR_SBE / _HDR_DBE bit positions.
 */
void TensixNocEccForce(uint8_t noc_x, uint8_t noc_y, uint8_t which);

/**
 * @brief Zero the selected NOC NIU ECC error counters on a tile.
 *
 * A cleared counter also drops the corresponding ECC_STATUS sel=4 bit, since that bit is just
 * "counter is non-zero" ANDed with the enable.
 *
 * @param noc_x NOC 0 X coordinate of the Tensix tile.
 * @param noc_y NOC 0 Y coordinate of the Tensix tile.
 * @param which Bitmask of NOC_ECC_MEM_PARITY / _HDR_SBE / _HDR_DBE bit positions.
 */
void TensixNocEccClear(uint8_t noc_x, uint8_t noc_y, uint8_t which);

#endif /* TENSIX_ECC_H */
