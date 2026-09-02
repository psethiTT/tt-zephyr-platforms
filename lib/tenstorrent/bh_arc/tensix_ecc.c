/*
 * Copyright (c) 2026 Tenstorrent AI ULC
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensix_ecc.h"
#include "noc2axi.h"

#include <errno.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>

/* TLB 15 is the only NOC TLB index not already claimed by another block. */
#define TENSIX_ECC_TLB  15
#define TENSIX_ECC_RING 0

/* ECC_CTRL[3:0] view select. */
#define ECC_CTRL_SEL_MASK GENMASK(3, 0)

/*
 * ECC_CTRL[4] holds the counters in reset and [5] holds the IRQ in reset. Both are
 * level-sensitive, not write-1-to-clear: leaving [4] set makes every counter read zero
 * forever, so a faulty part looks perfectly healthy. Always mask them out of a write.
 */
#define ECC_CTRL_STATUS_CLEAR BIT(4)
#define ECC_CTRL_IRQ_CLEAR    BIT(5)

/* Guards both the shared TLB and the non-atomic select-then-read sequence below. */
static K_MUTEX_DEFINE(tensix_ecc_lock);

uint32_t TensixEccReadStatus(uint8_t noc_x, uint8_t noc_y, uint8_t sel)
{
	uint32_t ctrl;
	uint32_t status;

	k_mutex_lock(&tensix_ecc_lock, K_FOREVER);

	NOC2AXITlbSetup(TENSIX_ECC_RING, TENSIX_ECC_TLB, noc_x, noc_y, TENSIX_ECC_CTRL);

	/*
	 * Read-modify-write. ECC_CTRL[31:4] holds the per-source IRQ enables, so writing the
	 * select alone would disable every ECC interrupt in this tile.
	 */
	ctrl = NOC2AXIRead32(TENSIX_ECC_RING, TENSIX_ECC_TLB, TENSIX_ECC_CTRL);
	ctrl &= ~(ECC_CTRL_SEL_MASK | ECC_CTRL_STATUS_CLEAR | ECC_CTRL_IRQ_CLEAR);
	ctrl |= FIELD_PREP(ECC_CTRL_SEL_MASK, sel);
	NOC2AXIWrite32(TENSIX_ECC_RING, TENSIX_ECC_TLB, TENSIX_ECC_CTRL, ctrl);

	status = NOC2AXIRead32(TENSIX_ECC_RING, TENSIX_ECC_TLB, TENSIX_ECC_STATUS);

	k_mutex_unlock(&tensix_ecc_lock);

	return status;
}

int TensixNocEccIntEnable(uint8_t noc_x, uint8_t noc_y, bool enable)
{
	uint32_t cfg;
	uint32_t readback;

	k_mutex_lock(&tensix_ecc_lock, K_FOREVER);

	NOC2AXITlbSetup(TENSIX_ECC_RING, TENSIX_ECC_TLB, noc_x, noc_y, NOC_NIU_CFG_0);

	/*
	 * Read-modify-write. NIU_CFG_0 also holds TILE_CLK_OFF, the coordinate translation
	 * enable and the command buffer enable - writing the ECC bits alone would gate the
	 * tile's clock and cut off ID translation.
	 */
	cfg = NOC2AXIRead32(TENSIX_ECC_RING, TENSIX_ECC_TLB, NOC_NIU_CFG_0);
	if (enable) {
		cfg |= NOC_NIU_CFG_0_ECC_INT_EN;
	} else {
		cfg &= ~NOC_NIU_CFG_0_ECC_INT_EN;
	}
	NOC2AXIWrite32(TENSIX_ECC_RING, TENSIX_ECC_TLB, NOC_NIU_CFG_0, cfg);

	/*
	 * NIU_CFG_0 is plain read/write storage, so a mismatch means the access never reached
	 * the NIU rather than a bit the hardware refused. Worth catching here: everything built
	 * on top of this reads as a healthy zero when the path is silently broken.
	 */
	readback = NOC2AXIRead32(TENSIX_ECC_RING, TENSIX_ECC_TLB, NOC_NIU_CFG_0);

	k_mutex_unlock(&tensix_ecc_lock);

	if ((readback & NOC_NIU_CFG_0_ECC_INT_EN) != (cfg & NOC_NIU_CFG_0_ECC_INT_EN)) {
		return -EIO;
	}

	return 0;
}

uint8_t TensixNocEccIntEnabled(uint8_t noc_x, uint8_t noc_y)
{
	uint32_t cfg;

	k_mutex_lock(&tensix_ecc_lock, K_FOREVER);

	NOC2AXITlbSetup(TENSIX_ECC_RING, TENSIX_ECC_TLB, noc_x, noc_y, NOC_NIU_CFG_0);
	cfg = NOC2AXIRead32(TENSIX_ECC_RING, TENSIX_ECC_TLB, NOC_NIU_CFG_0);

	k_mutex_unlock(&tensix_ecc_lock);

	/* FIELD_GET shifts [11:9] down to [2:0], so the result indexes as NOC_ECC_*. */
	return FIELD_GET(NOC_NIU_CFG_0_ECC_INT_EN, cfg);
}

void TensixNocEccReadCounters(uint8_t noc_x, uint8_t noc_y, uint32_t out[NOC_ECC_NUM_SOURCES])
{
	static const uint32_t counter_addr[NOC_ECC_NUM_SOURCES] = {
		[NOC_ECC_MEM_PARITY] = NOC_NIU_NUM_MEM_PARITY_ERR,
		[NOC_ECC_HDR_SBE] = NOC_NIU_NUM_HEADER_1B_ERR,
		[NOC_ECC_HDR_DBE] = NOC_NIU_NUM_HEADER_2B_ERR,
	};

	k_mutex_lock(&tensix_ecc_lock, K_FOREVER);

	NOC2AXITlbSetup(TENSIX_ECC_RING, TENSIX_ECC_TLB, noc_x, noc_y, NOC_NIU_NUM_MEM_PARITY_ERR);

	for (int i = 0; i < NOC_ECC_NUM_SOURCES; i++) {
		out[i] = NOC2AXIRead32(TENSIX_ECC_RING, TENSIX_ECC_TLB, counter_addr[i]);
	}

	k_mutex_unlock(&tensix_ecc_lock);
}

/* Single write to the write-only NIU ECC_CTRL. @p value must already be positioned. */
static void TensixNocEccCtrlWrite(uint8_t noc_x, uint8_t noc_y, uint32_t value)
{
	k_mutex_lock(&tensix_ecc_lock, K_FOREVER);

	NOC2AXITlbSetup(TENSIX_ECC_RING, TENSIX_ECC_TLB, noc_x, noc_y, NOC_NIU_ECC_CTRL);
	NOC2AXIWrite32(TENSIX_ECC_RING, TENSIX_ECC_TLB, NOC_NIU_ECC_CTRL, value);

	k_mutex_unlock(&tensix_ecc_lock);
}

void TensixNocEccForce(uint8_t noc_x, uint8_t noc_y, uint8_t which)
{
	TensixNocEccCtrlWrite(noc_x, noc_y,
			      FIELD_PREP(NOC_NIU_ECC_CTRL_FORCE, which & NOC_ECC_SOURCE_MASK));
}

void TensixNocEccClear(uint8_t noc_x, uint8_t noc_y, uint8_t which)
{
	TensixNocEccCtrlWrite(noc_x, noc_y,
			      FIELD_PREP(NOC_NIU_ECC_CTRL_CLEAR, which & NOC_ECC_SOURCE_MASK));
}
