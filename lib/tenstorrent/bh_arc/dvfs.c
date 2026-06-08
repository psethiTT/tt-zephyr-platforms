/*
 * Copyright (c) 2024 Tenstorrent AI ULC
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include "vf_curve.h"
#include "throttler.h"
#include "aiclk_ppm.h"
#include "voltage.h"

LOG_MODULE_REGISTER(dvfs, CONFIG_TT_APP_LOG_LEVEL);

bool dvfs_enabled;

/* Timing instrumentation: DVFSChange() runs ~1000x/sec, so we accumulate
 * cycle counts and emit a single min/max/avg summary once per window instead
 * of logging every call (which would flood the UART and distort the result).
 */
#define DVFS_TIMING_WINDOW 2000 /* calls per summary, ~1s at the 1ms interval */

static struct {
	uint32_t count;
	uint32_t min_cyc;
	uint32_t max_cyc;
	uint64_t sum_cyc;
} dvfs_timing = {.min_cyc = UINT32_MAX};

void DVFSChange(void)
{
	uint32_t start = k_cycle_get_32();

	CalculateThrottlers();
	CalculateTargAiclk();

	uint32_t targ_freq = GetAiclkTarg();
	uint32_t aiclk_voltage = VFCurve(targ_freq);

	VoltageArbRequest(VoltageReqAiclk, aiclk_voltage);

	CalculateTargVoltage();

	DecreaseAiclk();
	VoltageChange();
	IncreaseAiclk();

	/* Cycle counter is free-running; subtraction is correct across a 32-bit
	 * wrap as long as the interval is < 2^32 cycles (~5.3s at 800MHz).
	 */
	uint32_t cycle_time = k_cycle_get_32() - start;

	if (dvfs_timing.count < DVFS_TIMING_WINDOW) {
		dvfs_timing.sum_cyc += cycle_time;
		dvfs_timing.min_cyc = MIN(dvfs_timing.min_cyc, cycle_time);
		dvfs_timing.max_cyc = MAX(dvfs_timing.max_cyc, cycle_time);

		if(k_cyc_to_ns_floor64(cycle_time) > 250000) {
			LOG_INF("DVFSChange at %u count: time taken=%lluns",
				dvfs_timing.count,
				k_cyc_to_ns_floor64(cycle_time));
		}
		if (dvfs_timing.count == DVFS_TIMING_WINDOW - 1) {
			uint32_t avg_cyc = dvfs_timing.sum_cyc / dvfs_timing.count;
			LOG_INF("DVFSChange over %u calls: min=%lluns avg=%lluns max=%lluns",
				dvfs_timing.count,
				k_cyc_to_ns_floor64(dvfs_timing.min_cyc),
				k_cyc_to_ns_floor64(avg_cyc),
				k_cyc_to_ns_floor64(dvfs_timing.max_cyc));
		}
		dvfs_timing.count++;
	}	
}

static void dvfs_work_handler(struct k_work *work)
{
	DVFSChange();
}
static K_WORK_DEFINE(dvfs_worker, dvfs_work_handler);

static void dvfs_timer_handler(struct k_timer *timer)
{
	k_work_submit(&dvfs_worker);
}
static K_TIMER_DEFINE(dvfs_timer, dvfs_timer_handler, NULL);

void InitDVFS(void)
{
	InitVFCurve();
	InitVoltagePPM();
	InitArbMaxVoltage();
	InitThrottlers();
	dvfs_enabled = true;
}

#define DVFS_MSEC 1

void StartDVFSTimer(void)
{
	k_timer_start(&dvfs_timer, K_MSEC(DVFS_MSEC), K_MSEC(DVFS_MSEC));
}

#define DVFS_TICKS (CONFIG_SYS_CLOCK_TICKS_PER_SEC * DVFS_MSEC / MSEC_PER_SEC)

/* If DVFS is already scheduled "close enough" to the board power message, then don't try to adjust
 * it. There may be some jitter in the message arrival and we don't want to suddenly go from being
 * very close to very far away. 10% is arbitrary.
 */
#define DVFS_ADJUSTMENT_THRESHOLD (DVFS_TICKS * 10 / 100) /* 10% of DVFS interval */

/* DVFS's PID controllers assume they are run on a 1ms interval. Changing the interval implicitly
 * changes their behaviour. 1% should be small enough to not cause trouble.
 */
#define DVFS_ADJUSTMENT_STEP (DVFS_TICKS * 1 / 100) /* 1% of DVFS interval */

void AdjustDVFSTimer(void)
{
	/* We just received a board power update from the DMC. If DVFS is still more than 10% of
	 * its interval away, then reduce that time by 1%. Over enough cycles, this should bring
	 * the DMC->DVFS latency down.
	 */
	if (dvfs_enabled) {
		k_ticks_t dvfs_remaining = k_timer_remaining_ticks(&dvfs_timer);

		if (dvfs_remaining > DVFS_ADJUSTMENT_THRESHOLD) {
			k_timeout_t delay = K_TICKS(dvfs_remaining - DVFS_ADJUSTMENT_STEP);

			k_timer_start(&dvfs_timer, delay, K_MSEC(DVFS_MSEC));
		}
	}
}
