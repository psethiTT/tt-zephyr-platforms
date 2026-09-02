/*
 * Copyright (c) 2025 Tenstorrent AI ULC
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/logging/log.h>
#include <zephyr/shell/shell.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#include <tenstorrent/bh_power.h>
#ifdef CONFIG_TT_MSGQUEUE
#include <tenstorrent/msgqueue.h>
#endif

#include "telemetry.h"
#include "smbus_target.h"
#include "gddr.h"
#include "asic_state.h"
#include "noc.h"
#include "noc_init.h"
#include "tensix_ecc.h"

LOG_MODULE_REGISTER(tt_shell, CONFIG_LOG_DEFAULT_LEVEL);

static int parse_u32_arg(const char *arg, uint32_t *value)
{
	char *endptr;
	unsigned long parsed;

	errno = 0;
	parsed = strtoul(arg, &endptr, 0);
	if (errno != 0 || endptr == arg || *endptr != '\0' || parsed > UINT32_MAX) {
		return -EINVAL;
	}

	*value = (uint32_t)parsed;
	return 0;
}

static int l2cpu_enable_handler(const struct shell *sh, size_t argc, char **argv)
{
	bool on = false;

	if (strcmp(argv[1], "off") == 0) {
		on = false;

	} else if (strcmp(argv[1], "on") == 0) {
		on = true;
	} else {
		shell_error(sh, "Invalid L2CPU power setting");

		return -EINVAL;
	}

	int ret = bh_set_l2cpu_enable(on);

	if (ret != 0) {
		shell_error(sh, "Failure to set L2CPU power setting %u", on);
		return ret;
	}
	shell_print(sh, "OK");
	return 0;
}

static int tensix_enable_handler(const struct shell *sh, size_t argc, char **argv)
{
	bool on = false;

	if (strcmp(argv[1], "off") == 0) {
		on = false;

	} else if (strcmp(argv[1], "on") == 0) {
		on = true;
	} else {
		shell_error(sh, "Invalid tensix power setting");

		return -EINVAL;
	}

	int ret = set_tensix_enable(on);

	if (ret != 0) {
		shell_error(sh, "Failure to set tensix power setting %u", on);
		return ret;
	}
	shell_print(sh, "OK");
	return 0;
}

static int mrisc_power_handler(const struct shell *sh, size_t argc, char **argv)
{
	bool on = false;

	if (strcmp(argv[1], "off") == 0) {
		on = false;

	} else if (strcmp(argv[1], "on") == 0) {
		on = true;
	} else {
		shell_error(sh, "Invalid MRISC power setting");

		return -EINVAL;
	}

	int ret = set_mrisc_power_setting(on);

	if (ret != 0) {
		shell_error(sh, "Failure to set MRISC power setting %u", on);
		return ret;
	}
	shell_print(sh, "OK");
	return 0;
}

static int asic_state_handler(const struct shell *sh, size_t argc, char **argv)
{
	if (argc == 2U) {
		AsicState state = (AsicState)atoi(argv[1]);

		if (state == A0State || state == A3State) {
			set_asic_state(state);
			shell_print(sh, "OK");
		} else {
			shell_error(sh, "Invalid ASIC State");
			return -EINVAL;
		}
	} else {
		shell_print(sh, "ASIC State: %u", get_asic_state());
	}

	return 0;
}

static int telem_handler(const struct shell *sh, size_t argc, char **argv)
{
	int32_t idx = atoi(argv[1]);
	char fmt;
	uint32_t value;

	if (argc == 3 && (strlen(argv[2]) != 1U)) {
		shell_error(sh, "Invalid format");
		return -EINVAL;
	}
	if (argc == 2) {
		fmt = 'x';
	} else {
		fmt = argv[2][0];
	}

	if (!GetTelemetryTagValid(idx)) {
		shell_error(sh, "Invalid telemetry tag");
		return -EINVAL;
	}

	value = GetTelemetryTag(idx);

	if (fmt == 'x') {
		shell_print(sh, "0x%08X", value);
	} else if (fmt == 'f') {
		shell_print(sh, "%lf", (double)ConvertTelemetryToFloat(value));
	} else if (fmt == 'd') {
		shell_print(sh, "%d", value);
	} else {
		shell_error(sh, "Invalid format");
		return -EINVAL;
	}

	return 0;
}

#ifdef CONFIG_TT_MSGQUEUE
static int msg_handler(const struct shell *sh, size_t argc, char **argv)
{
	union request request = {0};
	struct response response = {0};
	uint32_t parsed;
	int ret;

	for (size_t i = 1; i < argc; ++i) {
		ret = parse_u32_arg(argv[i], &parsed);
		if (ret != 0) {
			shell_error(sh, "Invalid u32 value: %s", argv[i]);
			return ret;
		}

		request.data[i - 1U] = parsed;
	}

	ret = msgqueue_request_push(0, &request);
	if (ret != 0) {
		shell_error(sh, "Failed to queue request (%d)", ret);
		return ret;
	}

	process_message_queues();

	ret = msgqueue_response_pop(0, &response);
	if (ret != 0) {
		shell_error(sh, "Failed to read response (%d)", ret);
		return ret;
	}

	for (size_t i = 0; i < RESPONSE_MSG_LEN; ++i) {
		shell_print(sh, "rsp[%u] = 0x%08x", (unsigned int)i, response.data[i]);
	}

	return 0;
}
#endif

/*
 * Prints one row of five per-core parity counters. word[]/nibble[] are parallel arrays in
 * brisc, trisc0, trisc1, trisc2, ncrisc order, since the counters are not all in the same view.
 */
static void ecc_print_parity(const struct shell *sh, const char *label, const uint32_t word[5],
			     const uint8_t nibble[5])
{
	uint32_t counts[5];
	uint32_t ovfs[5];

	for (size_t i = 0; i < ARRAY_SIZE(counts); ++i) {
		uint32_t n = TENSIX_ECC_PAR_NIBBLE(word[i], nibble[i]);

		counts[i] = TENSIX_ECC_PAR_CNT(n);
		ovfs[i] = TENSIX_ECC_PAR_OVF(n);
	}

	shell_print(sh, "%s: b/t0/t1/t2/nc = %u/%u/%u/%u/%u  ovf = %u/%u/%u/%u/%u", label,
		    counts[0], counts[1], counts[2], counts[3], counts[4], ovfs[0], ovfs[1],
		    ovfs[2], ovfs[3], ovfs[4]);
}

/*
 * Optional trailing "<noc_x> <noc_y>" pair shared by the ecc commands, starting at argv[first].
 * Supplying neither picks an arbitrary enabled tile; supplying one is a typo, not a default.
 */
static int parse_noc_coord_args(const struct shell *sh, size_t argc, char **argv, size_t first,
				uint8_t *noc_x, uint8_t *noc_y)
{
	uint32_t parsed_x, parsed_y;

	if (argc == first) {
		GetEnabledTensix(noc_x, noc_y);
		return 0;
	}

	if (argc != first + 2) {
		shell_error(sh, "Provide both noc_x and noc_y, or neither");
		return -EINVAL;
	}

	if (parse_u32_arg(argv[first], &parsed_x) != 0 ||
	    parse_u32_arg(argv[first + 1], &parsed_y) != 0) {
		shell_error(sh, "Invalid NOC coordinate");
		return -EINVAL;
	}
	if (parsed_x >= NOC_X_SIZE || parsed_y >= NOC_Y_SIZE) {
		shell_error(sh, "NOC coordinate out of range (x < %u, y < %u)", NOC_X_SIZE,
			    NOC_Y_SIZE);
		return -EINVAL;
	}

	*noc_x = parsed_x;
	*noc_y = parsed_y;
	return 0;
}

static int ecc_int_handler(const struct shell *sh, size_t argc, char **argv)
{
	uint8_t noc_x, noc_y;
	bool on;
	int ret;

	if (strcmp(argv[1], "on") == 0) {
		on = true;
	} else if (strcmp(argv[1], "off") == 0) {
		on = false;
	} else {
		shell_error(sh, "Invalid setting; expected 'on' or 'off'");
		return -EINVAL;
	}

	ret = parse_noc_coord_args(sh, argc, argv, 2, &noc_x, &noc_y);
	if (ret != 0) {
		return ret;
	}

	/*
	 * No clock gate check here, unlike 'tt ecc'. This touches the NIU rather than the debug
	 * registers behind the tile clock, so it works on a gated tile.
	 */
	ret = TensixNocEccIntEnable(noc_x, noc_y, on);
	if (ret != 0) {
		shell_error(sh, "NIU_CFG_0 did not read back as written on tile (%u, %u)", noc_x,
			    noc_y);
		return ret;
	}

	shell_print(sh, "tile: noc0 (%u, %u)   NIU_CFG_0[11:9] %s", noc_x, noc_y,
		    on ? "set" : "cleared");
	return 0;
}

/*
 * ECC_STATUS sel=4 bit @p source, qualified by whether that source is gated on at all. The
 * sel=4 bits sit at the same positions as the NOC_ECC_* indices, hence the shared BIT().
 */
static const char *ecc_noc_flag_str(uint32_t status, uint8_t gate, uint8_t source)
{
	if ((gate & BIT(source)) == 0) {
		return "n/a";
	}

	return (status & BIT(source)) ? "yes" : "no";
}

static void ecc_print_counters(const struct shell *sh, const char *label,
			       const uint32_t counters[NOC_ECC_NUM_SOURCES])
{
	shell_print(sh, "%-7s mem_parity=%u hdr_sbe=%u hdr_dbe=%u", label,
		    counters[NOC_ECC_MEM_PARITY], counters[NOC_ECC_HDR_SBE],
		    counters[NOC_ECC_HDR_DBE]);
}

static int ecc_force_handler(const struct shell *sh, size_t argc, char **argv)
{
	uint32_t before[NOC_ECC_NUM_SOURCES];
	uint32_t after[NOC_ECC_NUM_SOURCES];
	uint8_t noc_x, noc_y;
	uint32_t which = BIT(NOC_ECC_MEM_PARITY);
	size_t coord_arg;
	int ret;

	/*
	 * Both the mask and the coordinate pair are optional, which makes "ecc_force 1 2"
	 * ambiguous. Resolve it on argument count alone: a trailing pair is always coordinates,
	 * matching every other ecc command, so a mask can only appear alongside both of them.
	 */
	coord_arg = (argc == 2 || argc == 4) ? 2 : 1;

	if (coord_arg == 2) {
		if (parse_u32_arg(argv[1], &which) != 0 || which == 0 ||
		    (which & ~NOC_ECC_SOURCE_MASK) != 0) {
			shell_error(sh, "Mask must be 1..%u (bit0 mem_parity, bit1 hdr_sbe, "
					"bit2 hdr_dbe)",
				    (unsigned int)NOC_ECC_SOURCE_MASK);
			return -EINVAL;
		}
	}

	ret = parse_noc_coord_args(sh, argc, argv, coord_arg, &noc_x, &noc_y);
	if (ret != 0) {
		return ret;
	}

	/*
	 * All three registers live in the NIU, so no clock gate check - see ecc_int_handler. The
	 * counters are the only evidence the force landed: NIU ECC_CTRL is write-only.
	 */
	TensixNocEccReadCounters(noc_x, noc_y, before);
	TensixNocEccForce(noc_x, noc_y, which);
	TensixNocEccReadCounters(noc_x, noc_y, after);

	shell_print(sh, "tile: noc0 (%u, %u)   forced mask 0x%X", noc_x, noc_y, which);
	ecc_print_counters(sh, "before:", before);
	ecc_print_counters(sh, "after:", after);

	for (int i = 0; i < NOC_ECC_NUM_SOURCES; i++) {
		if ((which & BIT(i)) != 0 && after[i] == before[i]) {
			shell_warn(sh, "Counter %d did not move; the force did not reach the NIU",
				   i);
		}
	}

	shell_print(sh, "Run 'tt ecc_int on' then 'tt ecc' to see this in ECC_STATUS sel=4");
	return 0;
}

static int ecc_clear_handler(const struct shell *sh, size_t argc, char **argv)
{
	uint32_t counters[NOC_ECC_NUM_SOURCES];
	uint8_t noc_x, noc_y;
	int ret;

	ret = parse_noc_coord_args(sh, argc, argv, 1, &noc_x, &noc_y);
	if (ret != 0) {
		return ret;
	}

	TensixNocEccClear(noc_x, noc_y, NOC_ECC_SOURCE_MASK);
	TensixNocEccReadCounters(noc_x, noc_y, counters);

	shell_print(sh, "tile: noc0 (%u, %u)   cleared", noc_x, noc_y);
	ecc_print_counters(sh, "now:", counters);
	return 0;
}

static int ecc_handler(const struct shell *sh, size_t argc, char **argv)
{
	/* Nibble index within the owning view; see the parity note in tensix_ecc.h. */
	static const uint8_t localmem_nibble[5] = {4, 5, 6, 7, 0};
	static const uint8_t iram_nibble[5] = {1, 2, 3, 4, 5};

	uint32_t raw[TENSIX_ECC_SEL_LAST + 1] = {0};
	uint8_t noc_x, noc_y;
	uint32_t liveness;
	uint8_t gate;
	int ret;

	ret = parse_noc_coord_args(sh, argc, argv, 1, &noc_x, &noc_y);
	if (ret != 0) {
		return ret;
	}

	/*
	 * The debug registers sit behind the tile clock, so a gated tile cannot answer. Read the
	 * gate from the tile's own NIU, which stays reachable either way.
	 */
	if (IsSingleTileClockGated(noc_x, noc_y)) {
		shell_error(sh, "Tile (%u, %u) is clock gated; run 'tt tensix_power on' first",
			    noc_x, noc_y);
		return -ENODEV;
	}

	/*
	 * On a healthy part every view reads zero, which is indistinguishable from a read that
	 * never landed. The liveness magic is the only positive proof the path works.
	 */
	liveness = TensixEccReadStatus(noc_x, noc_y, TENSIX_ECC_SEL_LIVENESS);
	shell_print(sh, "tile: noc0 (%u, %u)   liveness: 0x%08X %s", noc_x, noc_y, liveness,
		    liveness == TENSIX_ECC_LIVENESS_MAGIC ? "OK" : "BAD");
	if (liveness != TENSIX_ECC_LIVENESS_MAGIC) {
		shell_error(sh, "Expected 0x%08X - ECC_STATUS is not readable on this tile, "
				"remaining values are meaningless",
			    TENSIX_ECC_LIVENESS_MAGIC);
		return -EIO;
	}

	for (uint8_t sel = TENSIX_ECC_SEL_FIRST; sel <= TENSIX_ECC_SEL_LAST; ++sel) {
		raw[sel] = TensixEccReadStatus(noc_x, noc_y, sel);
	}

	shell_print(sh, "raw:  sel1=0x%08X sel2=0x%08X sel3=0x%08X", raw[1], raw[2], raw[3]);
	shell_print(sh, "      sel4=0x%08X sel5=0x%08X sel6=0x%08X", raw[4], raw[5], raw[6]);

	shell_print(sh, "L1:   err_seen=%u sbe=%u (ovf %u) dbe=%u (ovf %u)",
		    TENSIX_ECC_L1_ERR_VALID(raw[TENSIX_ECC_SEL_LAST_L1_LO]),
		    TENSIX_ECC_L1_SBE_CNT(raw[TENSIX_ECC_SEL_L1_CNT]),
		    TENSIX_ECC_L1_SBE_OVF(raw[TENSIX_ECC_SEL_L1_CNT]),
		    TENSIX_ECC_L1_DBE_CNT(raw[TENSIX_ECC_SEL_L1_CNT]),
		    TENSIX_ECC_L1_DBE_OVF(raw[TENSIX_ECC_SEL_L1_CNT]));
	/*
	 * err_last_l1_bank/bank_addr are not in the reset list in tt_ecc_manager.sv - they are
	 * written only when an error is captured, so until err_seen is set they hold whatever
	 * the flops powered up as. Never report them unqualified.
	 */
	if (TENSIX_ECC_L1_ERR_VALID(raw[TENSIX_ECC_SEL_LAST_L1_LO])) {
		/* Bank one-hot spans both views: banks 0-15 in sel=3, banks 16-31 in sel=6. */
		shell_print(sh, "      bank=0x%08X addr=0x%03X",
			    TENSIX_ECC_L1_ERR_BANK(raw[TENSIX_ECC_SEL_LAST_L1_LO]) |
				    (TENSIX_ECC_L1_ERR_BANK(raw[TENSIX_ECC_SEL_LAST_L1_HI]) << 16),
			    TENSIX_ECC_L1_ERR_ADDR(raw[TENSIX_ECC_SEL_LAST_L1_LO]));
	} else {
		shell_print(sh, "      bank/addr not valid (no L1 error captured)");
	}

	/*
	 * Flags, not counts. Each bit is "NIU counter is non-zero" ANDed with its NIU_CFG_0[11:9]
	 * enable, so a tile with 900 errors reports the same 1 as a tile with one - the counts
	 * live in the NIU, behind 'tt ecc_force'. A gated-off source prints n/a rather than no,
	 * because a closed gate and a clean tile are otherwise the same output.
	 */
	gate = TensixNocEccIntEnabled(noc_x, noc_y);
	shell_print(sh, "NOC:  mem_parity=%s hdr_sbe=%s hdr_dbe=%s%s",
		    ecc_noc_flag_str(raw[TENSIX_ECC_SEL_NOC], gate, NOC_ECC_MEM_PARITY),
		    ecc_noc_flag_str(raw[TENSIX_ECC_SEL_NOC], gate, NOC_ECC_HDR_SBE),
		    ecc_noc_flag_str(raw[TENSIX_ECC_SEL_NOC], gate, NOC_ECC_HDR_DBE),
		    gate == NOC_ECC_SOURCE_MASK ? "" : "  (n/a: gate closed, 'tt ecc_int on')");

	/* localmem: brisc..trisc2 come from sel=1, ncrisc from sel=5. iram is all sel=5. */
	const uint32_t localmem_word[5] = {
		raw[TENSIX_ECC_SEL_COMBINED_CNT], raw[TENSIX_ECC_SEL_COMBINED_CNT],
		raw[TENSIX_ECC_SEL_COMBINED_CNT], raw[TENSIX_ECC_SEL_COMBINED_CNT],
		raw[TENSIX_ECC_SEL_IRAM_CNT],
	};
	const uint32_t iram_word[5] = {
		raw[TENSIX_ECC_SEL_IRAM_CNT], raw[TENSIX_ECC_SEL_IRAM_CNT],
		raw[TENSIX_ECC_SEL_IRAM_CNT], raw[TENSIX_ECC_SEL_IRAM_CNT],
		raw[TENSIX_ECC_SEL_IRAM_CNT],
	};

	ecc_print_parity(sh, "par ", localmem_word, localmem_nibble);
	ecc_print_parity(sh, "iram", iram_word, iram_nibble);

	return 0;
}

SHELL_STATIC_SUBCMD_SET_CREATE(
	sub_tt_commands, SHELL_CMD_ARG(mrisc_power, NULL, "[off|on]", mrisc_power_handler, 2, 0),
	SHELL_CMD_ARG(tensix_power, NULL, "[off|on]", tensix_enable_handler, 2, 0),
	SHELL_CMD_ARG(l2cpu_power, NULL, "[off|on]", l2cpu_enable_handler, 2, 0),
	SHELL_CMD_ARG(asic_state, NULL, "[|0|3]", asic_state_handler, 1, 1),
	SHELL_CMD_ARG(telem, NULL, "<Telemetry Index> [|x|f|d]", telem_handler, 2, 1),
	SHELL_CMD_ARG(ecc, NULL, "[<noc_x> <noc_y>]", ecc_handler, 1, 2),
	SHELL_CMD_ARG(ecc_int, NULL, "<off|on> [<noc_x> <noc_y>]", ecc_int_handler, 2, 2),
	SHELL_CMD_ARG(ecc_force, NULL, "[<mask>] [<noc_x> <noc_y>]", ecc_force_handler, 1, 3),
	SHELL_CMD_ARG(ecc_clear, NULL, "[<noc_x> <noc_y>]", ecc_clear_handler, 1, 2),
#ifdef CONFIG_TT_MSGQUEUE
	SHELL_CMD_ARG(msg, NULL, "<cmd> [data1 ... data7]", msg_handler, 2, 7),
#endif
	SHELL_SUBCMD_SET_END);

SHELL_CMD_REGISTER(tt, &sub_tt_commands, "Tensorrent commands", NULL);
