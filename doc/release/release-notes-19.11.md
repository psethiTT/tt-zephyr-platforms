# v19.11.0

> This is a working draft for the up-coming 19.11.0 release.

We are pleased to announce the release of TT System Firmware version 19.11.0 🥳🎉.

Major enhancements with this release include:

## What's Changed


## Wormhole


## Blackhole

### PCIe

- For P300C boards, update PCIe speed to Gen4 to improve QB2 PCIe stability (SYS-4229).

### Stability Improvements

- Updated Blackhole ERISC FW to v1.11.0
  - Keep all ETH interrupt modes DISABLED to avoid jumping into the IVT on RISC0 kernel switchover (tt-metal#44188)
  - Enable MACPCS ECC cabability
  - Added MACPCS/SerDes reset-deassert timestamps to boot results
  - ETH msg PORT_STATUS_CLEAR: clear accumulated live status counts in L1

## Grendel


## Migration guide

An overview of required and recommended changes to make when migrating from the previous v19.10.0 release can be found in [19.11 Migration Guide](https://github.com/tenstorrent/tt-system-firmware/tree/main/doc/release/migration-guide-19.11.md).

## Full ChangeLog

The full ChangeLog from the previous v19.10.0 release can be found at the link below.

https://github.com/tenstorrent/tt-system-firmware/compare/v19.10.0...v19.11.0
