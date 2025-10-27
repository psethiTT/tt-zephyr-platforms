#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import argparse
from tabulate import tabulate

try:
    import pyluwen
except ImportError:
    print("Error, required modules missing. Please run 'pip install pyluwen'")
    sys.exit(1)

throttler_name = [
    "Fmax", "TDP", "FastTDC", "TDC", "Thm", "BoardPower", "Voltage", "GDDRThm"
]

RESET_UNIT_SCRATCH_RAM_BASE_ADDR = 0x80030400
THROTTLER_COUNT_BASE_REG_ADDR = RESET_UNIT_SCRATCH_RAM_BASE_ADDR + 4 * 22
NUM_THROTTLERS = 8

def read_throttler_counts(asic_id=0):
    """Reads scratch registers from the ARC"""
    try:
        chips = pyluwen.detect_chips()
        if len(chips) == 0:
            raise RuntimeError("No chips detected")

        chip = chips[asic_id]

        def read_scratch_ram(index):
            addr = THROTTLER_COUNT_BASE_REG_ADDR + (index * 4)
            return chip.axi_read32(addr)

        throttler_counts = {}
        for i in range(NUM_THROTTLERS):
            count = read_scratch_ram(i)
            throttler_counts[throttler_name[i]] = count
        return throttler_counts
    except Exception as e:
        print(f"Error reading throttler counts: {e}")
        return None

def tt_smi_reset():
    """Resets throttler counts"""
    try:
        result = subprocess.run(["tt-smi", "-r"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running tt-smi -r: \n{result.stderr}")
            raise RuntimeError("tt-smi reset failed")
        print("reset successfully")
    except FileNotFoundError:
        print("Error: tt-smi not found. Ensure it is installed and in PATH.")
        sys.exit(1)

def run_metal_test(command, timeout_min):
    """Runs tt-metal workload in Docker"""
    user = os.getenv("USER")
    llama_dir = f"/home/{user}/LLAMA_31_8B_INSTRUCT_DIR/"

    if not os.path.exists(llama_dir):
        print(f"Error: LLAMA directory {llama_dir} does not exist.")
        return
    
    docker_cmd = [
        "sudo", "-E", "docker", "run", "--rm",
        "-v", "/dev/hugepages-1G:/dev/hugepages-1G",
        "--device", "/dev/tenstorrent/0",
        "-v", f"{llama_dir}:{llama_dir}",
        "-e", f"LLAMA_DIR={llama_dir}",
        "--entrypoint", "/bin/bash",
        "ghcr.io/tenstorrent/tt-metal/upstream-tests-bh:v0.62.0-dev20251010-12-g23761277d0",
        "-c", (
            "source /opt/venv/bin/activate && "
            "pip3 install -r models/tt_transformers/requirements.txt && "
            f"{command}"
        )
    ]
    print(f"\nExecuting: {' '.join(docker_cmd)}")
    process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    time.sleep(timeout_min * 60)
    process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()

    print(f"\n--- STDOUT ---")
    print(stdout)
    print(f"--- END STDOUT ---")

    return stdout

def print_throttler_counts(counts):
    """Prints throttler counts in a table format"""
    if counts is None:
        print("No counts to display.")
        return 

    headers = ["Throttler", "Count"]
    table = [[name, counts.get(name, 0)] for name in throttler_name]
    print(f"\nThrottler counts before WL")
    print(tabulate(table, headers, tablefmt="grid"))

def print_throttler_delta(before, after):
    """Prints the difference in throttler counts before and after workload"""
    if before is None or after is None:
        print("Insufficient data to calculate delta.")
        return 

    headers = ["Throttler", "Before", "After", "Delta"]
    table = []
    for i in range(NUM_THROTTLERS):
        name = throttler_name[i]
        before_count = before[name]
        after_count = after[name]
        delta = after_count - before_count
        table.append([name, before_count, after_count, delta])
    
    print("\nThrottler Count Delta")
    print(tabulate(table, headers, tablefmt="grid"))

def main():

    parser = argparse.ArgumentParser(description="Run tt-metal test and read throttler counts")
    parser.add_argument("--command", default=None, help="Docker test command")
    parser.add_argument("--timeout", type=float, default=0, help="Timeout for each run in minutes")
    args = parser.parse_args()

    if args.command is None:
        args.command = input("Enter the tt-metal test command to run in Docker: ").strip()
        if not args.command:
            print("No command provided. Exiting.")
            sys.exit(1)

    if args.timeout <= 0:
        args.timeout = float(input("Enter the timeout per run in minutes (e.g., 5): ").strip() or "5")
        if args.timeout <= 0:
            print("Error: Timeout must be positive")
            sys.exit(1)

    before_counts = read_throttler_counts()
    print_throttler_counts(before_counts)

    print("\nRunning tt-metal workload...")
    run_metal_test(args.command, args.timeout)

    after_counts = read_throttler_counts()
    print_throttler_delta(before_counts, after_counts)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
