#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import argparse
import re
import itertools
from tabulate import tabulate
from datetime import datetime
from typing import List, Optional


try:
    import pyluwen
except ImportError:
    print("Error, required modules missing. Please run 'pip install pyluwen'")
    sys.exit(1)

throttler_name = [
    "Fmax", "TDP", "FastTDC", "TDC", "Thm", "BoardPower", "Voltage", "GDDRThm"
]

model_name = [
    "llama", "wan"
]

product_type = [
    "p150", "p300", "galaxy"
]

RESET_UNIT_SCRATCH_RAM_BASE_ADDR = 0x80030400
THROTTLER_COUNT_BASE_REG_ADDR = RESET_UNIT_SCRATCH_RAM_BASE_ADDR + 4 * 22
NUM_THROTTLERS = 8

BASE_LOG_DIR = os.path.expanduser("~/llama_logs/")
os.makedirs(BASE_LOG_DIR, exist_ok=True)

def num_chips_for_product(product_type):
    if product_type == "p150":
        return 1
    elif product_type == "p300":
        return 2
    elif product_type == "galaxy":
        return 32
    else:
        print(f"Error: Unknown product type '{product_type}'")
        sys.exit(1)

def read_throttler_counts(product_type):
    """Reads scratch registers from the ARC"""
    max_chips = num_chips_for_product(product_type)

    try:
        chips = pyluwen.detect_chips()
        if len(chips) != max_chips:
            print(f"Error: Expected {max_chips} chips for '{product_type}' but detected {len(chips)}.")
            sys.exit(1)

        all_throttler_data = {}

        for asic_id, chip in enumerate(chips):
            def read_scratch_ram(index):
                addr = THROTTLER_COUNT_BASE_REG_ADDR + (index * 4)
                return chip.axi_read32(addr)

            throttler_counts = {}
            for i in range(NUM_THROTTLERS):
                count = read_scratch_ram(i)
                throttler_counts[throttler_name[i]] = count
            all_throttler_data[asic_id] = throttler_counts

        return all_throttler_data

    except Exception as e:
        print(f"Error reading throttler counts: {e}")
        return None

def tt_smi_reset():
    """Resets throttler counts"""
    try:
        # use tt-smi from the venv at ~/tt-smi
        tt_smi_path = os.path.expanduser("~/tt-smi/.venv/bin/tt-smi")
        if not os.path.exists(tt_smi_path):
            print(f"Error: tt-smi not found at {tt_smi_path}")
            print("Please ensure ~/tt-smi/.venv is set up correctly.")
            sys.exit(1)
        
        result = subprocess.run([tt_smi_path, "-r"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running tt-smi -r: \n{result.stderr}")
            raise RuntimeError("tt-smi reset failed")
        print("reset successfully")
    except FileNotFoundError:
        print("Error: tt-smi not found. Ensure it is installed and in PATH.")
        sys.exit(1)

def sanitize_folder_name(model_name: str) -> str:
    """
    Turn any pytest / simple_text_demo command into a short, clean, human-readable folder name.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^\w]", "", model_name.lower())[:30]

    return f"{safe_model}_{timestamp}"

def throttler_delta_header(asic_id: int, delta_counts: dict) -> str:
    """
    Return a CSV header line that contains the 8 throttler deltas.
    """
    vals = [delta_counts.get(name, 0) for name in throttler_name]
    return f"{{ASIC_{asic_id}}}:" + "{" + ",".join(map(str, vals)) + "}"

def build_docker_command(model: str, command: str, llama_dir: Optional[str] = None) -> List[str]:
    """Builds the Docker command based on the model and command"""
    if model == "llama":
        image_name = "ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-p300:v0.62.0-dev20251010-12-g23761277d0"

        cmd_sequence = (
            "set -e && "
            "source /opt/venv/bin/activate && "
            "pip3 install -r models/tt_transformers/requirements.txt && "
            "export PYTHONUNBUFFERED=1 && "
            "pytest -vv --color=yes " + command
        )

        docker_cmd = [
            "sudo", "-E", "docker", "run", "--rm", "--user", "root",
            "-v", "/dev/hugepages-1G:/dev/hugepages-1G",
            "--device", "/dev/tenstorrent:/dev/tenstorrent",
            "-v", f"{llama_dir}:{llama_dir}",
            "-e", f"LLAMA_DIR={llama_dir}",
            "--entrypoint", "",
            image_name,
            "/bin/bash", "-c", cmd_sequence
        ]
        return docker_cmd

    elif model == "wan":
        image_name = "tt-metalium-dev:new_build_for_tlb"

        cmd_sequence = (
            "set -e && "
            "source /opt/venv/bin/activate && "
            "cd /tt-metal && "
            "export PYTHONUNBUFFERED=1 && "
            "pytest -vv --color=yes " + command
        )

        docker_cmd = [
            "sudo", "-E", "docker", "run", "--rm",
            "--device", "/dev/tenstorrent:/dev/tenstorrent",
            "-v", "/dev/hugepages-1G:/dev/hugepages-1G",
            "--entrypoint", "/bin/bash",
            image_name,
            "-c", cmd_sequence
        ]
        return docker_cmd

    else:
        print(f"Error: Unknown model type '{model}'. Must be 'llama' or 'wan'.")
        sys.exit(1)

def run_metal_test(model, command, timeout_min, arg, product_type):
    """Runs tt-metal workload in Docker"""

    llama_dir = None

    if model == "llama":
        llama_dir = f"/proj_syseng/LLAMA_31_8B_INSTRUCT_DIR/"
        if not os.path.exists(llama_dir):
            print(f"Error: {llama_dir} missing")
            sys.exit(1)

    run_dir = os.path.join(BASE_LOG_DIR, sanitize_folder_name(model))
    os.makedirs(run_dir, exist_ok=True)
    docker_log = os.path.join(run_dir, "docker_output.log")
    telem_base_path = os.path.join(run_dir, "telemetry")

    print(f"\nConsole log: {run_dir}")
    print(f"  Docker output: {docker_log}")
    if arg in ["all", "read_telemetry"]:
        print(f"  Telemetry CSV: {telem_base_path}_[0-N].csv")

    before_counts = None
    if arg in ["all", "read_throttler_count"]:
        before_counts = read_throttler_counts(product_type)
        num_chips = len(before_counts.keys())
    else:
        num_chips = num_chips_for_product(product_type)

    docker_cmd = build_docker_command(model, command, llama_dir)
    print(f"\nExecuting: {' '.join(docker_cmd)}")
    
    with open(docker_log, "w", buffering=1) as f:
        f.write(f"Command: {command}\n")
        process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, 
        bufsize=1, universal_newlines=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))

        time.sleep(20)

        telem_procs = []
        if arg in ["read_telemetry", "all"]:
            telem_script = os.path.expanduser("~/work/syseng/src/t6ifc/read_telem_pyluwen.py")
            if not os.path.isfile(telem_script):
                print(f"Telemetry script not found: {telem_script}")
                sys.exit(1)

            for asic_id in range(1):
                telem_csv = f"{telem_base_path}_{asic_id}.csv"
                telem_cmd = _venv_python_cmd(telem_script,
                                    ["--delay", "0.01", "--csv", f"{telem_csv}", "--chip-id", f"{asic_id}"])
                print(f"Starting telemetry logger for chip {asic_id} (venv): {' '.join(telem_cmd)}")
                proc = subprocess.Popen(telem_cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
                telem_procs.append(proc)

        start_time = time.time()
        timeout_sec = (timeout_min * 60) - 20

        try:
            while True:
                if process.poll() is not None:
                    break

                if time.time() - start_time > timeout_sec:
                    print(f"\nTimeout reached - terminating container.")
                    process.terminate()
                    break
                
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    print(line)
                    f.write(line + "\n")
                else:
                    time.sleep(0.01)

            remaining = process.stdout.read()
            if remaining:
                print(remaining, end="")
                f.write(remaining)

        except Exception as e:
            print(f"Error while reading Docker output: {e}")

        finally:
            if process.poll() is None:
                process.kill()

    if arg in ["all", "read_throttler_count"] and before_counts is not None:
        after_counts = read_throttler_counts(product_type)
        delta_lines = []

        chip_ids = sorted(list(set(before_counts.keys()) | set(after_counts.keys())))
        for asic_id in chip_ids:
            if asic_id not in before_counts or asic_id not in after_counts:
                continue
            before = before_counts[asic_id]
            after = after_counts[asic_id]
            delta = {name: after.get(name, 0) - before.get(name, 0) for name in throttler_name}
            header_line = throttler_delta_header(asic_id, delta) + "\n"
            delta_lines.append(header_line)
        final_header = "\n".join(delta_lines) + "\n"

        if os.path.exists(docker_log):
            log_path = docker_log
        else: 
            log_path = f"{telem_base_path}_0.csv"
        
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                orig = f.read()
            with open(log_path, "w") as f:
                f.write(final_header + orig)
            print(f"Throttler delta header added to {os.path.basename(log_path)}:\n{final_header.strip()}")
        else:
            print(f"({log_path}) missing – delta not written")
    
    if arg in ["all", "read_telemetry"] and telem_procs:
        for telem_proc in telem_procs:
            telem_proc.terminate()
            try:
                telem_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                 telem_proc.kill()
    
    print(f"\nFULL OUTPUT SAVED TO: {run_dir}")
    return run_dir

def print_throttler_counts(before, after=None):
    is_delta_mode = after is not None and bool(after)

    """Prints throttler counts in a table format"""
    if before is None or not before:
        print("no counts to display.")
        return

    if is_delta_mode:
        all_chip_ids = sorted(list(set(before.keys()) | set(after.keys())))
        headers = ["Throttler", "Before", "After", "Delta"]
        title_suffix = "COUNT DELTA ANALYSIS"
    else:
        all_chip_ids = sorted(before.keys())
        headers = ["Throttler", "Count"]
        title_suffix = "CURRENT THROTTLER COUNTS"

    for asic_id in all_chip_ids:
        before_counts = before.get(asic_id)
        if before_counts is None:
            continue
        if is_delta_mode:
            after_counts = after.get(asic_id)
            if after_counts is None:
                print(f"\nChip {asic_id}: Data is missing or corrupted.")
                continue

        table = []
        for name in throttler_name:
            b = before_counts.get(name, 0)
            row = [name, b]
            if is_delta_mode:
                a = after_counts.get(name, 0)
                delta = a - b
                row.extend([a, delta])
            table.append(row)

        print("\n" + "=" * 60)
        print(f"       ASIC ID {asic_id}: {title_suffix}       ")
        print("=" * 60)
        print(tabulate(table, headers, tablefmt="grid"))
        print("\n" + "-" * 60)

def _venv_python_cmd(script_path, extra_args=None):
    """
    Returns a list that runs `script_path` inside the venv at
    ~/work/syseng/src/t6ifc/venv
    """
    venv_dir = os.path.expanduser("~/work/syseng/src/t6ifc/venv")
    python   = os.path.join(venv_dir, "bin", "python3")
    cmd = [python, script_path]
    if extra_args:
        cmd.extend(extra_args)
    return cmd

def main():

    parser = argparse.ArgumentParser(description="Run tt-metal test and read telemetry + throttler counts")
    parser.add_argument("-t", "--test", default="all", choices=["all", "read_throttler_count", "read_telemetry", "get_throttler_count"], help="Test to run: all, read_throttler_count, read_telemetry")
    parser.add_argument("--command", default=None, help="Docker test command")
    parser.add_argument("--timeout", type=float, default=0, help="Timeout in minutes")
    parser.add_argument("--model", default=None, choices=model_name, help="Model name (eg. llama, wan)")
    parser.add_argument("--product_type", default="p150", choices=product_type, help="Product type (eg. p150, p300, galaxy)")
    args = parser.parse_args()

    if args.test == "get_throttler_count":
        counts = read_throttler_counts(args.product_type)
        print_throttler_counts(counts)
        sys.exit(0)

    if args.model is None:
        args.model = input(f"Enter the model name ({', '.join(model_name)}): ").strip()
        args.model = args.model.lower()
        if args.model not in model_name:
            print("Invalid model name")
            sys.exit(1)

    if args.command is None:
        args.command = input("Enter the tt-metal test command to run in Docker: ").strip()
        if not args.command:
            print("No command provided. Exiting.")
            sys.exit(1)
    
    if args.timeout <= 0:
        val = input("Enter the timeout in minutes (e.g. 5): ").strip() or "5"
        try:
            args.timeout = float(val)
        except ValueError:
            print("Invalid timeout")
            sys.exit(1)

    before_counts = None
    if args.test in ["all", "read_throttler_count"]:
        before_counts = read_throttler_counts(args.product_type)
        print_throttler_counts(before_counts)

    print("\nRunning tt-metal workload...")
    run_dir = run_metal_test(args.model, args.command, args.timeout, args.test, args.product_type)

    after_count = None
    if args.test in ["all", "read_throttler_count"]:
        after_counts = read_throttler_counts(args.product_type)
        print_throttler_counts(before_counts, after_counts)

    tt_smi_reset()
    print(f"\nAll files are in:\n  {run_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
