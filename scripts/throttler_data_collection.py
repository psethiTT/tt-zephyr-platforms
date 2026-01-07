#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import signal
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
    "Fmax", "TDP", "FastTDC", "TDC", "Thm", "BoardPower", "Voltage", "GDDRThm", "DopplerSlow", "DopplerCritical"
]

model_name = [
    "llama", "wan"
]

product_type = [
    "p150", "p300", "galaxy"
]

RESET_UNIT_SCRATCH_RAM_BASE_ADDR = 0x80030400
THROTTLER_COUNT_BASE_REG_ADDR = RESET_UNIT_SCRATCH_RAM_BASE_ADDR + 4 * 22
NUM_THROTTLERS = 10

BASE_LOG_DIR = os.path.expanduser("~/performance_analysis_logs/")
os.makedirs(BASE_LOG_DIR, exist_ok=True)

def read_throttler_counts(product_type):
    """Reads scratch registers from the ARC"""
    max_chips = 0
    if product_type == "p150":
        max_chips = 1
    elif product_type == "p300":
        max_chips = 2
    elif product_type == "galaxy":
        max_chips = 32
    else:
        print(f"Error: Unknown product type '{product_type}'")
        return None

    try:
        chips = pyluwen.detect_chips()
        if len(chips) != max_chips:
            print(f"Error: Expected {max_chips} chips for '{product_type}' but detected {len(chips)}.")
            return None

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

def build_docker_command(model: str, product_type: str, command: str, llama_dir: Optional[str] = None) -> List[str]:
    """Builds the Docker command based on the model and command"""
    if model == "llama":
        if product_type == "p150":
            image_name = "ghcr.io/tenstorrent/tt-metal/upstream-tests-bh:v0.62.0-dev20251010-12-g23761277d0"
        elif product_type == "p300":
            image_name = "ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-p300:v0.62.0-dev20251010-12-g23761277d0"
        elif product_type == "galaxy":
            image_name = "ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-galaxy:v0.62.0-dev20251010-12-g23761277d0"
        else:
            print(f"Error: Unknown product type '{product_type}'")
            sys.exit(1)

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
    elif model == "bursty":
        image_name = "tt-metallium-dev:bursty"

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
    telem_timeout = 45 if model == "wan" else 20

    llama_dir = None
    if model == "llama":
        llama_dir = f"/proj_syseng/LLAMA_31_8B_INSTRUCT_DIR/"
        if not os.path.exists(llama_dir):
            print(f"Error: {llama_dir} missing")
            sys.exit(1)

    run_dir = os.path.join(BASE_LOG_DIR, sanitize_folder_name(model))
    os.makedirs(run_dir, exist_ok=True)
    docker_log = os.path.join(run_dir, "docker_output.log")
    telem_csv = os.path.join(run_dir, "telemetry")

    print(f"\nConsole log: {run_dir}")
    print(f"  Docker output: {docker_log}")
    if arg in ["all", "read_telemetry"]:
        print(f"  Telemetry CSV: {telem_csv}")

    before_counts = None
    if arg in ["all", "read_throttler_count"]:
        before_counts = read_throttler_counts(product_type)

    docker_process = None
    #ipmi_power_proc = None
    telem_proc = None

    try:
        #power_script = os.path.expanduser("~/work/syseng/src/t6ifc/poll_ipmi_power.py")
        #ipmi_power_proc = subprocess.Popen(power_script, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        #print("Started IPMI power polling...")

        docker_cmd = build_docker_command(model, product_type, command, llama_dir)
        print(f"\nExecuting: {' '.join(docker_cmd)}")

        with open(docker_log, "w", buffering=1) as f:
            f.write(f"Command: {command}\n")
            docker_process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, 
            bufsize=1, universal_newlines=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))

            time.sleep(telem_timeout)

            telem_proc = None
            if arg in ["read_telemetry", "all"]:
                telem_script = os.path.expanduser("~/work/syseng/src/t6ifc/read_telem_pyluwen_2.0.py")
                if not os.path.isfile(telem_script):
                    print(f"Telemetry script not found: {telem_script}")
                    sys.exit(1)
            
                telem_cmd = _venv_python_cmd(telem_script,
                                    ["--product", product_type, "--delay", "0.001", "--csv", telem_csv])
                print(f"Starting telemetry logger (venv): {' '.join(telem_cmd)}")
                
                # Create a log file for telemetry script output
                telem_log = os.path.join(run_dir, "telemetry_script.log")
                telem_log_file = open(telem_log, "w", buffering=1)
                
                telem_proc = subprocess.Popen(telem_cmd,
                                    stdout=telem_log_file,
                                    stderr=subprocess.STDOUT)
                print(f"  Telemetry script log: {telem_log}")
                # Give the telemetry script time to start and detect chips
                time.sleep(2)

                # Check if the process is still running
                if telem_proc.poll() is not None:
                    telem_log_file.close()
                    print(f"WARNING: Telemetry script exited early! Check {telem_log} for errors")
                    telem_proc = None
            
            start_time = time.time()
            timeout_sec = (timeout_min * 60) - telem_timeout

            while True:
                if docker_process.poll() is not None:
                    break

                if time.time() - start_time > timeout_sec:
                    print(f"\nTimeout reached - terminating container.")
                    docker_process.terminate()
                    break
                
                line = docker_process.stdout.readline()
                if line:
                    line = line.rstrip()
                    print(line)
                    f.write(line + "\n")
                else:
                    time.sleep(0.01)

                remaining = docker_process.stdout.read()
                if remaining:
                    print(remaining, end="")
                    f.write(remaining)

    finally:
        if arg in ["all", "read_telemetry"] and telem_proc is not None:
            print("Stopping telemetry collection...")
            # Send SIGINT for graceful shutdown (telemetry script handles this)
            telem_proc.send_signal(signal.SIGINT)
            try:
                telem_proc.wait(timeout=10)
                print("Telemetry script stopped gracefully")
            except subprocess.TimeoutExpired:
                print("Telemetry script didn't respond to SIGINT, forcing termination")
                telem_proc.kill()

        # if ipmi_power_proc is not None:
        #     print("Stopping IPMI power polling...")
        #     ipmi_power_proc.terminate()  # SIGTERM
        #     try:
        #         ipmi_power_proc.wait(timeout=5)
        #         print("IPMI power polling stopped")
        #     except subprocess.TimeoutExpired:
        #         print("IPMI power polling did not exit, killing it")
        #         ipmi_power_proc.kill()

        if docker_process.poll() is None:
            docker_process.kill()

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
            orig = open(docker_log, "r").read()
            open(docker_log, "w").write(final_header + orig)
            print(f"Throttler delta header added: {final_header.strip()}")
        else:
            print("Telemetry CSV missing – delta not written")

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
    venv_dir = os.path.expanduser("~/work/syseng/src/t6ifc/.venv")
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
