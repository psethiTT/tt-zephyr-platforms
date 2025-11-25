#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import argparse
import re
from tabulate import tabulate
from datetime import datetime

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

BASE_LOG_DIR = os.path.expanduser("~/llama_logs/")
os.makedirs(BASE_LOG_DIR, exist_ok=True)

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
        # Use tt-smi from the venv at ~/tt-smi
        tt_smi_path = os.path.expanduser("~/tt-smi/venv/bin/tt-smi")
        if not os.path.exists(tt_smi_path):
            print(f"Error: tt-smi not found at {tt_smi_path}")
            print("Please ensure ~/tt-smi/venv is set up correctly.")
            sys.exit(1)
        
        result = subprocess.run([tt_smi_path, "-r"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running tt-smi -r: \n{result.stderr}")
            raise RuntimeError("tt-smi reset failed")
        print("reset successfully")
    except FileNotFoundError:
        print("Error: tt-smi not found. Ensure it is installed and in PATH.")
        sys.exit(1)

def sanitize_folder_name(pytest_cmd: str) -> str:
    """
    Turn any pytest / simple_text_demo command into a short, clean, human-readable folder name.
    Handles CLI flags, python kwargs, -k tags, long-context files, etc.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    m = re.search(r"-k\s+([A-Za-z0-9_-]+)", pytest_cmd)
    if m:
        tag = m.group(1)
    else:
        parts = []

        # batch size
        b = re.search(r"--batch[-_]size\s+(\d+)|batch[-_]size\s+(\d+)", pytest_cmd, re.I)
        if b:
            parts.append(f"b{b.group(1) or b.group(2)}")

        # max_generated_tokens
        g = re.search(r"--max[-_]generated[-_]tokens\s+(\d+)", pytest_cmd, re.I)
        if g:
            parts.append(f"gen{g.group(1)}")

        # max_seq_len (CLI flag or max_seq_len(...) in python call)
        s = re.search(r"--max[-_]seq[-_]len\s+(\d+)|max_seq_len\s*\(\s*(\d+)", pytest_cmd, re.I)
        if s:
            val = int(s.group(1) or s.group(2))
            if val >= 120000:
                parts.append("128k")
            elif val >= 60000:
                parts.append("64k")
            elif val >= 30000:
                parts.append("32k")
            else:
                parts.append(f"{val//1024}k")

        # long-context prompt files
        if "long_context_128k" in pytest_cmd:
            parts.append("128k")
        elif "long_context_64k" in pytest_cmd:
            parts.append("64k")
        elif "long_context_32k" in pytest_cmd:
            parts.append("32k")

        # paged attention + block size
        if re.search(r"--paged[-_]attention|paged_attention\s*=\s*True", pytest_cmd, re.I):
            blk = re.search(r"--page[-_]block[-_]size\s+(\d+)|page[-_]block[-_]size\s+(\d+)", pytest_cmd, re.I)
            size = blk.group(1) or blk.group(2) if blk else "32"
            parts.append(f"page{size}")

        # instruct vs base model
        if re.search(r"instruct\s*=\s*(True|1)", pytest_cmd, re.I):
            parts.append("instr")

        # repeat_batches > 1
        r = re.search(r"--repeat[-_]batches\s+(\d+)|repeat_batches\s*\(\s*(\d+)", pytest_cmd, re.I)
        if r and (val := r.group(1) or r.group(2)) and int(val) > 1:
            parts.append(f"rep{val}")

        # greedy vs sampling
        if re.search(r"temperature\s*[:=]\s*0\.?[\s,\}]", pytest_cmd, re.I):
            parts.append("greedy")

        tag = "_".join(parts) if parts else "run"

    safe_tag = re.sub(r"[^\w\-_]", "_", tag)[:80]
    safe_tag = re.sub(r"_+", "_", safe_tag).strip("_")

    return f"{safe_tag}_{timestamp}"

def throttler_delta_header(delta_counts: dict) -> str:
    """
    Return a CSV header line that contains the 8 throttler deltas.
    """
    vals = [delta_counts.get(name, 0) for name in throttler_name]
    return "{" + ",".join(map(str, vals)) + "}"

def run_metal_test(command, timeout_min, arg):
    """Runs tt-metal workload in Docker"""
    user = os.getenv("USER")
    llama_dir = f"/home/{user}/LLAMA_31_8B_INSTRUCT_DIR/"
    if not os.path.exists(llama_dir):
        print(f"Error: {llama_dir} missing")
        sys.exit(1)

    run_dir = os.path.join(BASE_LOG_DIR, sanitize_folder_name(command))
    os.makedirs(run_dir, exist_ok=True)
    docker_log = os.path.join(run_dir, "docker_output.log")
    telem_csv = os.path.join(run_dir, "telemetry.csv")

    print(f"\nConsole log: {run_dir}")
    print(f"  Docker output: {docker_log}")
    if arg in ["all", "read_telemetry"]:
        print(f"  Telemetry CSV: {telem_csv}")

    if arg in ["all", "read_throttler_count"]:
        before_counts = read_throttler_counts() if arg in ["all", "read_throttler_count"] else None

    docker_cmd = [
        "sudo", "-E", "docker", "run", "--rm",
        "-v", "/dev/hugepages-1G:/dev/hugepages-1G",
        "--device", "/dev/tenstorrent/0",
        "-v", f"{llama_dir}:{llama_dir}",
        "-e", f"LLAMA_DIR={llama_dir}",
        "--entrypoint", "",
        "ghcr.io/tenstorrent/tt-metal/upstream-tests-bh:v0.62.0-dev20251010-12-g23761277d0",
        "/bin/bash", "-c", (
            "set -e && "
            "source /opt/venv/bin/activate && "
            "pip3 install -r models/tt_transformers/requirements.txt && "
            "export PYTHONUNBUFFERED=1 && "
            "pytest -vv --color=yes " + command
        )
    ]
    print(f"\nExecuting: {' '.join(docker_cmd)}")
    
    with open(docker_log, "w", buffering=1) as f:
        f.write(f"Command: {command}\n")
        process = subprocess.Popen(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, 
        bufsize=1, universal_newlines=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))

        time.sleep(20)

        telem_proc = None
        if arg in ["read_telemetry", "all"]:
            telem_script = os.path.expanduser("~/work/syseng/src/t6ifc/read_telem_pyluwen.py")
            if not os.path.isfile(telem_script):
                print(f"Telemetry script not found: {telem_script}")
                sys.exit(1)
        
            telem_cmd = _venv_python_cmd(telem_script,
                                 ["--delay", "0.3", "--csv", f"{telem_csv}"])
            print(f"Starting telemetry logger (venv): {' '.join(telem_cmd)}")
            telem_proc = subprocess.Popen(telem_cmd,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        
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
        after_counts = read_throttler_counts()
        delta = {name: after_counts[name] - before_counts[name] for name in throttler_name}
        header_line = throttler_delta_header(delta) + "\n"

        if os.path.exists(telem_csv) or os.path.exists(docker_log):
            orig = open(telem_csv if os.path.exists(telem_csv) else docker_log, "r").read()
            open(telem_csv if os.path.exists(telem_csv) else docker_log, "w").write(header_line + orig)
            print(f"Throttler delta header added: {header_line.strip()}")
        else:
            print("Telemetry CSV missing – delta not written")     
    
    if arg in ["all", "read_telemetry"] and telem_proc is not None:
        telem_proc.terminate()
        try:
            telem_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telem_proc.kill()
    
    print(f"\nFULL OUTPUT SAVED TO: {run_dir}")
    return run_dir

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
        b = before[name]
        a = after[name]
        delta = a - b
        table.append([name, b, a, delta])
    
    print("\nThrottler Count Delta")
    print(tabulate(table, headers, tablefmt="grid"))

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
    args = parser.parse_args()

    if args.test == "get_throttler_count":
        counts = read_throttler_counts()
        print_throttler_counts(counts)
        sys.exit(0)

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
        before_counts = read_throttler_counts()
        print_throttler_counts(before_counts)

    print("\nRunning tt-metal workload...")
    run_dir = run_metal_test(args.command, args.timeout, args.test)

    after_count = None
    if args.test in ["all", "read_throttler_count"]:
        after_counts = read_throttler_counts()
        print_throttler_delta(before_counts, after_counts)

    tt_smi_reset()
    print(f"\nAll files are in:\n  {run_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
