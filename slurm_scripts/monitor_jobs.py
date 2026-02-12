#!/usr/bin/env python3
"""
Monitor progress of SLURM jobs (ToolEmu evaluations, DPO training, and standalone eval jobs).

Usage:
    python slurm_scripts/monitor_jobs.py [--watch SECONDS] [--include-eval]
"""

import argparse
import subprocess
import os
import glob
import sys
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Add src to path for utils imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.analysis_utils import calc_col_width
from utils.model_name_utils import clean_model_name, sanitize_model_name
from utils.toolemu_utils import (
    parse_toolemu_filename,
    TOOLEMU_FULL_DATASET_SIZE,
    LOG_TAIL_BYTES,
)


# ============================================================================
# ToolEmu Job Monitoring
# ============================================================================

def parse_case_count_from_filename(filename):
    """Extract total case count from filename range (e.g., _r0-29_ -> 29 cases, _r29-58_ -> 29 cases)."""
    if not filename:
        raise ValueError("Filename cannot be empty")

    parsed = parse_toolemu_filename(filename)
    if parsed['range_start'] is None or parsed['range_end'] is None:
        raise ValueError(f"Filename missing case range: {filename}")

    return parsed['range_end'] - parsed['range_start']


def get_job_start_time(job_id):
    """Get the start time of a job (when it began running)."""
    try:
        result = subprocess.run(
            ['scontrol', 'show', 'job', job_id],
            capture_output=True, text=True, check=True
        )
        # Parse StartTime from output (format: StartTime=2025-11-04T07:33:27)
        for line in result.stdout.split('\n'):
            if 'StartTime=' in line:
                # Extract timestamp
                start_str = line.split('StartTime=')[1].split()[0]
                # Convert to Unix timestamp
                dt = datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%S')
                return dt.timestamp()
        return None
    except Exception:
        return None


def get_toolemu_jobs(username='bplaut'):
    """Get list of running/pending toolemu jobs for user."""
    try:
        result = subprocess.run(
            ['squeue', '-u', username, '--format=%i,%j,%T,%M,%R'],
            capture_output=True, text=True, check=True
        )
        jobs = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if 'toolemu' in line.lower():
                parts = line.split(',')
                if len(parts) >= 5:
                    job_id = parts[0]
                    jobs.append({
                        'job_id': job_id,
                        'name': parts[1],
                        'state': parts[2],
                        'time': parts[3],
                        'node': parts[4],
                        'start_time': get_job_start_time(job_id)
                    })
        return jobs
    except subprocess.CalledProcessError:
        return []


def get_trajectory_progress(job_id, job_start_time=None):
    """Find trajectory file for job and count completed cases."""
    log_file = f"logs/{job_id}.out"
    if not os.path.exists(log_file):
        return None, None, None, None, None, None, None

    try:
        with open(log_file, 'r') as f:
            content = f.read()

            # Extract config from log output
            agent_model = None
            emu_model = None
            eval_model = None
            quantization = None
            case_range = None
            actual_case_count = None  # Actual count from "Total N [case]s after filtering"

            for line in content.split('\n'):
                if line.strip().startswith('Agent:'):
                    agent_model = line.split('Agent:')[-1].strip()
                elif line.strip().startswith('Emulator:'):
                    emu_model = line.split('Emulator:')[-1].strip()
                elif line.strip().startswith('Evaluator:'):
                    eval_model = line.split('Evaluator:')[-1].strip()
                elif line.strip().startswith('Quantization:'):
                    quantization = line.split('Quantization:')[-1].strip()
                elif line.strip().startswith('Case index range:'):
                    case_range = line.split('Case index range:')[-1].strip()
                elif "[case]s after filtering" in line:
                    # Parse: "Total N [case]s after filtering."
                    match = re.search(r'Total\s+(\d+)\s+\[case\]s after filtering', line)
                    if match:
                        actual_case_count = int(match.group(1))

            if not (agent_model and emu_model):
                return None, None, None, None, None, None, None

            # Sanitize model names for filesystem (use same function as run.py)
            agent_safe = sanitize_model_name(agent_model)
            emu_safe = sanitize_model_name(emu_model)

            # Build file pattern
            patterns = [
                f"output/trajectories/{agent_safe}/{agent_safe}_emu-{emu_safe}_*.jsonl",
            ]

            matches = []
            for pattern in patterns:
                matches.extend(glob.glob(pattern))

            # Filter out eval result files
            traj_files = [f for f in matches
                         if '_eval_agent_' not in f and '_unified_report' not in f]

            # Find the most recent one
            best_match = None
            best_mtime = 0

            for traj_file in traj_files:
                # Parse filename to get structured info (handles int4 default)
                try:
                    parsed = parse_toolemu_filename(os.path.basename(traj_file))
                    file_quant = parsed['quantization']
                    file_range = f"{parsed['range_start']}-{parsed['range_end']}" if parsed['range_start'] is not None else None
                except ValueError:
                    continue  # Skip unparseable files

                if quantization and file_quant != quantization:
                    continue
                if case_range and file_range != case_range:
                    continue

                mtime = os.path.getmtime(traj_file)
                if job_start_time and mtime < (job_start_time - 60):
                    continue

                if mtime > best_mtime:
                    best_mtime = mtime
                    best_match = traj_file

            if best_match and os.path.exists(best_match):
                # Count trajectory lines
                with open(best_match, 'r') as tf:
                    traj_lines = sum(1 for _ in tf)

                # Count eval file lines
                base_path = best_match.replace('.jsonl', '')
                safe_eval_file = f"{base_path}_eval_agent_safe.jsonl"
                help_eval_file = f"{base_path}_eval_agent_help.jsonl"
                help_ignore_safety_eval_file = f"{base_path}_eval_agent_help_ignore_safety.jsonl"

                safe_lines = 0
                help_lines = 0
                help_ignore_safety_lines = 0

                if os.path.exists(safe_eval_file):
                    with open(safe_eval_file, 'r') as f:
                        safe_lines = sum(1 for _ in f)

                if os.path.exists(help_eval_file):
                    with open(help_eval_file, 'r') as f:
                        help_lines = sum(1 for _ in f)

                if os.path.exists(help_ignore_safety_eval_file):
                    with open(help_ignore_safety_eval_file, 'r') as f:
                        help_ignore_safety_lines = sum(1 for _ in f)

                # Count how many help eval files exist (default is 1: ignore_safety only)
                help_eval_files = [
                    help_eval_file,
                    help_ignore_safety_eval_file,
                ]
                help_eval_count = sum(1 for f in help_eval_files if os.path.exists(f))
                # Default to 1 (ignore_safety only) unless more files exist
                help_eval_count = max(1, help_eval_count)

                total_help_lines = help_lines + help_ignore_safety_lines

                basename = os.path.basename(best_match)

                # Use actual case count from log if available, otherwise parse from filename
                if actual_case_count is not None:
                    total_cases = actual_case_count
                else:
                    total_cases = parse_case_count_from_filename(basename)

                return traj_lines, safe_lines, total_help_lines, basename, quantization, total_cases, help_eval_count

    except Exception:
        pass

    return None, None, None, None, None, None, None


def parse_time(time_str):
    """Parse SLURM time format to seconds."""
    if '-' in time_str:
        day_part, time_part = time_str.split('-')
        days = int(day_part)
        parts = time_part.split(':')
    else:
        days = 0
        parts = time_str.split(':')

    if len(parts) == 3:
        return days * 86400 + int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return days * 86400 + int(parts[0]) * 60 + int(parts[1])
    return 0


def format_time(seconds):
    """Format seconds to human readable time."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds//60}m {seconds%60}s"
    elif seconds < 86400:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        mins = (seconds % 3600) // 60
        return f"{days}d {hours}h {mins}m"


def get_job_config(filename):
    """Extract configuration from filename."""
    if not filename:
        raise ValueError("Filename cannot be empty")

    parsed = parse_toolemu_filename(filename)
    agent = clean_model_name(parsed['agent_model'], format="long")
    emulator = clean_model_name(parsed['emu_model'], format="long")
    evaluator = clean_model_name(parsed['eval_model'], format="long")
    return agent, emulator, evaluator


def print_table(headers, rows, separator_char="-", use_separators=True):
    """Print a formatted table with dynamic column widths. Rows are dicts with keys matching headers."""
    if not rows:
        return

    # Normalize keys to lowercase for matching
    header_keys = [h.lower().replace(' ', '_') for h in headers]

    # Calculate column widths
    col_widths = {}
    for i, header in enumerate(headers):
        key = header_keys[i]
        col_widths[key] = calc_col_width(header, [str(row.get(key, '')) for row in rows])

    # Print header
    sep = " | " if use_separators else " "
    header_line = sep.join(f"{h:<{col_widths[header_keys[i]]}}" for i, h in enumerate(headers))
    print(header_line)
    print(separator_char * len(header_line))

    # Print rows
    for row in rows:
        line = sep.join(f"{str(row.get(key, '')):<{col_widths[key]}}" for key in header_keys)
        print(line)


def print_toolemu_jobs(jobs, show_pending=True):
    """Print formatted table of ToolEmu job progress."""
    if not jobs:
        return

    running = [j for j in jobs if j['state'] == 'RUNNING']
    pending = [j for j in jobs if j['state'] in ['PENDING', 'PD']]

    print("\n" + "="*120)
    print("TOOLEMU EVALUATION JOBS")
    print("="*120)

    if running:
        print(f"\n🏃 RUNNING ({len(running)}):")

        # Prepare data rows
        running_sorted = sorted(running, key=lambda x: x['job_id'])
        gpu_info_map = get_gpu_info_batch([j['job_id'] for j in running_sorted])
        rows = []
        for job in running_sorted:
            traj_progress, safe_progress, help_progress, filename, quant, total_cases, help_eval_count = get_trajectory_progress(job['job_id'], job.get('start_time'))
            elapsed_sec = parse_time(job['time'])
            gpu_util_str, mem_str = format_gpu_summary(gpu_info_map.get(job['job_id'], []))

            if traj_progress is not None:
                agent, emulator, evaluator = get_job_config(filename)
                rows.append({
                    'job_id': job['job_id'],
                    'traj': f"{traj_progress}/{total_cases}",
                    'safe': f"{safe_progress}/{total_cases}",
                    'help': f"{help_progress}/{total_cases * help_eval_count}",
                    'elapsed': format_time(elapsed_sec),
                    'agent': agent,
                    'emu': emulator,
                    'eval': evaluator,
                    'gpu_util': gpu_util_str,
                    'mem_usage': mem_str,
                })
            else:
                rows.append({
                    'job_id': job['job_id'],
                    'traj': "?",
                    'safe': "?",
                    'help': "?",
                    'elapsed': format_time(elapsed_sec),
                    'agent': "?",
                    'emu': "?",
                    'eval': "?",
                    'gpu_util': gpu_util_str,
                    'mem_usage': mem_str,
                })

        print_table(['Job ID', 'Traj', 'Safe', 'Help', 'Elapsed', 'Agent', 'Emu', 'Eval', 'GPU Util', 'Mem Usage'], rows)

    if show_pending and pending:
        print(f"\n⏳ PENDING ({len(pending)}):")
        print("-"*120)
        for job in sorted(pending, key=lambda x: x['job_id']):
            print(f"  Job {job['job_id']} - Waiting for resources ({job.get('node', 'N/A')})")

    if running or pending:
        print(f"\nTotal: {len(running)} running, {len(pending)} pending")


# ============================================================================
# Training Job Monitoring (DPO)
# ============================================================================

def run_command(cmd, timeout=10):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", -1


def extract_config_from_path(stdout_path):
    """Extract config name from stdout path."""
    if stdout_path:
        match = re.search(r'dpo_output/([^/]+)', stdout_path)
        if match:
            return match.group(1)
    return "unknown"


def extract_progress(stdout_path):
    """Extract progress from stdout file"""
    if not stdout_path or not os.path.exists(stdout_path):
        return ""

    try:
        with open(stdout_path, 'r') as f:
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(max(0, file_size - LOG_TAIL_BYTES), 0)
            content = f.read()

        matches = re.findall(r'\s*(\d+)%\|[^|]*\|\s*(\d+/\d+)', content)
        if matches:
            percent, fraction = matches[-1]
            return f"{fraction} ({percent}%)"
    except Exception:
        pass

    return ""


def get_gpu_info(jobid):
    """Get GPU info for a job"""
    cmd = f"timeout 10s srun --jobid={jobid} nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null"
    output, exit_code = run_command(cmd)

    if exit_code != 0 or not output:
        return []

    gpus = []
    for line in output.strip().split('\n'):
        parts = line.split(',')
        if len(parts) == 3:
            util = parts[0].strip()
            mem_used = float(parts[1].strip()) / 1024
            mem_total = float(parts[2].strip()) / 1024
            gpus.append({
                'util': util,
                'mem_used': mem_used,
                'mem_total': mem_total
            })

    return gpus


def format_gpu_summary(gpus):
    """Format GPU info list into summary strings (gpu_util, mem_usage) for table display."""
    if not gpus:
        return "N/A", "N/A"
    gpu_util_str = ','.join(f"{g['util']}%" for g in gpus)
    mem_str = ','.join(f"{g['mem_used']:.1f}/{g['mem_total']:.1f}GB" for g in gpus)
    return gpu_util_str, mem_str


def get_gpu_info_batch(job_ids):
    """Fetch GPU info for multiple jobs in parallel. Returns dict of job_id -> gpu_info list."""
    if not job_ids:
        return {}
    with ThreadPoolExecutor(max_workers=len(job_ids)) as executor:
        futures = {executor.submit(get_gpu_info, jid): jid for jid in job_ids}
        results = {}
        for future in as_completed(futures):
            jid = futures[future]
            try:
                results[jid] = future.result()
            except Exception:
                results[jid] = []
        return results


def get_training_jobs(username='bplaut'):
    """Get list of running and pending DPO training jobs.

    Returns: (running_jobs, pending_count)
    """
    # Get all jobs (running and pending)
    cmd = f"squeue -u {username} -h -o '%A %j %R %M %T'"
    output, _ = run_command(cmd)

    if not output:
        return [], 0

    running_jobs = []
    pending_count = 0

    for line in output.strip().split('\n'):
        parts = line.split()
        if len(parts) < 5:
            continue

        jobid = parts[0]
        state = parts[4]

        # Filter for DPO jobs
        scontrol_output, _ = run_command(f"scontrol show job {jobid} 2>/dev/null")
        stdout_match = re.search(r'StdOut=(\S+)', scontrol_output)
        stdout_path = stdout_match.group(1) if stdout_match else ""

        if not stdout_path or 'dpo_output' not in stdout_path:
            continue

        # Count pending jobs
        if state in ['PENDING', 'PD']:
            pending_count += 1
            continue

        # Only add running jobs to the list
        if state == 'RUNNING':
            runtime = parts[3]
            config = extract_config_from_path(stdout_path)
            progress = extract_progress(stdout_path)

            running_jobs.append({
                'jobid': jobid,
                'config': config,
                'runtime': runtime,
                'progress': progress,
                'gpus': []
            })

    # Batch-fetch GPU info in parallel
    gpu_info_map = get_gpu_info_batch([job['jobid'] for job in running_jobs])
    for job in running_jobs:
        job['gpus'] = gpu_info_map.get(job['jobid'], [])

    return sorted(running_jobs, key=lambda x: int(x['jobid'])), pending_count


def print_training_jobs(jobs_data, pending_count=0):
    """Print formatted table of DPO training job progress."""
    if not jobs_data and pending_count == 0:
        return

    print("\n" + "="*120)
    print("DPO TRAINING JOBS")
    print("="*120)

    if jobs_data:
        # Prepare flat list of all GPU entries for width calculation
        all_gpu_util_strs = []
        all_mem_strs = []
        for job in jobs_data:
            for gpu in (job['gpus'] or []):
                all_gpu_util_strs.append(f"{gpu['util']}%")
                all_mem_strs.append(f"{gpu['mem_used']:.1f}/{gpu['mem_total']:.1f}GB")

        # Calculate column widths
        col_widths = {
            'jobid': calc_col_width('Job ID', [job['jobid'] for job in jobs_data]),
            'config': calc_col_width('Config', [job['config'] for job in jobs_data]),
            'runtime': calc_col_width('Runtime', [job['runtime'] for job in jobs_data]),
            'progress': calc_col_width('Progress', [job['progress'] for job in jobs_data]),
            'gpu_util': calc_col_width('GPU Util', all_gpu_util_strs),
            'mem_usage': calc_col_width('Mem Usage', all_mem_strs),
        }

        # Print header
        sep = " | "
        header = sep.join([
            f"{'Job ID':<{col_widths['jobid']}}",
            f"{'Config':<{col_widths['config']}}",
            f"{'Runtime':<{col_widths['runtime']}}",
            f"{'Progress':<{col_widths['progress']}}",
            f"{'GPU Util':>{col_widths['gpu_util']}}",
            f"{'Mem Usage':>{col_widths['mem_usage']}}"
        ])
        print("\n" + header)
        print("-" * len(header))

        # Print jobs
        for job in jobs_data:
            gpus = job['gpus'] or []

            # First line (or only line if no GPUs)
            first_gpu = gpus[0] if gpus else None
            gpu_util_str = f"{first_gpu['util']}%" if first_gpu else "N/A"
            mem_str = f"{first_gpu['mem_used']:.1f}/{first_gpu['mem_total']:.1f}GB" if first_gpu else "N/A"

            line = sep.join([
                f"{job['jobid']:<{col_widths['jobid']}}",
                f"{job['config']:<{col_widths['config']}}",
                f"{job['runtime']:<{col_widths['runtime']}}",
                f"{job['progress']:<{col_widths['progress']}}",
                f"{gpu_util_str:>{col_widths['gpu_util']}}",
                f"{mem_str:>{col_widths['mem_usage']}}"
            ])
            print(line)

            # Additional GPU lines (for multi-GPU jobs)
            for gpu in gpus[1:]:
                gpu_util_str = f"{gpu['util']}%"
                mem_str = f"{gpu['mem_used']:.1f}/{gpu['mem_total']:.1f}GB"
                gpu_line = sep.join([
                    f"{'':<{col_widths['jobid']}}",
                    f"{'':<{col_widths['config']}}",
                    f"{'':<{col_widths['runtime']}}",
                    f"{'':<{col_widths['progress']}}",
                    f"{gpu_util_str:>{col_widths['gpu_util']}}",
                    f"{mem_str:>{col_widths['mem_usage']}}"
                ])
                print(gpu_line)

        print(f"\nTotal: {len(jobs_data)} running, {pending_count} pending")
    else:
        print(f"\nTotal: 0 running, {pending_count} pending")


# ============================================================================
# Standalone Eval Job Monitoring
# ============================================================================

def get_eval_jobs(username='bplaut'):
    """Get list of running and pending standalone eval jobs (from run_eval.sh).

    Returns: (running_jobs, pending_count)
    """
    cmd = f"squeue -u {username} -h -o '%A %j %M %T'"
    output, _ = run_command(cmd)

    if not output:
        return [], 0

    running_jobs = []
    pending_count = 0

    for line in output.strip().split('\n'):
        parts = line.split()
        if len(parts) < 4:
            continue

        jobid = parts[0]
        job_name = parts[1]
        runtime = parts[2]
        state = parts[3]

        # Filter for eval jobs (job name is 'eval' from run_eval.sh)
        if job_name != 'eval':
            continue

        # Count pending jobs
        if state in ['PENDING', 'PD']:
            pending_count += 1
            continue

        # Only add running jobs to the list
        if state == 'RUNNING':
            eval_info = parse_eval_job_log(jobid)
            running_jobs.append({
                'jobid': jobid,
                'runtime': runtime,
                **eval_info
            })

    return sorted(running_jobs, key=lambda x: int(x['jobid'])), pending_count


def parse_eval_job_log(job_id):
    """Parse eval job log to extract configuration and progress."""
    log_file = f"logs/{job_id}.out"
    info = {
        'agent_model': '?',
        'emu_model': '?',
        'evaluator_model': '?',
        'quantization': '?',
        'eval_type': '?',
        'num_replicates': '?',
        'total_trajectories': '?',
        'progress': '?',
    }

    if not os.path.exists(log_file):
        return info

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Parse configuration from header
        for line in content.split('\n'):
            if line.startswith('Trajectory file:'):
                traj_path = line.split(':', 1)[1].strip()
                # Extract just the filename for display
                traj_filename = os.path.basename(traj_path)
                # Extract agent and emu model from trajectory filename
                try:
                    parsed = parse_toolemu_filename(traj_filename)
                    info['agent_model'] = parsed['agent_model']
                    info['emu_model'] = parsed['emu_model']
                except ValueError:
                    pass  # Keep default '?'
            elif line.startswith('Evaluator model:'):
                info['evaluator_model'] = line.split(':', 1)[1].strip()
            elif line.startswith('Quantization:'):
                info['quantization'] = line.split(':', 1)[1].strip()
            elif line.startswith('Eval type:'):
                info['eval_type'] = line.split(':', 1)[1].strip()
            elif line.startswith('Num replicates:'):
                info['num_replicates'] = line.split(':', 1)[1].strip()

        # Parse total trajectories from "Loaded X [trajectory]s" or "Totally X [trajectory]s"
        total_match = re.search(r'Totally (\d+) \[trajectory\]s', content)
        if total_match:
            info['total_trajectories'] = int(total_match.group(1))
        else:
            loaded_match = re.search(r'Loaded (\d+) \[trajectory\]s', content)
            if loaded_match:
                info['total_trajectories'] = int(loaded_match.group(1))

        # Parse progress from tqdm output (e.g., "50%|█████     | 7/14")
        # Look for the last progress bar line
        progress_matches = re.findall(r'(\d+)%\|[^|]*\|\s*(\d+)/(\d+)', content)
        if progress_matches:
            percent, current, total = progress_matches[-1]
            info['progress'] = f"{current}/{total}"
        elif info['total_trajectories'] != '?':
            # If no progress bar yet but we know total, show 0/total
            info['progress'] = f"0/{info['total_trajectories']}"

    except Exception:
        pass

    return info


def print_eval_jobs(jobs_data, pending_count=0):
    """Print formatted table of standalone eval job progress."""
    if not jobs_data and pending_count == 0:
        return

    print("\n" + "="*120)
    print("STANDALONE EVAL JOBS (run_eval.sh)")
    print("="*120)

    if jobs_data:
        # Batch-fetch GPU info in parallel
        gpu_info_map = get_gpu_info_batch([job['jobid'] for job in jobs_data])

        # Prepare rows for table
        rows = []
        for job in jobs_data:
            # Clean up model names for display
            agent_model = clean_model_name(job['agent_model'], format="short")
            emu_model = clean_model_name(job['emu_model'], format="short")
            eval_model = clean_model_name(job['evaluator_model'], format="short")
            gpu_util_str, mem_str = format_gpu_summary(gpu_info_map.get(job['jobid'], []))

            rows.append({
                'job_id': job['jobid'],
                'progress': job['progress'],
                'runtime': job['runtime'],
                'agent': agent_model,
                'emu': emu_model,
                'eval': eval_model,
                'eval_type': job['eval_type'],
                'gpu_util': gpu_util_str,
                'mem_usage': mem_str,
            })

        print_table(
            ['Job ID', 'Progress', 'Runtime', 'Agent', 'Emu', 'Eval', 'Eval Type', 'GPU Util', 'Mem Usage'],
            rows
        )

        print(f"\nTotal: {len(jobs_data)} running, {pending_count} pending")
    else:
        print(f"\nTotal: 0 running, {pending_count} pending")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Monitor ToolEmu, DPO, and eval job progress')
    parser.add_argument('--watch', '-w', type=int, metavar='SECONDS',
                       help='Refresh every N seconds (watch mode)')
    parser.add_argument('--no-pending', action='store_true',
                       help='Hide pending ToolEmu jobs')
    parser.add_argument('--include-eval', '-e', action='store_true',
                       help='Include standalone eval jobs (from run_eval.sh)')

    args = parser.parse_args()

    if args.watch:
        print(f"Watching jobs for bplaut, refreshing every {args.watch} seconds...")
        print("Press Ctrl+C to stop.\n")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')

                print(f"JOB MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Get and display ToolEmu jobs
                toolemu_jobs = get_toolemu_jobs('bplaut')
                print_toolemu_jobs(toolemu_jobs, show_pending=not args.no_pending)

                # Get and display DPO training jobs
                training_jobs, pending_count = get_training_jobs('bplaut')
                print_training_jobs(training_jobs, pending_count)

                # Get and display standalone eval jobs (if requested)
                if args.include_eval:
                    eval_jobs, eval_pending = get_eval_jobs('bplaut')
                    print_eval_jobs(eval_jobs, eval_pending)

                print("\n" + "="*120 + "\n")

                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        print(f"JOB MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Get and display ToolEmu jobs
        toolemu_jobs = get_toolemu_jobs('bplaut')
        print_toolemu_jobs(toolemu_jobs, show_pending=not args.no_pending)

        # Get and display DPO training jobs
        training_jobs, pending_count = get_training_jobs('bplaut')
        print_training_jobs(training_jobs, pending_count)

        # Get and display standalone eval jobs (if requested)
        if args.include_eval:
            eval_jobs, eval_pending = get_eval_jobs('bplaut')
            print_eval_jobs(eval_jobs, eval_pending)

        print("\n" + "="*120 + "\n")


if __name__ == '__main__':
    main()
