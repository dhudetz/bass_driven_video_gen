# util.py

import subprocess
import tempfile
import matplotlib.pyplot as plt

def run_command(cmd):
    """Run a shell command and return stdout as a decoded string."""
    res = subprocess.run(cmd, check=True, capture_output=True)
    return res.stdout.decode("utf-8", errors="ignore")

def plot_onsets(times, onset_env, kept_onsets, dropped_onsets, plot_path):
    """Save a plot visualizing detected bass hits."""
    plt.figure(figsize=(12, 4))
    plt.plot(times, onset_env, label="Bass Onset Envelope", color='blue')
    plt.vlines(kept_onsets, 0, onset_env.max(), color='red', alpha=0.8, linestyle='-', label="Kept Bass Hits")
    if dropped_onsets:
        plt.vlines(dropped_onsets, 0, onset_env.max(), color='orange', alpha=0.8, linestyle='--', label="Dropped Bass Hits")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Detected Bass Hits (Filtered)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def safe_tmp(suffix):
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=tempfile.gettempdir()).name

def ffprobe_duration(path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    try:
        return float(run_command(cmd).strip())
    except Exception:
        return None

def ffprobe_stream_info(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    try:
        lines = run_command(cmd).splitlines()
        for line in lines:
            if '/' in line:
                num, den = line.split('/')
                return {"r_fps": float(num) / float(den)}
    except Exception:
        pass
    return {"r_fps": 30.0}
