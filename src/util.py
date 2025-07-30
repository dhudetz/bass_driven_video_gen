# util.py

import subprocess
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
