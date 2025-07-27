#!/usr/bin/env python3
import os
import sys
import random
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# ========= CONFIG =========
video_folder    = "/Users/dhudetz/scripts/video_gen/footage_outside"
mp3_filename    = "audio.mp3"
output_filename = "compiled.mp4"
plot_filename   = "bass_hits_plot.png"

# Bass detection params
LF_MIN_HZ       = 10     # Lower bound for 808/sub
LF_MAX_HZ       = 50    # Upper bound for 808/sub
N_FFT           = 4096
HOP_LENGTH      = 256
ONSET_DELTA     = 0.1   # Lower = more sensitive
RANDOM_CLIP_MIN = 0.5    # Min clip length
RANDOM_CLIP_MAX = 5.0    # Max clip length
COOLDOWN        = 0.1   # Minimum time (seconds) between bass hits

# ========= UTILS =========
def run(cmd):
    res = subprocess.run(cmd, check=True, capture_output=True)
    return res.stdout.decode("utf-8", errors="ignore")

def ffprobe_duration(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    try:
        return float(run(cmd).strip())
    except Exception:
        return None

def safe_tmp(suffix):
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=tempfile.gettempdir()).name

import scipy.ndimage

def detect_bass_hits(mp3_path, plot_path):
    import librosa
    from scipy.signal import butter, sosfiltfilt

    y, sr = librosa.load(mp3_path, mono=True)
    nyquist = 0.5 * sr
    sos = butter(4, [LF_MIN_HZ / nyquist, LF_MAX_HZ / nyquist], btype='band', output='sos')
    y_filtered = sosfiltfilt(sos, y)
    y_filtered = np.nan_to_num(y_filtered)

    onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, hop_length=HOP_LENGTH)
    # Smooth onset envelope
    onset_env = scipy.ndimage.median_filter(onset_env, size=5)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP_LENGTH)

    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH,
        units='time',
        backtrack=True,
        pre_max=10, post_max=10,
        pre_avg=20, post_avg=20,
        delta=0.12  # try 0.1–0.15 for fewer hits
    )

    # Apply cooldown filter
    filtered_onsets = []
    last_onset = -COOLDOWN
    for onset in sorted(onsets):
        if onset - last_onset >= COOLDOWN:
            filtered_onsets.append(onset)
            last_onset = onset

    plt.figure(figsize=(12, 4))
    plt.plot(times, onset_env, label="Bass Onset Envelope", color='blue')
    plt.vlines(filtered_onsets, 0, onset_env.max(), color='red', alpha=0.8, linestyle='--', label="Bass Hits")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.title("Detected Bass Hits (Filtered)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return sorted(filtered_onsets)

def extract_random_segment(video_path, duration):
    vid_dur = ffprobe_duration(video_path)
    if vid_dur is None or vid_dur <= 0:
        return None
    start_time = random.uniform(0, max(0, vid_dur - duration))
    seg_path = safe_tmp(".mov")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path,
        "-c", "copy",
        seg_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return seg_path
    except subprocess.CalledProcessError:
        return None

# ========= MAIN =========
def main():
    print(f"[INFO] Scanning: {video_folder}")
    if not os.path.isdir(video_folder):
        print(f"[ERROR] Folder does not exist: {video_folder}")
        sys.exit(1)

    mp3_path = os.path.join(video_folder, mp3_filename)
    if not os.path.isfile(mp3_path):
        print(f"[ERROR] MP3 file not found: {mp3_path}")
        sys.exit(1)

    video_files = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if f.lower().endswith(".mov") and not f.startswith("._")
    ]
    if not video_files:
        print("[ERROR] No valid .mov files found.")
        sys.exit(1)

    # Filter out <5s files
    valid_files = [vf for vf in video_files if ffprobe_duration(vf) >= 5]
    if not valid_files:
        print("[ERROR] No video files >= 5s found.")
        sys.exit(1)

    print(f"[INFO] {len(valid_files)} valid video files found.")

    # Bass detection
    plot_path = os.path.join(video_folder, plot_filename)
    print("[INFO] Detecting bass hits and saving plot...")
    bass_hits = detect_bass_hits(mp3_path, plot_path)
    print(f"[INFO] Detected {len(bass_hits)} bass hits. Plot saved to {plot_path}")

    mp3_duration = ffprobe_duration(mp3_path)
    if mp3_duration <= 0:
        print("[ERROR] Invalid MP3 duration.")
        sys.exit(1)

    # Create segments per bass hit
    segments = []
    random.shuffle(valid_files)
    clip_idx = 0

    for i in range(len(bass_hits) - 1):
        seg_dur = max(RANDOM_CLIP_MIN, min(bass_hits[i+1] - bass_hits[i], RANDOM_CLIP_MAX))
        seg = extract_random_segment(valid_files[clip_idx], seg_dur)
        if seg:
            segments.append(seg)

        clip_idx = (clip_idx + 1) % len(valid_files)
        if clip_idx == 0:
            random.shuffle(valid_files)  # reshuffle after cycling all clips

    # Concatenate segments
    file_list_txt = safe_tmp(".txt")
    with open(file_list_txt, "w") as f:
        for seg in segments:
            f.write(f"file '{seg}'\n")

    temp_concat = safe_tmp(".mov")
    concat_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", file_list_txt, "-c", "copy", temp_concat]
    subprocess.run(concat_cmd, check=True)

    output_path = os.path.join(video_folder, output_filename)
    final_cmd = [
        "ffmpeg", "-y",
        "-i", temp_concat,
        "-i", mp3_path,
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(final_cmd, check=True)

    # Cleanup
    os.remove(temp_concat)
    os.remove(file_list_txt)
    for seg in segments:
        os.remove(seg)

    print(f"[✅ DONE] Compiled video saved to:\n{output_path}")
    print(f"[INFO] Bass plot saved to:\n{plot_path}")

if __name__ == "__main__":
    main()
