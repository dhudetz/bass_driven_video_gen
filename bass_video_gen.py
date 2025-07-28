#!/usr/bin/env python3
import os
import sys
import random
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import scipy.ndimage
from datetime import datetime

# ========= CONFIG =========
LF_MIN_HZ       = 1000     # Lower bound for energy capture
LF_MAX_HZ       = 1300     # Upper bound for energy capture
N_FFT           = 1024
HOP_LENGTH      = 256
ONSET_DELTA     = 0.10   # Onset detection threshold
RANDOM_CLIP_MIN = 0.1   # Min clip length
RANDOM_CLIP_MAX = 15     # Max clip length
COOLDOWN        = RANDOM_CLIP_MIN   # Minimum time between bass hits (seconds)
MIN_INPUT_VIDEO_LEN = 5

date_str = datetime.now().strftime("%Y%m%d_%H%M")
mp3_filename = "audio.mp3"
output_filename = f"compiled_{date_str}.mp4"

# ========= PIPELINE TOGGLES =========
ENABLE_BASS_DETECTION     = True
ENABLE_PLOTTING           = True
ENABLE_EXTRACT_SEGMENTS   = True
ENABLE_CONCATENATION      = True
ENABLE_ADD_AUDIO          = True
ENABLE_CLEANUP            = True

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

def ffprobe_stream_info(path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    try:
        lines = run(cmd).splitlines()
        fps = None
        for line in lines:
            if '/' in line:
                try:
                    num, den = line.split('/')
                    fps = float(num) / float(den)
                    break
                except:
                    continue
        return {"r_fps": fps}
    except Exception:
        return {"r_fps": 30.0}

def safe_tmp(suffix):
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=tempfile.gettempdir()).name

# ========= DETECTION =========
def detect_bass_hits(mp3_path, plot_path=None):
    import librosa
    from scipy.signal import butter, sosfiltfilt

    # Load audio
    y, sr = librosa.load(mp3_path, mono=True)

    # Bandpass for sub/bass drums
    nyquist = 0.5 * sr
    sos = butter(4, [LF_MIN_HZ / nyquist, LF_MAX_HZ / nyquist], btype='band', output='sos')
    y_filtered = sosfiltfilt(sos, y)
    y_filtered = np.nan_to_num(y_filtered)

    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, hop_length=HOP_LENGTH)
    onset_env = scipy.ndimage.median_filter(onset_env, size=5)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP_LENGTH)

    # Detect onsets
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH,
        units='time',
        backtrack=True,
        pre_max=10, post_max=10,
        pre_avg=20, post_avg=20,
        delta=ONSET_DELTA
    )

    # Apply cooldown
    all_onsets = sorted(onsets)
    kept_onsets = []
    dropped_onsets = []
    if len(all_onsets) > 0:
        kept_onsets.append(all_onsets[0])
        last_time = all_onsets[0]
        for onset in all_onsets[1:]:
            if onset - last_time >= COOLDOWN:
                kept_onsets.append(onset)
                last_time = onset
            else:
                dropped_onsets.append(onset)

    if plot_path:
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

    return kept_onsets

# ========= VIDEO SEGMENTS =========
def extract_random_segment(video_path, duration):
    vid_dur = ffprobe_duration(video_path)
    if vid_dur is None or vid_dur <= 0:
        return None
    start_time = random.uniform(0, max(0, vid_dur - duration))
    seg_path = safe_tmp(".mp4")

    # Keep source FPS
    info = ffprobe_stream_info(video_path)
    fps = info["r_fps"] or 30.0
    gop = int(round(fps * 2))

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
        "-pix_fmt", "yuv420p",
        "-g", str(gop), "-keyint_min", str(gop),
        "-sc_threshold", "0",
        "-movflags", "+faststart",
        seg_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return seg_path
    except subprocess.CalledProcessError:
        return None

# ========= MAIN =========
def main():
    if len(sys.argv) < 2:
        print("Usage: python chaos_video_gen.py <video_directory>")
        sys.exit(1)

    video_folder = os.path.abspath(sys.argv[1])
    plot_filename = f"bass_plot_{LF_MIN_HZ}_{LF_MAX_HZ}.png"

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
    valid_files = []
    for vf in tqdm(video_files, desc="Validating video files"):
        dur = ffprobe_duration(vf)
        if dur is not None and dur >= MIN_INPUT_VIDEO_LEN:
            valid_files.append(vf)

    if not valid_files:
        print("[ERROR] No video files >= 5s found.")
        sys.exit(1)

    print(f"[INFO] {len(valid_files)} valid video files found.")

    # Bass detection
    bass_hits_path = os.path.join(video_folder, "bass_hits.txt")
    if ENABLE_BASS_DETECTION:
        print("[INFO] Detecting bass hits...")
        plot_path = os.path.join(video_folder, plot_filename) if ENABLE_PLOTTING else None
        bass_hits = detect_bass_hits(mp3_path, plot_path)
        with open(bass_hits_path, "w") as f:
            for t in bass_hits:
                f.write(f"{t}\n")
        print(f"[INFO] Detected {len(bass_hits)} bass hits.")
        if ENABLE_PLOTTING:
            print(f"[INFO] Plot saved to {plot_path}")
    else:
        if not os.path.isfile(bass_hits_path):
            print(f"[ERROR] Bass hits file not found: {bass_hits_path}")
            sys.exit(1)
        with open(bass_hits_path, "r") as f:
            bass_hits = [float(line.strip()) for line in f if line.strip()]
        print(f"[INFO] Loaded {len(bass_hits)} bass hits from file.")

    mp3_duration = ffprobe_duration(mp3_path)
    if mp3_duration <= 0:
        print("[ERROR] Invalid MP3 duration.")
        sys.exit(1)

    # Create segments per bass hit
    segments_dir = os.path.join(video_folder, "segments")
    if ENABLE_EXTRACT_SEGMENTS:
        os.makedirs(segments_dir, exist_ok=True)
        random.shuffle(valid_files)
        segments = []
        clip_idx = 0
        for i in tqdm(range(len(bass_hits) - 1), desc="Extracting video segments"):
            seg_dur = max(RANDOM_CLIP_MIN, min(bass_hits[i+1] - bass_hits[i], RANDOM_CLIP_MAX))
            seg_path = os.path.join(segments_dir, f"seg_{i:04d}.mp4")
            seg_file = extract_random_segment(valid_files[clip_idx], seg_dur)
            if seg_file:
                shutil.move(seg_file, seg_path)
                segments.append(seg_path)

            clip_idx = (clip_idx + 1) % len(valid_files)
            if clip_idx == 0:
                random.shuffle(valid_files)
    else:
        if not os.path.isdir(segments_dir):
            print(f"[ERROR] Segments directory not found: {segments_dir}")
            sys.exit(1)
        segments = [os.path.join(segments_dir, f) for f in sorted(os.listdir(segments_dir)) if f.lower().endswith(".mp4")]
        print(f"[INFO] Loaded {len(segments)} segments from directory.")

    # Concatenate segments
    file_list_txt = os.path.join(video_folder, "file_list.txt")
    with open(file_list_txt, "w") as f:
        for seg in segments:
            f.write(f"file '{seg}'\n")

    temp_concat = os.path.join(video_folder, "concat.mp4")
    if ENABLE_CONCATENATION:
        print("[INFO] Concatenating segments...")
        concat_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", file_list_txt, "-c", "copy", temp_concat]
        try:
            subprocess.run(concat_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("[WARN] Concat copy failed, re-encoding...")
            fallback_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", file_list_txt,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                temp_concat
            ]
            subprocess.run(fallback_cmd, check=True, capture_output=True)
    else:
        if not os.path.exists(temp_concat):
            print(f"[ERROR] Concat file not found: {temp_concat}")
            sys.exit(1)
        print("[INFO] Using existing concat file.")

    output_path = os.path.join(video_folder, output_filename)
    if ENABLE_ADD_AUDIO:
        print("[INFO] Adding audio to video...")
        final_cmd = [
            "ffmpeg", "-y",
            "-i", temp_concat,
            "-i", mp3_path,
            "-shortest",
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ]
        subprocess.run(final_cmd, check=True, capture_output=True)

    # Cleanup
    if ENABLE_CLEANUP:
        print("[INFO] Cleaning up intermediates...")
        os.remove(file_list_txt)
        os.remove(temp_concat)
        shutil.rmtree(segments_dir)

    print(f"[✅ DONE] Compiled video saved to:\n{output_path}")
    if ENABLE_PLOTTING and ENABLE_BASS_DETECTION:
        print(f"[INFO] Bass plot saved to:\n{os.path.join(video_folder, plot_filename)}")

if __name__ == "__main__":
    main()

