#!/usr/bin/env python3
"""
chaos_video_gen.py

Creates a chaotic video compilation synced to detected bass hits in an audio file.
Uses random video segments from input videos and aligns cuts to audio onsets.

Modules:
    - BassDetector: Detects onsets from audio.
    - VideoCompiler: Compiles video based on those onsets.
    - Util functions: command wrappers and helpers.

Usage:
    python chaos_video_gen.py <directory_with_videos_and_audio>
"""

import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from editor_ui import launch_editor_ui

import numpy as np
import scipy.ndimage
from tqdm import tqdm

from util import plot_onsets, safe_tmp, ffprobe_duration, ffprobe_stream_info

# ========= CONFIG =========
LF_MIN_HZ       = 80
LF_MAX_HZ       = 5000
N_FFT           = 1024
HOP_LENGTH      = 256
ONSET_DELTA     = 0.1
RANDOM_CLIP_MIN = 0.001
RANDOM_CLIP_MAX = 30
COOLDOWN        = RANDOM_CLIP_MIN

MIN_INPUT_VIDEO_LEN = 10
DELETE_SMALL_FILES = True

mp3_filename = "audio.mp3"
date_str = datetime.now().strftime("%Y%m%d_%H%M")
output_filename = f"compiled_{date_str}.mp4"

ENABLE_BASS_DETECTION     = True
ENABLE_PLOTTING           = True
ENABLE_EXTRACT_SEGMENTS   = True
ENABLE_CONCATENATION      = True
ENABLE_ADD_AUDIO          = True
ENABLE_CLEANUP            = True


# ========= BASS DETECTOR =========
class BassDetector:
    """Detects onsets (bass hits) from a mono-filtered audio file using librosa."""

    def __init__(self, mp3_path: Path, plot_path: Path | None = None):
        """Initialize the detector with audio path and optional plot path.

        Args:
            mp3_path (Path): Path to the input MP3 file.
            plot_path (Path | None): Optional path to save the debug onset plot.

        """
        self.mp3_path = mp3_path
        self.plot_path = plot_path

    def detect(self):
        """Perform band-pass filtering and onset detection on the MP3.

        Returns:
            list[float]: List of detected onset times (in seconds).

        """
        import librosa
        from scipy.signal import butter, sosfiltfilt

        y, sr = librosa.load(self.mp3_path, mono=True)
        nyquist = 0.5 * sr
        sos = butter(4, [LF_MIN_HZ / nyquist, LF_MAX_HZ / nyquist], btype='band', output='sos')
        y_filtered = sosfiltfilt(sos, y)
        y_filtered = np.nan_to_num(y_filtered)

        onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, hop_length=HOP_LENGTH)
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
            delta=ONSET_DELTA
        )

        kept, dropped = [], []
        kept.append(onsets[0])
        last_time = onsets[0]
        for o in onsets[1:]:
            if o - last_time >= COOLDOWN:
                kept.append(o)
                last_time = o
            else:
                dropped.append(o)

        if self.plot_path:
            plot_onsets(times, onset_env, kept, dropped, self.plot_path)

        return kept


# ========= VIDEO COMPILER =========
class VideoCompiler:
    """Handles random segment extraction, concatenation, and final video/audio composition."""

    def __init__(self, folder: Path, mp3_path: Path, bass_hits: list[float]):
        """Initializes the VideoCompiler with directory and metadata.

        Args:
            folder (Path): Directory containing input videos.
            mp3_path (Path): Path to the audio track (MP3).
            bass_hits (list[float]): List of time-based cutpoints.

        """
        self.folder = folder
        self.mp3_path = mp3_path
        self.bass_hits = bass_hits
        self.video_files = []
        self.output_path = folder / output_filename
        self.segments_dir = folder / "segments"

    def collect_videos(self):
        """Scans the folder for valid video files (.mov), 
        
        optionally removing those below the minimum duration.

        """
        for f in self.folder.iterdir():
            if f.name.lower().endswith(".mov") and not f.name.startswith("._"):
                self.video_files.append(f)

        for file in tqdm(self.video_files, desc="Validating video files"):
            dur = ffprobe_duration(file)
            if dur is None or dur < MIN_INPUT_VIDEO_LEN:
                self.video_files.remove(file)
                if DELETE_SMALL_FILES:
                    file.unlink()

    def extract_random_segment(self, video_path: Path, duration: float) -> Path | None:
        """Extracts a random clip of a given duration from a video.

        Args:
            video_path (Path): Path to source video.
            duration (float): Duration of clip in seconds.

        Returns:
            Path | None: Temporary path to extracted segment, or None on failure.

        """
        vid_dur = ffprobe_duration(video_path)
        if not vid_dur or vid_dur <= 0:
            return None
        start_time = random.uniform(0, max(0, vid_dur - duration))
        seg_path = safe_tmp(".mp4")

        fps = ffprobe_stream_info(video_path)["r_fps"]
        gop = int(round(fps * 2))

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(video_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
            "-pix_fmt", "yuv420p",
            "-g", str(gop), "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-movflags", "+faststart",
            seg_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return Path(seg_path)
        except subprocess.CalledProcessError:
            return None

    def extract_segments(self):
        """Extracts segments for each bass hit interval.

        Returns:
            list[Path]: Paths to all extracted segments.

        """
        self.segments_dir.mkdir(exist_ok=True)
        random.shuffle(self.video_files)
        clip_idx = 0
        segments = []

        for i in tqdm(range(len(self.bass_hits) - 1), desc="Extracting segments"):
            dur = max(RANDOM_CLIP_MIN, min(self.bass_hits[i + 1] - self.bass_hits[i], RANDOM_CLIP_MAX))
            seg_path = self.segments_dir / f"seg_{i:04d}.mp4"
            seg_file = self.extract_random_segment(self.video_files[clip_idx], dur)
            if seg_file:
                shutil.move(seg_file, seg_path)
                segments.append(seg_path)
            clip_idx = (clip_idx + 1) % len(self.video_files)
            if clip_idx == 0:
                random.shuffle(self.video_files)

        return segments

    def concatenate_segments(self, segments: list[Path]) -> Path:
        """
        Concatenates all video segments into a single file.

        Args:
            segments (list[Path]): List of segment paths.

        Returns:
            Path: Path to the concatenated video file.

        """
        file_list_path = self.folder / "file_list.txt"
        with open(file_list_path, "w") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        temp_concat = self.folder / "concat.mp4"
        try:
            print("[INFO] Concatenating segments...")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", file_list_path, "-c", "copy", temp_concat
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", file_list_path,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                temp_concat
            ], check=True, capture_output=True)

        file_list_path.unlink()
        return temp_concat

    def add_audio(self, concat_path: Path):
        """
        Merges the audio file with the compiled video.

        Args:
            concat_path (Path): Path to the concatenated video.

        """
        print("[INFO] Adding audio...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", concat_path,
            "-i", self.mp3_path,
            "-shortest",
            "-c:v", "copy",
            "-c:a", "aac",
            str(self.output_path)
        ], check=True, capture_output=True)

    def cleanup(self, concat_path: Path):
        """Cleans up intermediate files and folders.

        Args:
            concat_path (Path): Path to concatenated video file.

        """
        print("[INFO] Cleaning up...")
        concat_path.unlink()
        shutil.rmtree(self.segments_dir)

    def compile(self):
        """Runs the full compilation process:

        Video collection, segment extraction, concatenation, and final audio sync.
        """
        self.collect_videos()
        if not self.video_files:
            print("[ERROR] No valid videos.")
            return
        segments = self.extract_segments()
        concat_path = self.concatenate_segments(segments)
        self.add_audio(concat_path)
        self.cleanup(concat_path)
        print(f"[✅ DONE] Compiled video saved to:\n{self.output_path}")


# ========= ENTRY POINT =========
def run():
    config, hits, audio_path = launch_editor_ui()
    if not audio_path or not hits:
        print("No audio or hits detected. Exiting.")
        return

    folder = audio_path.parent

    VideoCompiler(folder, audio_path, hits).compile()


if __name__ == "__main__":
    run()