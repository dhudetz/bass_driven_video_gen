"""
Pipeline Overview:

┌─────────────┐
│ Segment     │
│ Durations   │
└─────┬───────┘
      ↓
┌─────────────┐
│ Pick Random │
│ Video Clips │
└─────┬───────┘
      ↓
┌─────────────┐
│ Extract     │
│ Segments    │
└─────┬───────┘
      ↓
┌─────────────┐
│ Concatenate │
│ Segments    │
└─────┬───────┘
      ↓
┌─────────────┐
│ Add Audio   │
└─────┬───────┘
      ↓
┌─────────────┐
│ Final Video │
└─────────────┘
"""


import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from global_config import *
from tqdm import tqdm

from util import safe_tmp, ffprobe_duration, ffprobe_stream_info


# ========= CONSTANTS =========
date_str = datetime.now().strftime("%Y%m%d_%H%M")
output_filename = f"compiled_{date_str}.mp4"

# ========= VIDEO COMPILER =========
class VideoCompiler:
    """Handles random segment extraction, concatenation, and final video/audio composition."""

    def __init__(self, config: dict, folder: Path, mp3_path: Path, bass_hits: list[float]):
        """Initializes the VideoCompiler with directory and metadata."""
        self.config = config
        self.folder = folder
        self.mp3_path = mp3_path
        self.bass_hits = bass_hits
        self.video_files = []
        self.output_path = folder / output_filename
        self.segments_dir = folder / "segments"
        # Cache video metadata early so we can weight selection by duration.
        # Keys are Path objects, values are float durations (seconds).
        self._duration_map: dict[Path, float] = {}
        # Corresponding weights (aligned to self.video_files) used for weighted selection rolls.
        self._weights: list[float] = []

    def collect_videos(self):
        """Scan folder for valid videos, cache durations, optionally remove short ones, build selection weights."""
        valid_extensions = {".mov", ".mp4"}

        # Stage 1: discover candidates
        candidates: list[Path] = [
            f for f in self.folder.iterdir()
            if f.suffix.lower() in valid_extensions
            and not f.name.lower().startswith(("._", "compiled"))
        ]

        # Stage 2: validate + cache durations, optionally prune short files
        for file in tqdm(candidates, desc="Validating video files"):
            dur = ffprobe_duration(file)
            if dur is None or dur < MIN_INPUT_VIDEO_LEN:
                if DELETE_SMALL_FILES:
                    try:
                        file.unlink()
                    except Exception:
                        pass
                continue
            self.video_files.append(file)
            self._duration_map[file] = float(dur)

        # Build weights aligned to self.video_files, biasing by duration
        if self.video_files:
            durations = [self._duration_map[p] for p in self.video_files]
            min_floor = 1e-6
            self._weights = [max(d, min_floor) for d in durations]


    def extract_random_segment(self, video_path: Path, duration: float, vid_dur: float | None = None) -> Path | None:
        """Extracts a random clip of a given duration from a video.
        Accepts an optional cached duration to avoid redundant probes.
        Trimming is precise to avoid extra frames or overshoot."""
        if vid_dur is None:
            vid_dur = ffprobe_duration(video_path)
        if not vid_dur or vid_dur <= 0:
            return None

        # Ensure we never exceed video length
        duration = min(duration, vid_dur)
        start_time = random.uniform(0, max(0, vid_dur - duration))
        seg_path = safe_tmp(".mp4")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),       # input
            "-ss", f"{start_time:.3f}",  # seek after input for accuracy
            "-t", f"{duration:.3f}",     # exact duration
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",  # avoid timestamp drift
            seg_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return Path(seg_path)
        except subprocess.CalledProcessError:
            return None


    def extract_segments(self):
        """Extracts segments for each bass hit interval.
        Uses a length-weighted random picker so short source videos are drawn less often."""
        self.segments_dir.mkdir(exist_ok=True)
        segments = []
        random_clip_min = self.config["RANDOM_CLIP_MIN"]
        random_clip_max = self.config["RANDOM_CLIP_MAX"]

        for i in tqdm(range(len(self.bass_hits) - 1), desc="Extracting segments"):
            dur = max(random_clip_min, min(self.bass_hits[i + 1] - self.bass_hits[i], random_clip_max))

            # Length-weighted roll: longer videos have proportionally higher probability.
            # We roll per segment to avoid repeating the same short clip disproportionately.
            chosen_video = random.choices(self.video_files, weights=self._weights, k=1)[0]
            chosen_vid_dur = self._duration_map.get(chosen_video)

            seg_path = self.segments_dir / f"seg_{i:04d}.mp4"
            seg_file = self.extract_random_segment(chosen_video, dur, vid_dur=chosen_vid_dur)
            if seg_file:
                shutil.move(seg_file, seg_path)
                segments.append(seg_path)

        return segments

    def concatenate_segments(self, segments: list[Path]) -> Path:
        """Concatenates all video segments into a single file."""
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
        """Merges the audio file with the compiled video."""
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
        """Cleans up intermediate files and folders."""
        print("[INFO] Cleaning up...")
        concat_path.unlink()
        shutil.rmtree(self.segments_dir)

    def compile(self):
        """Runs the full compilation process."""
        self.collect_videos()
        if not self.video_files:
            print("[ERROR] No valid videos.")
            return
        segments = self.extract_segments()
        concat_path = self.concatenate_segments(segments)
        self.add_audio(concat_path)
        self.cleanup(concat_path)
        print(f"[✅ DONE] Compiled video saved to:\n{self.output_path}")