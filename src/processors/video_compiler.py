import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from src.common.env import logger
from tqdm import tqdm

from config.global_config import DO_FAST_VIDEO_GEN, MIN_INPUT_VIDEO_LEN, DELETE_SMALL_FILES
from src.util.util import safe_tmp, ffprobe_duration, ffprobe_stream_info
from src.util.ffmpeg import (
    FAST_COPY_EXTRACT_CMD,
    FRAME_ACCURATE_EXTRACT_CMD,
    CONCAT_COPY_CMD,
    CONCAT_REENCODE_CMD,
    ADD_AUDIO_CMD,
    format_ffmpeg_cmd
)

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
        self._duration_map: dict[Path, float] = {}
        self._weights: list[float] = []
        self._cumulative_error: float = 0.0

    def collect_videos(self):
        """Scans the folder for valid video files and caches durations."""
        valid_extensions = {".mov", ".mp4"}
        candidates: list[Path] = [
            f for f in self.folder.iterdir()
            if f.suffix.lower() in valid_extensions
            and not f.name.lower().startswith(("._", "compiled"))
        ]
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

        if self.video_files:
            durations = [self._duration_map[p] for p in self.video_files]
            min_floor = 1e-6
            self._weights = [max(d, min_floor) for d in durations]

    def extract_random_segment(
        self, video_path: Path, duration: float, vid_dur: float | None = None
    ) -> tuple[Path | None, float]:
        """Extracts a random clip of a given duration from a video."""
        if vid_dur is None:
            vid_dur = ffprobe_duration(video_path)
        if not vid_dur or vid_dur <= 0:
            return None, 0.0

        duration = min(duration, vid_dur)
        start_time = random.uniform(0, max(0, vid_dur - duration))
        seg_path = safe_tmp(".mp4")

        # Configuration toggle for choosing segment extraction type.
        if DO_FAST_VIDEO_GEN:
            cmd = FAST_COPY_EXTRACT_CMD
        else:
            cmd = FRAME_ACCURATE_EXTRACT_CMD

        cmd = format_ffmpeg_cmd(
            cmd,
            start_time=f"{start_time:.6f}",
            video_path=str(video_path),
            duration=f"{duration:.6f}",
            output_path=seg_path
        )

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            actual_dur = ffprobe_duration(seg_path) or 0.0
            return Path(seg_path), actual_dur
        except subprocess.CalledProcessError:
            return None, 0.0

    def extract_segments(self):
        """Extracts segments for each bass hit interval, compensating for timing drift."""
        self.segments_dir.mkdir(exist_ok=True)
        segments = []
        random_clip_min = self.config["RANDOM_CLIP_MIN"]
        random_clip_max = self.config["RANDOM_CLIP_MAX"]
        self._cumulative_error = 0.0

        for i in tqdm(range(len(self.bass_hits) - 1), desc="Extracting segments"):
            ideal_dur = max(random_clip_min, min(self.bass_hits[i + 1] - self.bass_hits[i], random_clip_max))
            adjusted_dur = ideal_dur - self._cumulative_error

            chosen_video = random.choices(self.video_files, weights=self._weights, k=1)[0]
            chosen_vid_dur = self._duration_map.get(chosen_video)

            seg_path = self.segments_dir / f"seg_{i:04d}.mp4"
            seg_file, actual_dur = self.extract_random_segment(chosen_video, adjusted_dur, vid_dur=chosen_vid_dur)

            if seg_file:
                shutil.move(seg_file, seg_path)
                segments.append(seg_path)
                self._cumulative_error += (actual_dur - ideal_dur)
            else:
                self._cumulative_error -= ideal_dur

        return segments

    def concatenate_segments(self, segments: list[Path]) -> Path:
        """Concatenates all video segments into a single file."""
        file_list_path = self.folder / "file_list.txt"
        with open(file_list_path, "w") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        temp_concat = self.folder / "concat.mp4"

        try:
            logger.info("Concatenating segments...")
            cmd = format_ffmpeg_cmd(
                CONCAT_COPY_CMD,
                file_list=file_list_path,
                output_path=temp_concat
            )
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            cmd = format_ffmpeg_cmd(
                CONCAT_REENCODE_CMD,
                file_list=file_list_path,
                output_path=temp_concat
            )
            subprocess.run(cmd, check=True, capture_output=True)

        file_list_path.unlink()
        return temp_concat

    def add_audio(self, concat_path: Path):
        """Merges the audio file with the compiled video."""
        logger.info("Adding audio...")
        cmd = format_ffmpeg_cmd(
            ADD_AUDIO_CMD,
            video_path=concat_path,
            audio_path=self.mp3_path,
            output_path=self.output_path
        )
        subprocess.run(cmd, check=True, capture_output=True)

    def cleanup(self, concat_path: Path):
        """Cleans up intermediate files and folders."""
        logger.info("Cleaning up...")
        concat_path.unlink()
        shutil.rmtree(self.segments_dir)

    def compile(self):
        """Runs the full compilation process."""
        self.collect_videos()
        if not self.video_files:
            logger.error("No valid videos.")
            return
        segments = self.extract_segments()
        concat_path = self.concatenate_segments(segments)
        self.add_audio(concat_path)
        self.cleanup(concat_path)
        logger.info(f"✅ Compiled video saved to:\n{self.output_path}")
