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

from datetime import datetime
from pathlib import Path
import argparse

from editor_ui import launch_editor_ui
from video_compiler import VideoCompiler


# ========= ENTRY POINT =========
def main():
    parser = argparse.ArgumentParser(description="Chaos Video Pipeline Runner")
    parser.add_argument("folder", type=Path, help="Path to the folder containing video files")
    args = parser.parse_args()

    folder = args.folder.resolve()

    config, hits, audio_path = launch_editor_ui()
    if not audio_path or not hits:
        print("No audio or hits detected. Exiting.")
        return

    VideoCompiler(config, folder, audio_path, hits).compile()


if __name__ == "__main__":
    main()

