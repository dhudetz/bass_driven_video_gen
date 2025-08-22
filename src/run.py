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
    """Entry point for the Chaos Video Pipeline Runner.

    Parses folder argument, launches the editor UI for configuration, and runs the video compiler.
    """
    parser = argparse.ArgumentParser(description="Chaos Video Pipeline Runner")
    parser.add_argument("folder", type=Path, help="Path to the folder containing video files")
    args = parser.parse_args()

    folder = args.folder.resolve()

    # GUI pops up on entry for selecting generator settings.
    config, bass_hits, audio_path, _success = launch_editor_ui()

    # Exit if editor was closed or returned invalid values.
    if not _success:
        print("Editor closed or invalid input. Exiting.")
        return

    # Run the video compiler; progress is visible in the CLI.
    video_compiler = VideoCompiler(config, folder, audio_path, bass_hits)
    video_compiler.compile()


if __name__ == "__main__":
    main()

