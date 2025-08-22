from pathlib import Path

# Segment extraction command template. Faster by using chunks of video.
FAST_COPY_EXTRACT_CMD = [
    "ffmpeg", "-y",
    "-ss", "{start_time}",
    "-i", "{video_path}",
    "-t", "{duration}",
    "-c:v", "copy",
    "-c:a", "copy",
    "-avoid_negative_ts", "make_zero",
    "-map", "0:v:0", "-map", "0:a?",
    "{output_path}"
]

# Slow (frame-accurate) segment extraction command template
FRAME_ACCURATE_EXTRACT_CMD = [
    "ffmpeg", "-y",
    "-ss", "{start_time}",        # start time (before input still works, but slower)
    "-i", "{video_path}",         # input video
    "-t", "{duration}",           # duration of clip
    "-c:v", "libx264",            # re-encode video for precise trimming
    "-preset", "veryfast",        # preset for speed/efficiency
    "-crf", "18",                 # high quality
    "-c:a", "aac",                # re-encode audio
    "-b:a", "192k",               # audio bitrate
    "-pix_fmt", "yuv420p",        # pixel format for broad compatibility
    "{output_path}"               # output file
]

# Concatenation (copy) command template
CONCAT_COPY_CMD = [
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", "{file_list}", "-c", "copy", "{output_path}"
]

# Concatenation (re-encode fallback) command template
CONCAT_REENCODE_CMD = [
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", "{file_list}",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "{output_path}"
]

# Add audio command template
ADD_AUDIO_CMD = [
    "ffmpeg", "-y",
    "-i", "{video_path}",
    "-i", "{audio_path}",
    "-shortest",
    "-c:v", "copy",
    "-c:a", "aac",
    "{output_path}"
]

def format_ffmpeg_cmd(cmd_template: list[str], **kwargs) -> list[str]:
    """
    Replaces placeholders in a ffmpeg command template with keyword arguments.
    
    Args:
        cmd_template (list[str]): The ffmpeg command template with placeholders.
        **kwargs: Placeholder replacements as keyword arguments.
        
    Returns:
        list[str]: Fully formatted command ready to pass to subprocess.run.
    """
    return [arg.format(**kwargs) for arg in cmd_template]
