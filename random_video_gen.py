import os
import subprocess
import tempfile
import random

# === CONFIG ===
video_folder = "/Users/dhudetz/scripts/video_gen/footage"
mp3_filename = "krabs.mp3"
output_filename = "compiled_output2.mp4"

# === STEP 1: VERIFY PATHS ===
print(f"[INFO] Scanning: {video_folder}")
if not os.path.isdir(video_folder):
    print(f"[ERROR] Folder does not exist: {video_folder}")
    exit(1)

# Get all valid .mov files (case-insensitive, no '._' files)
all_files = os.listdir(video_folder)
video_files = sorted([
    os.path.join(video_folder, f)
    for f in all_files
    if f.lower().endswith(".mov") and not f.startswith("._")
])

if not video_files:
    print("[ERROR] No valid .mov video files found.")
    exit(1)
else:
    print(f"[INFO] Found {len(video_files)} video files.")

# Check MP3 exists
mp3_path = os.path.join(video_folder, mp3_filename)
if not os.path.isfile(mp3_path):
    print(f"[ERROR] MP3 file not found: {mp3_path}")
    exit(1)

# === STEP 2: EXTRACT 5-SECOND SEGMENTS AND ADD CHAOS ===
print("[INFO] Extracting 5-second segments and filtering short clips...")
segments = []
temp_dir = tempfile.gettempdir()

for video_path in video_files:
    # Get duration using ffprobe
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        duration_str = subprocess.check_output(probe_cmd).decode().strip()
        duration = float(duration_str)
    except:
        print(f"[WARNING] Failed to get duration for {video_path}, skipping.")
        continue

    if duration < 5:
        print(f"[INFO] Skipping short clip: {video_path} ({duration:.2f}s)")
        continue

    num_segments = int(duration // 5)
    print(f"[INFO] Extracting {num_segments} segments from {video_path}")

    for i in range(num_segments):
        temp_segment = tempfile.NamedTemporaryFile(suffix='.mov', delete=False, dir=temp_dir).name
        extract_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-ss", str(i * 5),
            "-t", "5",
            "-c", "copy",
            temp_segment
        ]
        subprocess.run(extract_cmd, check=True)
        segments.append(temp_segment)

if not segments:
    print("[ERROR] No valid 5-second segments found.")
    exit(1)

# Add chaos: Shuffle the segments randomly
random.shuffle(segments)
print(f"[INFO] Shuffled {len(segments)} segments for chaotic editing.")

# === STEP 3: CREATE TEMP FILE LIST IN /tmp ===
file_list_txt = os.path.join(temp_dir, "ffmpeg_file_list.txt")
with open(file_list_txt, "w") as f:
    for segment_path in segments:
        f.write(f"file '{segment_path}'\n")

print(f"[INFO] Using temp file list: {file_list_txt}")

# === STEP 4: CONCAT SEGMENTS TO TEMP OUTPUT ===
temp_concat = os.path.join(temp_dir, "temp_concat.mov")
print("[INFO] Concatenating shuffled segments...")
concat_cmd = [
    "ffmpeg",
    "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", file_list_txt,
    "-c", "copy",
    temp_concat
]
subprocess.run(concat_cmd, check=True)

# === STEP 5: ADD/LOOP AUDIO ===
output_path = os.path.join(video_folder, output_filename)
print("[INFO] Adding audio...")
final_cmd = [
    "ffmpeg",
    "-y",
    "-i", temp_concat,
    "-stream_loop", "-1",
    "-i", mp3_path,
    "-shortest",
    "-c:v", "copy",
    "-c:a", "aac",
    output_path
]
subprocess.run(final_cmd, check=True)

# === CLEANUP ===
print("[INFO] Cleaning up temporary files...")
try:
    os.remove(temp_concat)
    os.remove(file_list_txt)
    for segment in segments:
        os.remove(segment)
except FileNotFoundError:
    pass

print(f"[✅ DONE] Compiled video saved to:\n{output_path}")
