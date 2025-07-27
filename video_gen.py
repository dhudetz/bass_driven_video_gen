import os
import subprocess
import tempfile

# === CONFIG ===
video_folder = "/Volumes/DJI_VIDEOS/DCIM/100MEDIA"
mp3_filename = "krabs.mp3"
output_filename = "compiled_output.mp4"

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

# === STEP 2: CREATE TEMP FILE LIST IN /tmp ===
temp_dir = tempfile.gettempdir()
file_list_txt = os.path.join(temp_dir, "ffmpeg_file_list.txt")
with open(file_list_txt, "w") as f:
    for video_path in video_files:
        f.write(f"file '{video_path}'\n")

print(f"[INFO] Using temp file list: {file_list_txt}")

# === STEP 3: CONCAT VIDEOS TO TEMP OUTPUT ===
temp_concat = os.path.join(temp_dir, "temp_concat.mov")
print("[INFO] Concatenating videos...")
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

# === STEP 4: ADD/LOOP AUDIO ===
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
except FileNotFoundError:
    pass

print(f"[✅ DONE] Compiled video saved to:\n{output_path}")

