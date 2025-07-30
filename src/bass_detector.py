import numpy as np
import scipy.signal
import scipy.ndimage
import librosa
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

# =========== CONSTANTS ============
COOLDOWN = 1

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

    def detect(self, config: dict[str, float | int]):
        """Perform band-pass filtering and onset detection on the MP3.

        Args:
            config: Dictionary of configuration.

        Returns:
            list[float]: List of detected onset times (in seconds).

        """
        lf_min_hz = config["LF_MIN_HZ"]
        lf_max_hz = config["LF_MAX_HZ"]
        onset_delta = config["ONSET_DELTA"]
        hop_length = config["HOP_LENGTH"]
        cooldown = config["COOLDOWN"]

        y, sr = librosa.load(self.mp3_path, mono=True)
        nyquist = 0.5 * sr
        sos = butter(4, [lf_min_hz / nyquist, lf_max_hz / nyquist], btype='band', output='sos')
        y_filtered = sosfiltfilt(sos, y)
        y_filtered = np.nan_to_num(y_filtered)

        onset_env = librosa.onset.onset_strength(y=y_filtered, sr=sr, hop_length=hop_length)
        onset_env = scipy.ndimage.median_filter(onset_env, size=5)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)

        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            units='time',
            backtrack=True,
            pre_max=10, post_max=10,
            pre_avg=20, post_avg=20,
            delta=onset_delta
        )

        kept, dropped = [], []
        kept.append(onsets[0])
        last_time = onsets[0]
        for o in onsets[1:]:
            if o - last_time >= cooldown:
                kept.append(o)
                last_time = o
            else:
                dropped.append(o)

        return kept, dropped, times, onset_env