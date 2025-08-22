import numpy as np
import scipy.signal
import scipy.ndimage
import librosa
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

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

    def detect(self, config: dict[str, float | int]) -> list[float]:
        """Perform band-pass filtering and onset detection on the MP3.

        Args:
            config: Dictionary of configuration. Expected keys:
                - LF_MIN_HZ: Minimum frequency to detect (Hz, typically 20 Hz for deep bass)
                - LF_MAX_HZ: Maximum frequency to detect (Hz, typically 150 Hz)
                - ONSET_DELTA: Sensitivity threshold for detecting onsets (smaller = more sensitive)
                - HOP_LENGTH: Number of samples between successive analysis frames
                - RANDOM_CLIP_MIN: Minimum seconds between consecutive hits to avoid duplicates

        Returns:
            list[float]: Tuple containing:
                - hits: List of detected bass hit times (seconds)
                - dropped: List of detected hits ignored due to cooldown
                - times: Times corresponding to each frame of the onset envelope
                - onset_envelope: Computed energy curve of the filtered bass signal
        """

        # --- Step 1: Set up frequency range for bass detection ---
        bass_min_hz = config.get("LF_MIN_HZ", 20)        # Lower bound of bass (deep bass floor)
        bass_max_hz = max(config.get("LF_MAX_HZ", 150), bass_min_hz + 1)  # Upper bound (avoid 0-width band)

        # --- Step 2: Onset detection parameters ---
        onset_sensitivity = config.get("ONSET_DELTA", 0.2)  # Smaller delta = more sensitive, may detect noise
        hop_length = config.get("HOP_LENGTH", 512)          # Number of audio samples per analysis frame
        min_hit_spacing = config.get("RANDOM_CLIP_MIN", 0.25)      # Minimum seconds between consecutive hits

        # --- Step 3: Load the audio signal ---
        # sr=None ensures we load the audio at its native sample rate to avoid timing drift.
        # mono=True converts stereo to mono by averaging channels.
        audio_signal, sample_rate = librosa.load(self.mp3_path, sr=None, mono=True)

        # --- Step 4: Design a band-pass filter to isolate bass ---
        # half_sample_rate is the Nyquist frequency (half the sample rate)
        # Butterworth filter designed with order=4, applied in forward-backward mode for zero phase shift
        half_sample_rate = 0.5 * sample_rate
        sos = butter(
            N=4,
            Wn=[bass_min_hz / half_sample_rate, bass_max_hz / half_sample_rate],
            btype="band",
            output="sos"
        )
        # Apply filter to the signal
        filtered_signal = sosfiltfilt(sos, audio_signal)
        # Replace any NaNs or infinite values (can happen in audio processing)
        filtered_signal = np.nan_to_num(filtered_signal)

        # --- Step 5: Compute the onset envelope ---
        # The onset envelope represents energy changes over time, emphasizing percussive events.
        # We use librosa.onset.onset_strength which analyzes spectral flux (change in frequency content)
        onset_envelope = librosa.onset.onset_strength(
            y=filtered_signal,
            sr=sample_rate,
            hop_length=hop_length
        )
        # Median filter smooths the envelope to reduce spurious peaks
        onset_envelope = scipy.ndimage.median_filter(onset_envelope, size=5)

        # --- Step 6: Map frames to time ---
        # Each value in onset_envelope corresponds to a frame index; convert to seconds
        times = librosa.frames_to_time(
            np.arange(len(onset_envelope)),
            sr=sample_rate,
            hop_length=hop_length
        )

        # --- Step 7: Detect onsets (bass hits) ---
        # librosa.onset.onset_detect returns the times of significant peaks in the envelope
        # backtrack=False avoids snapping to an earlier local peak, which could cause drift
        detected_onsets = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=sample_rate,
            hop_length=hop_length,
            units="time",
            backtrack=False,
            pre_max=10, post_max=10,    # Local peak neighborhood to consider for detection
            pre_avg=20, post_avg=20,    # Local average neighborhood for adaptive thresholding
            delta=onset_sensitivity
        )

        # --- Step 8: Apply cooldown to avoid multiple hits from the same bass event ---
        hits, dropped = [], []
        if len(detected_onsets) > 0:
            hits.append(detected_onsets[0])
            last_time = detected_onsets[0]
            for onset_time in detected_onsets[1:]:
                if onset_time - last_time >= min_hit_spacing:
                    hits.append(onset_time)
                    last_time = onset_time
                else:
                    dropped.append(onset_time)  # Too close to previous hit; ignored

        # --- Step 9: Return results ---
        return hits, dropped, times, onset_envelope
