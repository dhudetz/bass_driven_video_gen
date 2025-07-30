# editor_ui.py

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
import librosa
import numpy as np
import scipy.signal
import scipy.ndimage
import sys

class EditorUserInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bass Detection Configurator")

        self.audio_path = None
        self.y = None
        self.sr = None

        # Constants and ranges
        self.param_ranges = {
            "LF_MIN_HZ": (20, 500),
            "LF_MAX_HZ": (200, 8000),
            "ONSET_DELTA": (0.01, 1.0),
            "RANDOM_CLIP_MIN": (0.001, 5),
            "RANDOM_CLIP_MAX": (5, 60),
            "COOLDOWN": (0.001, 5),
            "MIN_INPUT_VIDEO_LEN": (1, 60)
        }

        self.config = {
            "LF_MIN_HZ": 80,
            "LF_MAX_HZ": 5000,
            "ONSET_DELTA": 0.1,
            "RANDOM_CLIP_MIN": 0.001,
            "RANDOM_CLIP_MAX": 30,
            "COOLDOWN": 0.001,
            "MIN_INPUT_VIDEO_LEN": 10,
        }

        self.bass_hits = []
        self.canvas = FigureCanvas(Figure(figsize=(10, 4)))
        self.ax = self.canvas.figure.add_subplot(111)

        self._build_ui()
        self.show()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Plot
        layout.addWidget(QLabel("Bass Detection Plot"))
        layout.addWidget(self.canvas)

        # Config sliders
        for key, (min_val, max_val) in self.param_ranges.items():
            box = QHBoxLayout()
            label = QLabel(f"{key}:")
            box.addWidget(label)

            if isinstance(min_val, int) and isinstance(max_val, int):
                spin = QSpinBox()
                spin.setRange(min_val, max_val)
                spin.setValue(self.config[key])
                spin.valueChanged.connect(lambda val, k=key: self._update_config(k, val))
            else:
                spin = QDoubleSpinBox()
                spin.setRange(min_val, max_val)
                spin.setSingleStep(0.01)
                spin.setDecimals(4)
                spin.setValue(self.config[key])
                spin.valueChanged.connect(lambda val, k=key: self._update_config(k, val))

            box.addWidget(spin)
            layout.addLayout(box)

        # Bass count label
        self.bass_label = QLabel("Detected Bass Hits: 0")
        layout.addWidget(self.bass_label)

        # File loader and run
        file_btn = QPushButton("Load Audio")
        file_btn.clicked.connect(self._choose_audio_file)
        layout.addWidget(file_btn)

        run_btn = QPushButton("Generate")
        run_btn.clicked.connect(self._finish_and_exit)
        layout.addWidget(run_btn)

        self.setLayout(layout)

    def _update_config(self, key, value):
        self.config[key] = value
        self._refresh_plot()

    def _choose_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select MP3", "", "Audio Files (*.mp3)")
        if file_path:
            self.audio_path = Path(file_path)
            self.y, self.sr = librosa.load(self.audio_path, mono=True)
            self._refresh_plot()

    def _refresh_plot(self):
        if self.y is None:
            return

        nyquist = 0.5 * self.sr
        sos = scipy.signal.butter(
            4,
            [self.config['LF_MIN_HZ'] / nyquist, self.config['LF_MAX_HZ'] / nyquist],
            btype='band',
            output='sos'
        )
        y_filtered = scipy.signal.sosfiltfilt(sos, self.y)
        y_filtered = np.nan_to_num(y_filtered)

        onset_env = librosa.onset.onset_strength(y=y_filtered, sr=self.sr, hop_length=256)
        onset_env = scipy.ndimage.median_filter(onset_env, size=5)
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sr, hop_length=256)

        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=256,
            units='time',
            backtrack=True,
            delta=self.config['ONSET_DELTA']
        )

        kept = []
        last_time = onsets[0] if onsets.any() else 0
        for o in onsets:
            if o - last_time >= self.config["COOLDOWN"]:
                kept.append(o)
                last_time = o

        self.bass_hits = kept
        self.bass_label.setText(f"Detected Bass Hits: {len(kept)}")

        self.ax.clear()
        self.ax.plot(times, onset_env, label="Onset Envelope", color='blue')
        self.ax.vlines(kept, 0, onset_env.max(), color='red', label='Kept', linestyle='-')
        self.ax.set_title("Filtered Onsets")
        self.ax.set_xlabel("Time (s)")
        self.ax.legend()
        self.canvas.draw()

    def _finish_and_exit(self):
        self.close()

    def get_final_config(self):
        return self.config, self.bass_hits, self.audio_path


def launch_editor_ui():
    app = QApplication(sys.argv)
    window = EditorUserInterface()
    app.exec()
    return window.get_final_config()
