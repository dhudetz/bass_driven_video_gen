from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
import librosa
import sys
from global_config import *
from bass_detector import BassDetector
from util import ffprobe_duration, save_config, load_config

class EditorUserInterface(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bass Detection Configurator")

        self.audio_path = None
        self.y = None
        self.sr = None

        # Constants and ranges
        self.param_ranges = {
            "LF_MIN_HZ": (1, 8000),
            "LF_MAX_HZ": (1, 8000),
            "ONSET_DELTA": (0.01, 1.0),
            "HOP_LENGTH": (1,2048),
            "RANDOM_CLIP_MIN": (0.001, 5),
            "RANDOM_CLIP_MAX": (5, 60),
        }        
        
        # Default configuration if config file is not loaded
        self.config = {
            "LF_MIN_HZ": 80,
            "LF_MAX_HZ": 5000,
            "ONSET_DELTA": 0.1,
            "HOP_LENGTH": 256,
            "RANDOM_CLIP_MIN": 0.001,
            "RANDOM_CLIP_MAX": 30,
        }

        self._setup()


    # ============== SETUP ===============

    def _setup(self):
        try:
            self.config = load_config()
        except:
            print("[WARNING] Failed to load config. Falling back on defaults.")
            
        if "AUDIO_PATH" in self.config.keys():
            try:
                self.audio_path = Path(self.config["AUDIO_PATH"])
                print("[INFO] Loaded audio file from config")
            except:
                print("[WARNING] Audio file is no longer there.")

        self.bass_hits = []
        self.canvas = FigureCanvas(Figure(figsize=(10, 4)))
        self.ax = self.canvas.figure.add_subplot(111)

        self._build_ui()
        if self.audio_path:
            self._choose_audio_file(self.audio_path)
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

    # =============== UPDATE ================
    def _update_config(self, key, value):
        self.config[key] = value
        self._refresh_plot()

    def _choose_audio_file(self, file_path: Path):
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select MP3", "", "Audio Files (*.mp3)")
        if file_path:
            self.config["AUDIO_PATH"] = file_path
            self.audio_path = Path(file_path)
            self.y, self.sr = librosa.load(self.audio_path, mono=True)
            self._refresh_plot()

    def _get_bass_detections(self):
        detector = BassDetector(self.audio_path)
        hits, dropped, times, onset_env = detector.detect(self.config)

        # Add a timestamp for the end of the audio clip
        mp3_dur = ffprobe_duration(self.audio_path)
        if hits and hits[-1] < mp3_dur:
            hits.append(mp3_dur)
        
        return hits, dropped, times, onset_env

    def _refresh_plot(self):
        if self.y is None:
            return

        hits, dropped, times, onset_env = self._get_bass_detections()
        
        self.bass_hits = hits
        self.bass_label.setText(f"Detected Bass Hits: {len(hits)}")

        self.ax.clear()
        self.ax.plot(times, onset_env, label="Onset Envelope", color='blue')
        self.ax.vlines(hits, 0, onset_env.max(), color='red', label='Kept', linestyle='-')
        self.ax.vlines(dropped, 0, onset_env.max(), color='orange', label='Dropped', linestyle='-')
        self.ax.set_title("Filtered Onsets")
        self.ax.set_xlabel("Time (s)")
        self.ax.legend()
        self.canvas.draw()

    def _finish_and_exit(self):
        save_config(self.config)
        self.close()

    def get_final_config(self):
        return self.config, self.bass_hits, self.audio_path


def launch_editor_ui():
    app = QApplication(sys.argv)
    window = EditorUserInterface()
    app.exec()
    return window.get_final_config()
