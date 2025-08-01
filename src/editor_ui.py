from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
import librosa
import sys
from global_config import *
from bass_detector import BassDetector
from util import ffprobe_duration, save_config, load_config

class DetectionWorker(QObject):
    """Worker class to run bass detection and send results to the UI thread."""
    finished = pyqtSignal(object, object, object, object)
    canceled = False

    def __init__(self, audio_path: Path, config: dict):
        super().__init__()
        self.audio_path = audio_path
        self.config = config

    @pyqtSlot()
    def run(self):
        """Performs bass detection and emits results unless canceled."""
        detector = BassDetector(self.audio_path)
        hits, dropped, times, onset_env = detector.detect(self.config)

        if self.canceled:
            return

        mp3_dur = ffprobe_duration(self.audio_path)
        if hits and hits[-1] < mp3_dur:
            hits.append(mp3_dur)

        if not self.canceled:
            self.finished.emit(hits, dropped, times, onset_env)


class EditorUserInterface(QWidget):
    """UI widget for bass detection configuration and visualization."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bass Detection Configurator")

        self.audio_path = None
        self.y = None
        self.sr = None
        self.thread = None
        self.worker = None

        # Constants and ranges
        self.param_ranges = {
            "LF_MIN_HZ": (1, 8000),
            "LF_MAX_HZ": (1, 8000),
            "ONSET_DELTA": (0.01, 1.0),
            "HOP_LENGTH": (1, 2048),
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

        self.bass_hits = []

        self._setup()

    def _setup(self):
        """Loads config, initializes UI and audio file if specified."""
        try:
            self.config = load_config()
        except Exception:
            print("[WARNING] Failed to load config. Falling back on defaults.")

        if "AUDIO_PATH" in self.config:
            try:
                self.audio_path = Path(self.config["AUDIO_PATH"])
                print("[INFO] Loaded audio file from config")
            except Exception:
                print("[WARNING] Audio file is no longer there.")

        self.canvas = FigureCanvas(Figure(figsize=(10, 4)))
        self.ax = self.canvas.figure.add_subplot(111)

        self._build_ui()
        if self.audio_path:
            self._choose_audio_file(self.audio_path)
        self.show()

    def _build_ui(self):
        """Constructs the full UI layout and event bindings."""
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Bass Detection Plot"))
        layout.addWidget(self.canvas)

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

        self.bass_label = QLabel("Detected Bass Hits: 0")
        layout.addWidget(self.bass_label)

        file_btn = QPushButton("Load Audio")
        file_btn.clicked.connect(self._choose_audio_file)
        layout.addWidget(file_btn)

        run_btn = QPushButton("Generate")
        run_btn.clicked.connect(self._finish_and_exit)
        layout.addWidget(run_btn)

        self.setLayout(layout)

    def _update_config(self, key, value):
        """Updates config and schedules a refresh.

        Args:
            key (str): Config key to update.
            value (float | int): New value.
        """
        self.config[key] = value
        self._refresh_plot()

    def _choose_audio_file(self, file_path: Path = None):
        """Loads audio file and triggers refresh.

        Args:
            file_path (Path, optional): Pre-selected file. If None, opens dialog.
        """
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select MP3", "", "Audio Files (*.mp3)")
        if file_path:
            self.config["AUDIO_PATH"] = file_path
            self.audio_path = Path(file_path)
            self.y, self.sr = librosa.load(self.audio_path, mono=True)
            self._refresh_plot()

    def _refresh_plot(self):
        """Triggers bass detection in background thread and manages cancelation."""
        if self.y is None or not self.audio_path:
            return

        # Cancel existing thread if running
        if self.thread and self.thread.isRunning():
            self.worker.canceled = True
            self.thread.quit()
            self.thread.wait()

        self.worker = DetectionWorker(self.audio_path, dict(self.config))  # pass a copy
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_detection_result)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def _on_detection_result(self, hits, dropped, times, onset_env):
        """Handles finished signal from worker to update plot.

        Args:
            hits (list): Detected bass hit times.
            dropped (list): Dropped onset times.
            times (list): All onset envelope times.
            onset_env (list): Onset envelope values.
        """
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
        """Saves config and exits the UI."""
        save_config(self.config)
        self.close()

    def get_final_config(self):
        """Returns config and results after UI closes.

        Returns:
            tuple: (config, bass_hits, audio_path)
        """
        return self.config, self.bass_hits, self.audio_path


def launch_editor_ui():
    """Launches the bass detection UI window.

    Returns:
        tuple: (config, bass_hits, audio_path)
    """
    app = QApplication(sys.argv)
    window = EditorUserInterface()
    app.exec()
    return window.get_final_config()
