from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
import librosa
import sys
from global_config import *
from bass_detector import BassDetector
from util import ffprobe_duration, save_config, load_config

# =========== CONFIG ============
DEBOUNCE_TIMEOUT = 250 # ms

# ========= WORKER CLASS ============
class DetectionWorker(QObject):
    """Worker class to run bass detection and prepare plot data.

    Emits:
        finished (list, list, list, list, float): hits, dropped, times, onset_env, max_env
    """
    finished = pyqtSignal(list, list, list, list, float)
    canceled = False

    def __init__(self, audio_path: Path, config: dict):
        super().__init__()
        self.audio_path = audio_path
        self.config = config

    @pyqtSlot()
    def run(self):
        """Performs bass detection and preprocessing for the plot."""
        detector = BassDetector(self.audio_path)
        hits, dropped, times, onset_env = detector.detect(self.config)

        if self.canceled:
            return

        mp3_dur = ffprobe_duration(self.audio_path)
        if hits and hits[-1] < mp3_dur:
            hits.append(mp3_dur)

        if self.canceled:
            return

        hits = list(hits)
        dropped = list(dropped)
        times = list(times)
        onset_env = list(onset_env)
        max_env = max(onset_env) if onset_env else 1.0

        self.finished.emit(hits, dropped, times, onset_env, max_env)

# ========== EDITOR USER INTERFACE ================
class EditorUserInterface(QWidget):
    """Main UI for bass detection configuration and preview."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bass Detection Configurator")

        self.audio_path = None
        self.y = None
        self.sr = None
        self.bass_hits = []
        self.thread = None
        self.worker = None

        self.param_ranges = {
            "LF_MIN_HZ": (1, 8000),
            "LF_MAX_HZ": (1, 8000),
            "ONSET_DELTA": (0.01, 1.0),
            "HOP_LENGTH": (1, 2048),
            "RANDOM_CLIP_MIN": (0.001, 5),
            "RANDOM_CLIP_MAX": (5, 60),
        }

        self.config = {
            "LF_MIN_HZ": 80,
            "LF_MAX_HZ": 5000,
            "ONSET_DELTA": 0.1,
            "HOP_LENGTH": 256,
            "RANDOM_CLIP_MIN": 0.001,
            "RANDOM_CLIP_MAX": 30,
        }

        self._setup()

    def _setup(self):
        """Loads configuration and builds the UI."""
        try:
            self.config = load_config()
        except Exception:
            print("[WARNING] Failed to load config. Using defaults.")

        if "AUDIO_PATH" in self.config:
            try:
                self.audio_path = Path(self.config["AUDIO_PATH"])
                print("[INFO] Loaded audio file from config")
            except Exception:
                print("[WARNING] Audio path in config is invalid.")

        self.canvas = FigureCanvas(Figure(figsize=(10, 4)))
        self.ax = self.canvas.figure.add_subplot(111)

        self._build_ui()

        if self.audio_path:
            self._choose_audio_file(self.audio_path)

        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._start_detection)

        self.show()

    def _build_ui(self):
        """Constructs layout and widget bindings."""
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
        """Updates configuration and triggers debounced refresh."""
        self.config[key] = value
        self._debounce_refresh()

    def _debounce_refresh(self):
        """Restarts the debounce timer for refresh throttling."""
        if self.debounce_timer.isActive():
            self.debounce_timer.stop()
        self.debounce_timer.start(DEBOUNCE_TIMEOUT)

    def _start_detection(self):
        """Starts detection thread after debounce delay."""
        if self.audio_path is None or self.y is None:
            return

        if self.thread and self.thread.isRunning():
            self.worker.canceled = True
            self.thread.quit()
            self.thread.wait()

        self.worker = DetectionWorker(self.audio_path, dict(self.config))
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_detection_result)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def _choose_audio_file(self, file_path: Path = None):
        """Loads an audio file and updates plot.

        Args:
            file_path (Path, optional): Pre-selected path or dialog fallback.
        """
        if not file_path:
            file_path_str, _ = QFileDialog.getOpenFileName(self, "Select MP3", "", "Audio Files (*.mp3)")
            if not file_path_str:
                return  # User cancelled
            file_path = Path(file_path_str)

        if not file_path.exists():
            print(f"[ERROR] Audio file not found: {file_path}")
            return

        self.audio_path = file_path
        self.config["AUDIO_PATH"] = str(file_path)
        self.y, self.sr = librosa.load(self.audio_path, mono=True)
        self._refresh_plot()


    def _refresh_plot(self):
        """Starts worker thread to run detection and prepare plot data."""
        if self.audio_path is None or self.y is None:
            return

        if self.thread and self.thread.isRunning():
            self.worker.canceled = True
            self.thread.quit()
            self.thread.wait()

        self.worker = DetectionWorker(self.audio_path, dict(self.config))
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_detection_result)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def _on_detection_result(self, hits, dropped, times, onset_env, max_env):
        """Handles the results from worker and updates the plot.

        Args:
            hits (list): Times of kept bass hits.
            dropped (list): Dropped onset times.
            times (list): Onset envelope time axis.
            onset_env (list): Onset envelope values.
            max_env (float): Maximum envelope value.
        """
        self.bass_hits = hits
        self.bass_label.setText(f"Detected Bass Hits: {len(hits)}")

        self.ax.clear()
        self.ax.plot(times, onset_env, label="Onset Envelope", color='blue')
        self.ax.vlines(hits, 0, max_env, color='red', label='Kept', linestyle='-')
        self.ax.vlines(dropped, 0, max_env, color='orange', label='Dropped', linestyle='-')
        self.ax.set_title("Filtered Onsets")
        self.ax.set_xlabel("Time (s)")
        self.ax.legend()
        self.canvas.draw()

    def _finish_and_exit(self):
        """Saves the current config and closes the UI."""
        save_config(self.config)
        self.close()

    def get_final_config(self):
        """Returns final config and results.

        Returns:
            tuple: (config, bass_hits, audio_path)
        """
        return self.config, self.bass_hits, self.audio_path


def launch_editor_ui():
    """Launches the main PyQt6 application.

    Returns:
        tuple: Final configuration, bass hit times, and selected audio path.
    """
    app = QApplication(sys.argv)
    window = EditorUserInterface()
    app.exec()
    return window.get_final_config()
