import numpy as np
import sounddevice as sd
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QFrame,
    QPushButton,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from scipy.fft import rfft
import copy

# Configuration
SAMPLE_RATE = 44100
FFT_SIZE = 4096
LOW_CUTOFF = 80  # Lower frequency for guitar (E2 = 82.41 Hz)
HIGH_CUTOFF = 350  # Higher frequency for guitar (E4 = 329.63 Hz)


CONCERT_PITCH = 440
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
NUM_HPS = 4  # Number of harmonics to consider for HPS


class GuitarTunerUI(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guitar Tuner")
        self.setGeometry(100, 100, 800, 600)
        self.setup_style()
        self.init_ui()

    def setup_style(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                color: white;
                font-size: 16px;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: white;
                border: 2px solid #4CAF50;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4CAF50;
            }
            QPushButton:pressed {
                background-color: #45a049;
            }
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                text-align: center;
                height: 40px;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 6px;
            }
            QFrame {
                background-color: #2d2d2d;
                border-radius: 12px;
                padding: 20px;
            }
        """
        )

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        self.title = QLabel("Guitar Tuner")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Arial", 36, QFont.Bold))
        layout.addWidget(self.title)

        # Main display frame
        display_frame = QFrame()
        display_layout = QVBoxLayout(display_frame)

        # Note and frequency display
        self.note_label = QLabel("Note: --")
        self.note_label.setAlignment(Qt.AlignCenter)
        self.note_label.setFont(QFont("Arial", 48, QFont.Bold))
        display_layout.addWidget(self.note_label)

        self.freq_label = QLabel("Frequency: -- Hz")
        self.freq_label.setAlignment(Qt.AlignCenter)
        self.freq_label.setFont(QFont("Arial", 24))
        display_layout.addWidget(self.freq_label)

        # Tuning meter
        self.tuner_bar = QProgressBar()
        self.tuner_bar.setRange(-50, 50)
        self.tuner_bar.setValue(0)
        self.tuner_bar.setFormat("%v cents")
        display_layout.addWidget(self.tuner_bar)

        # Direction label
        self.direction_label = QLabel("Tune: --")
        self.direction_label.setAlignment(Qt.AlignCenter)
        self.direction_label.setFont(QFont("Arial", 24))
        display_layout.addWidget(self.direction_label)

        layout.addWidget(display_frame)

        # Reference notes frame
        ref_frame = QFrame()
        ref_layout = QVBoxLayout(ref_frame)

        ref_title = QLabel("Standard Guitar Tuning")
        ref_title.setAlignment(Qt.AlignCenter)
        ref_title.setFont(QFont("Arial", 18, QFont.Bold))
        ref_layout.addWidget(ref_title)

        ref_notes = QLabel(
            "E2 (82.41 Hz) | A2 (110.00 Hz) | D3 (146.83 Hz)\n"
            "G3 (196.00 Hz) | B3 (246.94 Hz) | E4 (329.63 Hz)"
        )
        ref_notes.setAlignment(Qt.AlignCenter)
        ref_notes.setFont(QFont("Arial", 14))
        ref_layout.addWidget(ref_notes)

        layout.addWidget(ref_frame)

        # Control buttons
        self.start_button = QPushButton("Start Tuning")
        layout.addWidget(self.start_button)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.status_label)

        self.setLayout(layout)


class AudioProcessor:
    def get_dominant_frequency(self, audio_data):
        # Normalize audio data
        normalized_audio = audio_data.astype(float)
        normalized_audio = normalized_audio / np.max(np.abs(normalized_audio))

        # Apply windowing
        window = np.hanning(len(normalized_audio))
        windowed_signal = normalized_audio * window

        # Compute FFT
        freq_spectrum = np.abs(rfft(windowed_signal))

        # interpolate the spectrum to get a more accurate frequency
        interpolated_magnitude_spectrum = np.interp(
            np.arange(0, len(freq_spectrum), 1 / NUM_HPS),  # x_new
            np.arange(0, len(freq_spectrum)),  # x_old
            freq_spectrum,  # y_old
        )

        # Normalize the spectrum
        interpolated_magnitude_spectrum = (
            interpolated_magnitude_spectrum
            / np.linalg.norm(interpolated_magnitude_spectrum, ord=2)
        )

        # calculate HPS
        hps_spec = copy.deepcopy(interpolated_magnitude_spectrum)
        for i in range(1, NUM_HPS):
            # Get the first segment of hps_spec up to length/harmonic
            a = hps_spec[: int(np.ceil(len(interpolated_magnitude_spectrum) / (i + 1)))]

            # Get every (i+1)th sample from interpolated spectrum for harmonic comparison
            b = interpolated_magnitude_spectrum[:: (i + 1)]

            temp_hps_spec = np.multiply(a, b)

            if not any(temp_hps_spec):
                break
            hps_spec = temp_hps_spec

        # Find the peak frequency
        peak_index = np.argmax(hps_spec)
        resolution = SAMPLE_RATE / len(normalized_audio)
        peak_frequency = peak_index * resolution / NUM_HPS

        if peak_frequency < LOW_CUTOFF or peak_frequency > HIGH_CUTOFF:
            return None

        return peak_frequency


class GuitarTunerApp:
    def __init__(self):
        self.app = QApplication([])
        self.ui = GuitarTunerUI()
        self.processor = AudioProcessor()
        self.is_recording = False
        self.detected_freq = None

        self.ui.start_button.clicked.connect(self.toggle_tuning)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)

    def toggle_tuning(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.is_recording:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=FFT_SIZE,
            )
            self.stream.start()
            self.is_recording = True
            self.ui.start_button.setText("Stop Tuning")
            self.ui.status_label.setText("Recording...")

    def stop_recording(self):
        if self.is_recording and self.stream:
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            self.ui.start_button.setText("Start Tuning")
            self.ui.status_label.setText("Ready")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        mono_audio = indata[:, 0]
        self.detected_freq = self.processor.get_dominant_frequency(mono_audio)

    def update_ui(self):
        freq = self.detected_freq
        if freq:
            closest_note, closest_pitch, cents = self.find_closest_note(freq)
            self.ui.freq_label.setText(f"Frequency: {freq:.2f} Hz")
            self.ui.note_label.setText(f"Note: {closest_note}")

            # Update tuning meter
            if cents is not None:
                self.ui.tuner_bar.setValue(int(cents))

                # Update direction label
                if abs(cents) < 5:
                    self.ui.direction_label.setText("In Tune!")
                    self.ui.direction_label.setStyleSheet(
                        "color: #4CAF50; font-size: 24px;"
                    )
                else:
                    direction = "Tighten" if cents < 0 else "Loosen"
                    self.ui.direction_label.setText(f"{direction} the string")
                    self.ui.direction_label.setStyleSheet(
                        "color: #ff4444; font-size: 24px;"
                    )
            else:
                self.ui.tuner_bar.setValue(0)
                self.ui.direction_label.setText("Tune: --")

    def find_closest_note(self, pitch):
        i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
        closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
        closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
        # Calculate cents difference
        cents = 1200 * np.log2(pitch / closest_pitch)
        return closest_note, closest_pitch, cents

    def run(self):
        self.ui.show()
        self.app.exec()


if __name__ == "__main__":
    tuner = GuitarTunerApp()
    tuner.run()
