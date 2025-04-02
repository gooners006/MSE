import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.fftpack import rfft
import copy
import time
import matplotlib.pyplot as plt

# Constants
SAMPLE_RATE = 44100
DURATION = 1  # seconds
NUM_HPS = 5
LOW_CUTOFF = 20  # Hz
HIGH_CUTOFF = 1000  # Hz
CONCERT_PITCH = 440.0
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
GUITAR_STRINGS = {"E2": 82.41, "A2": 110.00, "D3": 146.83, "G3": 196.00, "B3": 246.94, "E4": 329.63}

class AudioProcessor:
    def get_dominant_frequency(self, audio_data):
        normalized_audio = audio_data.astype(float) / np.max(np.abs(audio_data))
        window = np.hanning(len(normalized_audio))
        windowed_signal = normalized_audio * window
        freq_spectrum = abs(rfft(windowed_signal))[: len(normalized_audio) // 2]
        
        interpolated_magnitude_spectrum = np.interp(
            np.arange(0, len(freq_spectrum), 1 / NUM_HPS),
            np.arange(0, len(freq_spectrum)),
            freq_spectrum,
        )
        interpolated_magnitude_spectrum /= np.linalg.norm(interpolated_magnitude_spectrum, ord=2)
        hps_spec = copy.deepcopy(interpolated_magnitude_spectrum)
        
        for i in range(1, NUM_HPS):
            a = hps_spec[: int(np.ceil(len(interpolated_magnitude_spectrum) / (i + 1)))]
            b = interpolated_magnitude_spectrum[:: (i + 1)]
            temp_hps_spec = np.multiply(a, b)
            if not any(temp_hps_spec):
                break
            hps_spec = temp_hps_spec
        
        peak_index = np.argmax(hps_spec)
        resolution = SAMPLE_RATE / len(normalized_audio)
        peak_frequency = peak_index * resolution / NUM_HPS

        #debug
        currentfrequency = st.empty()
        currentfrequency.write(f"Peak Frequency: {peak_frequency:.2f} Hz")
        
        return peak_frequency if LOW_CUTOFF < peak_frequency < HIGH_CUTOFF else None,freq_spectrum

    def find_closest_note(self, pitch):
        i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
        closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
        closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
        cents = 1200 * np.log2(pitch / closest_pitch)
        return closest_note, closest_pitch, cents

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return np.squeeze(audio_data)

def get_tuning_direction(cents):
    if cents > 5:
        return ":red[Tune Down ⬇]"
    elif cents < -5:
        return ":red[Tune Up ⬆]"
    else:
        return ":green[In Tune ✅]"

# Streamlit UI
st.title("🎸 Guitar Tuner")
st.write("Pluck a string and keep playing. The tuner will continuously analyze the sound.")
active_recording = st.button("Start Tuning")
if active_recording:
    processor = AudioProcessor()
    tuning_display = st.empty()
    placeHolder_display = st.empty()
    plot_display = st.empty()
    tuning_display.write(f"Listening for audio...")
    while active_recording:
        audio_data = record_audio()
        frequency,freq_spectrum = processor.get_dominant_frequency(audio_data)
        
        if frequency:

            note, closest_pitch, cents = processor.find_closest_note(frequency)
            string_name, target_freq = min(GUITAR_STRINGS.items(), key=lambda x: abs(x[1] - frequency))
            tuning_direction = get_tuning_direction(cents)
            st.toast('Note detected!', icon='😍')
            tuning_display.markdown(
                f"**Detected Frequency:** {frequency:.2f} Hz  \
                **Closest Note:** {note} ({closest_pitch:.2f} Hz)  \
                **String:** {string_name} (Target: {target_freq} Hz) \n  \
                **Tuning Status:** : {tuning_direction}"
            )
            # Display DFT Plot
            fig, ax = plt.subplots()
            ax.plot(np.linspace(0, SAMPLE_RATE / 2, len(freq_spectrum)), freq_spectrum)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.set_title("DFT Spectrum of the Detected Pitch")
            ax.set_xlim([LOW_CUTOFF, HIGH_CUTOFF])
            plot_display.pyplot(fig)

        else:
            placeHolder_display.write("No clear frequency detected. Try again!")
        
        time.sleep(DURATION)  # Avoid high CPU usage
else:
    st.info("Click the 'Record' button to start recording audio.")


# to run : streamlit run .\streamlitSample.py