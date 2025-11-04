import os
import math
import csv
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
from scipy.fft import fft, fftfreq

# === Aktueller Arbeitsordner ===
base_path = os.getcwd()
folder_path = os.path.join(base_path, "Reed_Test_wav")
output_path = os.path.join(folder_path, "analysis_output")
os.makedirs(output_path, exist_ok=True)

# === Main frequency and Pitch recognizer ===
def estimate_f0_and_pitch(audio_path, fmin=50, fmax=1000):
    # 1. Audio laden
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # 2. Grundfrequenz mit YIN (Autokorrelation basiert)
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    
    # 3. Mittelwert berechnen (robust gegen Ausreißer)
    f0_mean = np.nanmean(f0)
    
    # 4. Pitch bestimmen (MIDI + Notenname)
    midi_note = librosa.hz_to_midi(f0_mean)
    pitch_name = librosa.midi_to_note(midi_note)
    
    print(f"Geschätzte Grundfrequenz: {f0_mean:.2f} Hz")
    print(f"Entsprechender Pitch: {pitch_name} (MIDI: {midi_note:.2f})")
    
    return f0_mean, pitch_name

def extract_harmonics(xf, yf, f0, filename, output_path, num_harmonics=20, tolerance=0.03):
    """
    Extrahiert die ersten `num_harmonics` harmonischen Frequenzen und speichert sie in einer CSV-Datei.
    Die Amplituden werden normiert.
    """
    # 1. Peaks im Spektrum finden
    # prominence_threshold = np.max(yf) * 0.005  
    # peaks, _ = find_peaks(yf, prominence=prominence_threshold, distance=sample_rate//f0)
    peaks, _ = find_peaks(yf, height=np.max(yf) * 0.002, distance=sample_rate//f0)
    peak_freqs = xf[peaks]
    peak_amps = yf[peaks]

    harmonics = []
    for n in range(1, num_harmonics + 1):
        target_freq = f0 * n
        # 2. Cluster um die erwartete harmonische Frequenz finden
        mask = np.abs(peak_freqs - target_freq) < (target_freq * tolerance)
        if np.any(mask):
            # 3. Mittelwert der Frequenz und Amplitude im Cluster
            freq_cluster = peak_freqs[mask]
            amp_cluster = peak_amps[mask]
            harmonic_freq = np.mean(freq_cluster)
            harmonic_amp = np.mean(amp_cluster)
            harmonics.append((n, harmonic_freq, harmonic_amp))
        else:
            harmonics.append((n, np.nan, 0.0))  # Kein Peak gefunden

    # 4. Amplituden normieren
    max_amp = max([amp for _, _, amp in harmonics])
    harmonics_norm = [(n, freq, amp / max_amp if max_amp > 0 else 0.0) for n, freq, amp in harmonics]

    # 5. CSV speichern
    csv_path = os.path.join(output_path, f"{filename}_harmonics.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Harmonische", "Frequenz (Hz)", "Normierte Amplitude"])
        for row in harmonics_norm:
            writer.writerow(row)


# === Alle WAV-Dateien durchgehen ===
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        print(f"Analysiere: {filename}")

        # === WAV-Datei laden ===
        sample_rate, data = wavfile.read(file_path)
        if data.ndim > 1:
            data = data[:, 0]  # Nur linken Kanal verwenden

        duration = len(data) / sample_rate
        time = np.linspace(0., duration, len(data))

        # === 1. Waveform ===
        plt.figure(figsize=(10, 3))
        plt.plot(time, data)
        plt.title(f"Waveform – {filename}")
        plt.xlabel("Zeit (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{filename}_waveform.png"))
        plt.close()

        # === 2. Spektrogramm ===
        f, t, Sxx = spectrogram(data, fs=sample_rate)
        plt.figure(figsize=(10, 4))
        Sxx[Sxx == 0] = np.nan
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
        plt.title(f"Spektrogramm – {filename}")
        plt.ylabel("Frequenz (Hz)")
        plt.xlabel("Zeit (s)")
        plt.colorbar(label="Intensität (dB)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{filename}_spectrogram.png"))
        plt.close()

        # === 3. Fourier-Analyse ===
        N = len(data)
        yf = fft(data)
        xf = fftfreq(N, 1 / sample_rate)
        idx = np.where(xf >= 0)
        xf = xf[idx]
        yf = np.abs(yf[idx])

        plt.figure(figsize=(10, 3))
        plt.plot(xf, yf)
        plt.title(f"Frequenzspektrum – {filename}")
        plt.xlabel("Frequenz (Hz)")
        plt.ylabel("Amplitude")
        plt.xlim(0, sample_rate / 2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{filename}_fft.png"))
        plt.close()

        # === 4. Grundfrequenz schätzen ===
        f0, pitch = estimate_f0_and_pitch(file_path)

        # === 5. Harmonische extrahieren und speichern ===
        extract_harmonics(xf, yf, f0, filename, output_path)


