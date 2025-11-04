import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# === Pfade ===
base_path = os.getcwd()
input_folder = os.path.join(base_path, "Reed_Test_wav", "analysis_output")
output_folder = os.path.join(input_folder, "harmonic_superposition")
os.makedirs(output_folder, exist_ok=True)

# === Parameter ===
duration = 0.01  # Sekunden
sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# === Alle CSV-Dateien durchgehen ===
for filename in os.listdir(input_folder):
    if filename.endswith("_harmonics.csv"):
        csv_path = os.path.join(input_folder, filename)
        print(f"Verarbeite: {filename}")

        harmonics = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    freq = float(row["Frequenz (Hz)"])
                    amp = float(row["Normierte Amplitude"])
                    if not np.isnan(freq) and amp > 0:
                        harmonics.append((freq, amp))
                except ValueError:
                    continue

        # === Superposition berechnen ===
        signal = np.zeros_like(t)
        for freq, amp in harmonics:
            signal += amp * np.sin(2 * np.pi * freq * t)

        # === Plotten ===
        plt.figure(figsize=(10, 3))
        plt.plot(t, signal)
        plt.title(f"Harmonische Superposition – {filename}")
        plt.xlabel("Zeit (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f"{filename}_superposition.png")
        plt.savefig(plot_path)
        plt.close()
