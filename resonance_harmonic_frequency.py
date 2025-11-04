import os
import csv

# === Aktueller Arbeitsordner ===
base_path = os.getcwd()
folder_path = os.path.join(base_path, "Reed_Test_wav")
output_path = os.path.join(folder_path, "analysis_output")
os.makedirs(output_path, exist_ok=True)

# Parameter
c = 343.0  # Schallgeschwindigkeit in m/s
L = 0.13   # Länge der Luftsäule in m
num_harmonics = 20
p = 1.5    # Amplituden-Abfallfaktor

# Berechnung für geschlossene Röhre (nur ungerade Harmonische)
harmonics = []
for n in range(1, num_harmonics + 1):
    harmonic_index = 2 * n - 1  # 1, 3, 5, ...
    f_n = harmonic_index * c / (4 * L)
    a_n = 1 / (harmonic_index ** p)
    harmonics.append((harmonic_index, f_n, a_n))

# Normierung
max_amp = max([a for _, _, a in harmonics])
harmonics_norm = [(n, f, a / max_amp) for n, f, a in harmonics]

# CSV schreiben
csv_path = os.path.join(output_path, "resonance_frequency_harmonics.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Harmonische", "Frequenz (Hz)", "Normierte Amplitude"])
    for row in harmonics_norm:
        writer.writerow(row)
