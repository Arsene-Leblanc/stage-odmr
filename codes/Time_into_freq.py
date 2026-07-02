import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# PARAMÈTRES EXPÉRIMENTAUX (à modifier uniquement ici)
# ==================================================

# Fichier de données
file_path = "../data/MWsweep/06-30-105MHz-110MHz-RT0.5-S10000.csv"

# Sweep en fréquence
freq_start = 105         # MHz
freq_end = 110           # MHz
n_freq = 10000           # nombre de fréquences du sweep

# Acquisition
sampling_rate = 100      # Hz
time_per_freq = 0.5      # secondes

# Affichage
figure_size = (10, 5)
line_width = 1

# Sauvegarde
save_csv = True
save_png = True
save_pdf = True

csv_filename = "donnees_moyennees.csv"
png_filename = "signal_vs_frequence.png"
pdf_filename = "signal_vs_frequence.pdf"

# ==================================================
# CALCULS AUTOMATIQUES
# ==================================================

samples_per_freq = int(sampling_rate * time_per_freq)
expected_points = n_freq * samples_per_freq

print(f"Échantillons par fréquence : {samples_per_freq}")
print(f"Nombre total de points attendus : {expected_points}")

# ==================================================
# LECTURE DES DONNÉES
# ==================================================

values = np.loadtxt(file_path)
values = np.ravel(values)

if len(values) < expected_points:
    raise ValueError(
        f"Seulement {len(values)} points trouvés "
        f"alors que {expected_points} étaient attendus."
    )

# On garde exactement les points correspondant au sweep
values = values[:expected_points]

# ==================================================
# MOYENNAGE
# ==================================================

values = values.reshape(n_freq, samples_per_freq)

signal_mean = values.mean(axis=1)
signal_std = values.std(axis=1)

frequencies = np.linspace(freq_start, freq_end, n_freq)

# ==================================================
# SAUVEGARDE DES DONNÉES
# ==================================================

df = pd.DataFrame({
    "Frequency": frequencies,
    "Signal_mean": signal_mean,
    "Signal_std": signal_std
})

if save_csv:
    df.to_csv(csv_filename, index=False)

# ==================================================
# GRAPHIQUE
# ==================================================

plt.figure(figsize=figure_size)

plt.plot(
    frequencies,
    signal_mean,
    linewidth=line_width
)

plt.xlabel("Fréquence (MHz)")
plt.ylabel("Signal LIA moyen (V)")
plt.title("Signal LIA moyen en fonction de la fréquence")

plt.grid()

plt.tight_layout()

if save_png:
    plt.savefig(png_filename, dpi=300)

if save_pdf:
    plt.savefig(pdf_filename)

plt.show()