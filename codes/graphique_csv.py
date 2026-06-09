import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Chemins ---
RACINE       = Path(__file__).resolve().parent.parent
DOSSIER_DATA = RACINE / "data" / "data_spectrum_fluorescence"

# --- Saisie ---
nom_fichier = input("Nom du fichier .txt : ")
titre       = input("Titre du graphique : ")
label_x     = input("Nom de l'axe X : ")
label_y     = input("Nom de l'axe Y : ")

# --- Chargement ---
chemin = DOSSIER_DATA / nom_fichier
if not chemin.exists():
    print(f"Fichier introuvable : {chemin}")
    print("Fichiers disponibles :")
    for f in sorted(DOSSIER_DATA.glob("*.txt")):
        print(f"  {f.name}")
    exit(1)

df = pd.read_csv(
    chemin,
    sep="\t",
    header=None,
    names=["x", "y"],
    decimal=",",    # virgule comme séparateur décimal (format européen)
    skip_blank_lines=True,
)
df = df.dropna().sort_values("x").reset_index(drop=True)

# --- Graphique noir et blanc ---
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df["x"], df["y"], color="black", linewidth=1)

ax.set_title(titre, fontsize=14, fontweight="bold", color="black")
ax.set_xlabel(label_x, fontsize=12)
ax.set_ylabel(label_y, fontsize=12)

ax.set_facecolor("white")
fig.patch.set_facecolor("white")
ax.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
ax.spines[:].set_color("black")
ax.tick_params(colors="black")

plt.tight_layout()

# --- Sauvegarde ---
nom_sortie = RACINE / (titre.replace(" ", "_") + ".png")
plt.savefig(nom_sortie, dpi=150, facecolor="white")
print(f"Graphique sauvegardé : {nom_sortie}")
plt.show()

