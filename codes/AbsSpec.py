from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import pandas as pd
import matplotlib.pyplot as plt


def plotAbs():

    # Taille de police globale
    plt.rcParams.update({
        "font.size": 22
    })

    n_series = int(
        input("Combien de séries veux-tu tracer ? ")
    )

    plt.figure()

    for i in range(n_series):

        # Sélection du fichier
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = askopenfilename(
            title=f"Choisir le spectre {i+1}/{n_series}",
            initialdir=Path.cwd()
        )

        root.destroy()

        if not file_path:
            print(f"Série {i+1} ignorée.")
            continue

        # Nom de la légende
        legend = input(
            f"Nom à afficher dans la légende pour la série {i+1} : "
        )

        # Détection automatique de la première ligne numérique
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        start_row = None

        for j, line in enumerate(lines):

            parts = (
                line.strip()
                .replace(",", " ")
                .replace(";", " ")
                .split()
            )

            if len(parts) >= 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    start_row = j
                    break
                except ValueError:
                    pass

        if start_row is None:
            print(
                f"Aucune donnée numérique trouvée dans : {file_path}"
            )
            continue

        # Lecture du fichier
        data = pd.read_csv(
            file_path,
            skiprows=start_row,
            header=None,
            names=["Wavelength", "Absorbance"],
            sep=r"\s+|,|;",
            engine="python"
        )

        # Conversion en numérique
        data["Wavelength"] = pd.to_numeric(
            data["Wavelength"],
            errors="coerce"
        )

        data["Absorbance"] = pd.to_numeric(
            data["Absorbance"],
            errors="coerce"
        )

        data = data.dropna()

        # Tracé
        plt.plot(
            data["Wavelength"],
            data["Absorbance"],
            linewidth=2,
            label=legend
        )

    plt.xlabel(
        "Wavelength (nm)",
        fontsize=22
    )

    plt.ylabel(
        "Absorbance",
        fontsize=22
    )

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    if n_series > 1:
        plt.legend(fontsize=22)

    plt.tight_layout()
    plt.show()