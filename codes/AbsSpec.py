from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def plotAbs():

    plt.rcParams.update({
        "font.size": 22
    })

    n_series = int(input("Combien de séries veux-tu tracer ? "))

    show_peaks = input("Veux-tu afficher les peaks ? (o/n) ")

    if show_peaks.lower() in ["o", "oui", "y", "yes"]:
        show_peaks = True
        prominence = float(
            input("Prominence minimale des peaks ? Exemple 0.01 : ")
        )
    else:
        show_peaks = False
        prominence = None

    plt.figure()

    for i in range(n_series):

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

        if n_series > 1:
         legend = input(
          f"Nom à afficher dans la légende pour la série {i+1} : "
        )
        else:
         legend = None
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
            print(f"Aucune donnée numérique trouvée dans : {file_path}")
            continue

        data = pd.read_csv(
            file_path,
            skiprows=start_row,
            header=None,
            names=["Wavelength", "Absorbance"],
            sep=r"\s+|,|;",
            engine="python"
        )

        data["Wavelength"] = pd.to_numeric(
            data["Wavelength"],
            errors="coerce"
        )

        data["Absorbance"] = pd.to_numeric(
            data["Absorbance"],
            errors="coerce"
        )

        data = data.dropna()

        plt.plot(
            data["Wavelength"],
            data["Absorbance"],
            linewidth=2,
            label=legend
        )

        if show_peaks:

            peaks, properties = find_peaks(
                data["Absorbance"],
                prominence=prominence
            )

            peak_x = data["Wavelength"].iloc[peaks]
            peak_y = data["Absorbance"].iloc[peaks]

            plt.scatter(
                peak_x,
                peak_y,
                s=80
            )

            for x, y in zip(peak_x, peak_y):
                plt.text(
                    x,
                    y,
                    f"{x:.0f} nm",
                    fontsize=16,
                    ha="center",
                    va="bottom"
                )

            print(f"\nPeaks trouvés pour {legend}:")
            for x, y in zip(peak_x, peak_y):
                print(f"{x:.1f} nm    Abs = {y:.4f}")

    plt.xlabel("Wavelength (nm)", fontsize=22)
    plt.ylabel("Absorbance (OD)", fontsize=22)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    if n_series > 1:
        plt.legend(fontsize=22)

    plt.tight_layout()

    save_choice = input("Veux-tu sauvegarder la figure ? (o/n) ")

    if save_choice.lower() in ["o", "oui", "y", "yes"]:

        file_name = input("Nom du fichier (sans extension) : ")

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        save_dir = askdirectory(
            title="Choisir le dossier de sauvegarde",
            initialdir=Path.cwd()
        )

        root.destroy()

        if save_dir:

            png_path = Path(save_dir) / f"{file_name}.png"
            pdf_path = Path(save_dir) / f"{file_name}.pdf"

            plt.savefig(
                png_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.savefig(
                pdf_path,
                bbox_inches="tight"
            )

            print("\nPNG sauvegardé :")
            print(png_path)

            print("\nPDF sauvegardé :")
            print(pdf_path)

        else:
            print("Aucun dossier sélectionné. Figure non sauvegardée.")

    plt.show()