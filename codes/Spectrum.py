from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import re

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def Spectrum():

    def read_spectrum_file(file_path):
        x_values = []
        y_values = []

        number_pattern = r"[+-]?\d+(?:[.,]\d+)?(?:[Ee][+-]?\d+)?"

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        has_begin_marker = any("Begin Spectral Data" in line for line in lines)
        start_reading = not has_begin_marker

        for line in lines:

            if "Begin Spectral Data" in line:
                start_reading = True
                continue

            if not start_reading:
                continue

            line = line.strip()

            if not line:
                continue

            numbers = re.findall(number_pattern, line)

            if len(numbers) >= 2:
                try:
                    x = float(numbers[0].replace(",", "."))
                    y = float(numbers[1].replace(",", "."))

                    x_values.append(x)
                    y_values.append(y)

                except ValueError:
                    pass

        if len(x_values) == 0:
            return None

        return pd.DataFrame({
            "Wavelength": x_values,
            "Signal": y_values
        })

    def choose_file(title):
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        file_path = askopenfilename(
            title=title,
            initialdir=Path.cwd()
        )

        root.destroy()

        return file_path

    def ask_yes_no(question):
        answer = input(question)
        return answer.lower() in ["o", "oui", "y", "yes"]

    plt.rcParams.update({
        "font.size": 22
    })

    colors = [
        "black",
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "magenta",
        "cyan"
    ]

    fig, ax_left = plt.subplots(figsize=(12, 7))
    ax_right = None

    left_ylabel = input("Titre de l'axe Y gauche : ")

    graph_title_choice = ask_yes_no(
        "Veux-tu ajouter un titre au graphique ? (o/n) "
    )

    if graph_title_choice:
        graph_title = input("Titre du graphique : ")
    else:
        graph_title = ""

    xlim_choice = ask_yes_no(
        "Veux-tu imposer des limites sur l'axe des x ? (o/n) "
    )

    if xlim_choice:
        x_min = float(input("Limite minimale de x : "))
        x_max = float(input("Limite maximale de x : "))
    else:
        x_min = None
        x_max = None

    n_series_left = int(
        input("Combien de séries veux-tu tracer sur l'axe Y gauche ? ")
    )

    show_peaks_left = ask_yes_no(
        "Veux-tu afficher les peaks sur l'axe Y gauche ? (o/n) "
    )

    if show_peaks_left:
        prominence_left = float(
            input("Prominence minimale des peaks gauche ? Exemple 0.01 : ")
        )
    else:
        prominence_left = None

    right_choice = ask_yes_no(
        "Veux-tu ajouter des séries sur l'axe Y droit ? (o/n) "
    )

    if right_choice:
        ax_right = ax_left.twinx()
        right_ylabel = input("Titre de l'axe Y droit : ")

        n_series_right = int(
            input("Combien de séries veux-tu tracer sur l'axe Y droit ? ")
        )

        show_peaks_right = ask_yes_no(
            "Veux-tu afficher les peaks sur l'axe Y droit ? (o/n) "
        )

        if show_peaks_right:
            prominence_right = float(
                input("Prominence minimale des peaks droit ? Exemple 0.01 : ")
            )
        else:
            prominence_right = None

    else:
        right_ylabel = None
        n_series_right = 0
        show_peaks_right = False
        prominence_right = None

    total_series = n_series_left + n_series_right

    def plot_series(axis, data, legend, color, show_peaks_axis, prominence_axis):

        axis.plot(
            data["Wavelength"],
            data["Signal"],
            linewidth=2,
            color=color,
            label=legend
        )

        if show_peaks_axis:

            peaks, properties = find_peaks(
                data["Signal"],
                prominence=prominence_axis
            )

            if len(peaks) == 0:
                return

            peak_x = data["Wavelength"].iloc[peaks]
            peak_y = data["Signal"].iloc[peaks]

            peak_data = pd.DataFrame({
                "Wavelength": peak_x,
                "Signal": peak_y,
                "Prominence": properties["prominences"]
            })

            if xlim_choice:
                peak_data = peak_data[
                    (peak_data["Wavelength"] >= x_min)
                    & (peak_data["Wavelength"] <= x_max)
                ]

            if len(peak_data) == 0:
                return

            peak_data = peak_data.sort_values(
                "Prominence",
                ascending=False
            ).head(6)

            peak_data = peak_data.sort_values("Wavelength")

            y_min = data["Signal"].min()
            y_max = data["Signal"].max()
            y_range = y_max - y_min

            if y_range == 0:
                y_range = 1

            axis.scatter(
                peak_data["Wavelength"],
                peak_data["Signal"],
                s=45,
                color=color,
                zorder=5
            )

            label_positions = []

            for x, y in zip(
                peak_data["Wavelength"],
                peak_data["Signal"]
            ):

                y_label = y + 0.04 * y_range

                while any(
                    abs(x - old_x) < 45
                    and abs(y_label - old_y) < 0.08 * y_range
                    for old_x, old_y in label_positions
                ):
                    y_label += 0.06 * y_range

                label_positions.append((x, y_label))

                axis.text(
                    x,
                    y_label,
                    f"{x:.0f} nm",
                    fontsize=10,
                    color=color,
                    ha="center",
                    va="bottom",
                    clip_on=True
                )

            current_min, current_max = axis.get_ylim()

            axis.set_ylim(
                current_min,
                max(
                    current_max,
                    y_max + 0.25 * y_range
                )
            )

            if legend is not None:
                print(f"\nPeaks trouvés pour {legend}:")
            else:
                print("\nPeaks trouvés :")

            for x, y in zip(
                peak_data["Wavelength"],
                peak_data["Signal"]
            ):
                print(f"{x:.1f} nm    Signal = {y:.4f}")

    for i in range(n_series_left):

        file_path = choose_file(
            f"Choisir le spectre gauche {i+1}/{n_series_left}"
        )

        if not file_path:
            print(f"Série gauche {i+1} ignorée.")
            continue

        if total_series > 1:
            legend = input(
                f"Nom à afficher dans la légende pour la série gauche {i+1} : "
            )
        else:
            legend = None

        data = read_spectrum_file(file_path)

        if data is None:
            print(f"Aucune donnée numérique trouvée dans : {file_path}")
            continue

        color = colors[i % len(colors)]

        plot_series(
            ax_left,
            data,
            legend,
            color,
            show_peaks_left,
            prominence_left
        )

    for i in range(n_series_right):

        file_path = choose_file(
            f"Choisir le spectre droit {i+1}/{n_series_right}"
        )

        if not file_path:
            print(f"Série droite {i+1} ignorée.")
            continue

        if total_series > 1:
            legend = input(
                f"Nom à afficher dans la légende pour la série droite {i+1} : "
            )
        else:
            legend = None

        data = read_spectrum_file(file_path)

        if data is None:
            print(f"Aucune donnée numérique trouvée dans : {file_path}")
            continue

        color = colors[(n_series_left + i) % len(colors)]

        plot_series(
            ax_right,
            data,
            legend,
            color,
            show_peaks_right,
            prominence_right
        )

    if graph_title.strip():
        ax_left.set_title(
            graph_title,
            fontsize=24,
            pad=20
        )

    ax_left.set_xlabel("Wavelength (nm)", fontsize=22)
    ax_left.set_ylabel(left_ylabel, fontsize=22)
    ax_left.tick_params(axis="both", labelsize=22)

    if ax_right is not None:
        ax_right.set_ylabel(right_ylabel, fontsize=22)
        ax_right.tick_params(axis="y", labelsize=22)

    if xlim_choice:
        ax_left.set_xlim(x_min, x_max)

    if total_series > 1:
        lines_left, labels_left = ax_left.get_legend_handles_labels()

        if ax_right is not None:
            lines_right, labels_right = ax_right.get_legend_handles_labels()
        else:
            lines_right, labels_right = [], []

        ax_left.legend(
            lines_left + lines_right,
            labels_left + labels_right,
            fontsize=14,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=True,
            facecolor="white",
            framealpha=0.9
        )

    plt.subplots_adjust(
     left=0.13,
     right=0.82,
     bottom=0.17,
     top=0.86
    ) 

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
                dpi=300
            )

            plt.savefig(
                pdf_path
            )

            print("\nPNG sauvegardé :")
            print(png_path)

            print("\nPDF sauvegardé :")
            print(pdf_path)

        else:
            print("Aucun dossier sélectionné. Figure non sauvegardée.")

    plt.show()