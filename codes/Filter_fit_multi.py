from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


# ============================================================
# Lecture robuste de fichiers CSV/TXT avec entêtes variables
# ============================================================

def is_numeric_line(line):
    """
    Détecte si une ligne contient au moins deux nombres.
    Gère :
    345,274   10,75122
    900.0,0.0347
    900.0;0.0347
    """
    number = r"[+-]?\d+(?:[.,]\d+)?(?:[Ee][+-]?\d+)?"
    nums = re.findall(number, line)
    return len(nums) >= 2


def extract_numbers(line):
    number = r"[+-]?\d+(?:[.,]\d+)?(?:[Ee][+-]?\d+)?"
    nums = re.findall(number, line)

    values = []
    for n in nums:
        values.append(float(n.replace(",", ".")))

    return values


def read_spectrum_file(path):
    """
    Lit automatiquement un fichier contenant des données numériques.
    Ignore les entêtes jusqu'à trouver des lignes numériques.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data = []

    for line in lines:
        if is_numeric_line(line):
            values = extract_numbers(line)
            data.append(values)

    if len(data) == 0:
        raise ValueError("Aucune donnée numérique trouvée dans le fichier.")

    min_cols = min(len(row) for row in data)
    data = [row[:min_cols] for row in data]

    df = pd.DataFrame(data)

    return df


# ============================================================
# Choix de fichier
# ============================================================

def find_file_by_name(filename, root="."):
    matches = list(Path(root).rglob(filename))

    if not matches:
        raise FileNotFoundError(f"Fichier introuvable : {filename}")

    if len(matches) > 1:
        print("\nPlusieurs fichiers trouvés :")
        for i, m in enumerate(matches):
            print(f"{i} - {m}")

        idx = int(input("Choisis l'index du fichier : "))
        return matches[idx]

    return matches[0]


def choose_file():
    print("\nChoix du fichier :")
    print("1 - Parcourir manuellement")
    print("2 - Écrire le nom du fichier")

    choice = input("Choix [1/2] : ").strip()

    if choice == "1":
        Tk().withdraw()
        path = askopenfilename(
            title="Choisir un fichier",
            filetypes=[
                ("Fichiers CSV/TXT", "*.csv *.txt"),
                ("Tous les fichiers", "*.*")
            ]
        )
        return Path(path)

    elif choice == "2":
        name = input("Nom du fichier avec extension : ").strip()
        return find_file_by_name(name)

    else:
        raise ValueError("Choix invalide.")


# ============================================================
# Filtres
# ============================================================

def moving_average(y, window):
    window = int(window)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def apply_filter(x, y, series_name):
    print(f"\nFiltrage pour la série : {series_name}")
    use_filter = input("Appliquer un filtre ? [o/n] : ").strip().lower()

    if use_filter != "o":
        return y, "aucun filtre"

    print("\nChoix du filtre :")
    print("1 - Savitzky-Golay")
    print("2 - Moyenne mobile")

    choice = input("Choix [1/2] : ").strip()

    if choice == "1":
        window = int(input("Taille de fenêtre impaire, ex. 21, 51, 101 : "))
        order = int(input("Ordre polynomial, ex. 2 ou 3 : "))

        if window % 2 == 0:
            window += 1
            print(f"Fenêtre corrigée à {window}.")

        if window >= len(y):
            raise ValueError("La fenêtre est plus grande que la série.")

        y_filtered = savgol_filter(y, window_length=window, polyorder=order)
        filter_name = f"Savitzky-Golay fenêtre={window}, ordre={order}"

    elif choice == "2":
        window = int(input("Taille de fenêtre, ex. 10, 20, 50 : "))

        if window >= len(y):
            raise ValueError("La fenêtre est plus grande que la série.")

        y_filtered = moving_average(y, window)
        filter_name = f"Moyenne mobile fenêtre={window}"

    else:
        raise ValueError("Choix invalide.")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, alpha=0.35, label="Données brutes")
    plt.plot(x, y_filtered, label="Données filtrées")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{series_name} — {filter_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    ok = input("Le filtre est-il correct ? [o/n] : ").strip().lower()

    if ok != "o":
        print("Arrêt du programme.")
        raise SystemExit

    return y_filtered, filter_name


# ============================================================
# Modèles de fit
# ============================================================

def lorentzian(x, A, x0, gamma, y0):
    return y0 + A / (1 + ((x - x0) / (gamma / 2)) ** 2)


def gaussian(x, A, x0, sigma, y0):
    return y0 + A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def polynomial(x, *coeffs):
    return np.polyval(coeffs, x)


def fit_one_series(x, y, series_name):
    n = int(input(f"\nCombien de courbes veux-tu fitter pour {series_name} ? : "))

    results = []
    fit_curves = []

    for i in range(n):
        print(f"\nFit {i+1} pour {series_name}")
        print("1 - Lorentzienne")
        print("2 - Gaussienne")
        print("3 - Polynomiale")

        choice = input("Choix [1/2/3] : ").strip()

        xmin = float(input("x minimum de la région à fitter : "))
        xmax = float(input("x maximum de la région à fitter : "))

        mask = (x >= xmin) & (x <= xmax)
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 5:
            raise ValueError("Pas assez de points dans cette région.")

        if choice == "1":
            A0 = y_fit.max() - np.median(y_fit)
            x0 = x_fit[np.argmax(y_fit)]
            gamma0 = (xmax - xmin) / 5
            y0 = np.median(y_fit)

            popt, _ = curve_fit(
                lorentzian,
                x_fit,
                y_fit,
                p0=[A0, x0, gamma0, y0],
                maxfev=20000
            )

            y_model = lorentzian(x, *popt)
            A, x0, gamma, y0 = popt

            results.append({
                "serie": series_name,
                "fit": i + 1,
                "type": "lorentzienne",
                "centre": x0,
                "amplitude": A,
                "FWHM": abs(gamma),
                "offset": y0,
                "xmin_fit": xmin,
                "xmax_fit": xmax
            })

        elif choice == "2":
            A0 = y_fit.max() - np.median(y_fit)
            x0 = x_fit[np.argmax(y_fit)]
            sigma0 = (xmax - xmin) / 5
            y0 = np.median(y_fit)

            popt, _ = curve_fit(
                gaussian,
                x_fit,
                y_fit,
                p0=[A0, x0, sigma0, y0],
                maxfev=20000
            )

            y_model = gaussian(x, *popt)
            A, x0, sigma, y0 = popt
            fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(sigma)

            results.append({
                "serie": series_name,
                "fit": i + 1,
                "type": "gaussienne",
                "centre": x0,
                "amplitude": A,
                "FWHM": fwhm,
                "sigma": sigma,
                "offset": y0,
                "xmin_fit": xmin,
                "xmax_fit": xmax
            })

        elif choice == "3":
            order = int(input("Ordre du polynôme : "))
            coeffs = np.polyfit(x_fit, y_fit, order)
            y_model = polynomial(x, *coeffs)

            result = {
                "serie": series_name,
                "fit": i + 1,
                "type": f"polynôme ordre {order}",
                "centre": np.nan,
                "amplitude": np.nan,
                "FWHM": np.nan,
                "xmin_fit": xmin,
                "xmax_fit": xmax
            }

            for j, c in enumerate(coeffs):
                result[f"coef_{j}"] = c

            results.append(result)

        else:
            raise ValueError("Choix invalide.")

        fit_curves.append((i + 1, y_model))

    return results, fit_curves


# ============================================================
# Sauvegardes
# ============================================================

def save_current_figure():
    answer = input("\nSauvegarder le graphique ? [o/n] : ").strip().lower()

    if answer == "o":
        Tk().withdraw()
        folder = askdirectory(title="Choisir le dossier")
        name = input("Nom du fichier sans extension : ").strip()

        png_path = Path(folder) / f"{name}.png"
        pdf_path = Path(folder) / f"{name}.pdf"

        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path)

        print(f"Sauvegardé : {png_path}")
        print(f"Sauvegardé : {pdf_path}")


def save_dataframe(df, default_name="resultats"):
    answer = input("\nSauvegarder ce tableau ? [o/n] : ").strip().lower()

    if answer == "o":
        Tk().withdraw()
        folder = askdirectory(title="Choisir le dossier")
        name = input(f"Nom du fichier sans extension [{default_name}] : ").strip()

        if name == "":
            name = default_name

        path = Path(folder) / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"Tableau sauvegardé : {path}")


def save_filtered_series(all_series):
    answer = input("\nProduire un CSV des données filtrées ? [o/n] : ").strip().lower()

    if answer != "o":
        return

    Tk().withdraw()
    folder = askdirectory(title="Choisir le dossier")

    for serie in all_series:
        safe_name = serie["name"].replace(" ", "_")
        path = Path(folder) / f"{safe_name}_filtre.csv"

        df = pd.DataFrame({
            serie["x_label"]: serie["x"],
            serie["y_label"] + "_filtre": serie["y_filtered"]
        })

        df.to_csv(path, index=False)
        print(f"Données filtrées sauvegardées : {path}")


# ============================================================
# Programme principal
# ============================================================

def main():
    n_series = int(input("Combien de séries de données veux-tu entrer ? : "))

    all_series = []
    all_results = []

    for s in range(n_series):
        print(f"\n==============================")
        print(f"Série {s+1}")
        print(f"==============================")

        path = choose_file()
        print(f"Fichier choisi : {path}")

        df = read_spectrum_file(path)

        print("\nAperçu des colonnes numériques détectées :")
        print(df.head())

        print("\nLe fichier contient les colonnes suivantes :")
        for i in range(df.shape[1]):
            print(f"{i} - colonne {i}")

        x_col = int(input("\nQuelle colonne correspond à l'axe x ? : "))
        x_label = input("Nom physique de cette colonne, ex. Fréquence (GHz), Longueur d'onde (nm) : ")

        y_col = int(input("Quelle colonne correspond à l'axe y ? : "))
        y_label = input("Nom physique de cette colonne, ex. Amplitude (V), Intensité, Absorbance : ")

        series_name = input("Nom de cette série pour la légende : ").strip()
        if series_name == "":
            series_name = path.stem

        x = df.iloc[:, x_col].to_numpy(dtype=float)
        y = df.iloc[:, y_col].to_numpy(dtype=float)

        y_filtered, filter_name = apply_filter(x, y, series_name)

        results, fit_curves = fit_one_series(x, y_filtered, series_name)

        all_results.extend(results)

        all_series.append({
            "name": series_name,
            "path": path,
            "x": x,
            "y": y,
            "y_filtered": y_filtered,
            "x_label": x_label,
            "y_label": y_label,
            "filter": filter_name,
            "fit_curves": fit_curves
        })

    # ========================================================
    # Graphique final combiné
    # ========================================================

    show_all = input("\nAfficher toutes les séries dans un seul graphique ? [o/n] : ").strip().lower()

    if show_all == "o":
        plt.figure(figsize=(11, 7))

        for serie in all_series:
            x = serie["x"]
            y_filtered = serie["y_filtered"]

            plt.plot(x, y_filtered, label=f"{serie['name']} filtrée")

            for fit_number, y_fit in serie["fit_curves"]:
                plt.plot(
                    x,
                    y_fit,
                    linestyle="--",
                    label=f"{serie['name']} fit {fit_number}"
                )

        plt.xlabel(all_series[0]["x_label"])
        plt.ylabel(all_series[0]["y_label"])
        plt.title("Données filtrées et fits")
        plt.legend()
        plt.tight_layout()
        save_current_figure()
        plt.show()

        

    # ========================================================
    # Tableau final combiné
    # ========================================================

    results_df = pd.DataFrame(all_results)

    print("\nTableau combiné des résultats :")
    print(results_df)

    save_dataframe(results_df, default_name="tableau_fits")

    save_filtered_series(all_series)


if __name__ == "__main__":
    main()