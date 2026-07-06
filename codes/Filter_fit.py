from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def find_file_by_name(filename, root="."):
    matches = list(Path(root).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Fichier introuvable : {filename}")
    return matches[0]


def choose_file():
    print("\nChoix du fichier :")
    print("1 - Parcourir manuellement")
    print("2 - Écrire le nom du fichier")

    choice = input("Choix [1/2] : ").strip()

    if choice == "1":
        Tk().withdraw()
        path = askopenfilename(
            title="Choisir le fichier",
            filetypes=[("CSV/TXT files", "*.csv *.txt"), ("All files", "*.*")]
        )
        return Path(path)

    elif choice == "2":
        name = input("Nom du fichier avec extension : ").strip()
        return find_file_by_name(name)

    else:
        raise ValueError("Choix invalide.")


def load_data(path):
    df = pd.read_csv(path)

    print("\nColonnes trouvées :")
    for i, col in enumerate(df.columns):
        print(f"{i} - {col}")

    x_col = int(input("Index de la colonne fréquence : "))
    y_col = int(input("Index de la colonne amplitude : "))

    x = df.iloc[:, x_col].to_numpy(dtype=float)
    y = df.iloc[:, y_col].to_numpy(dtype=float)

    return x, y


def moving_average(y, window):
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def apply_filter(x, y):
    use_filter = input("\nAppliquer un filtre ? [o/n] : ").strip().lower()

    if use_filter != "o":
        return y, "Aucun filtre"

    print("\nType de filtre :")
    print("1 - Savitzky-Golay")
    print("2 - Moyenne mobile")

    choice = input("Choix [1/2] : ").strip()

    if choice == "1":
        window = int(input("Taille de fenêtre impaire, ex. 51 : "))
        order = int(input("Ordre polynomial, ex. 2 ou 3 : "))

        if window % 2 == 0:
            window += 1
            print(f"Fenêtre corrigée à {window}, car elle doit être impaire.")

        y_filtered = savgol_filter(y, window_length=window, polyorder=order)
        label = f"Savitzky-Golay, fenêtre={window}, ordre={order}"

    elif choice == "2":
        window = int(input("Taille de fenêtre, ex. 20 : "))
        y_filtered = moving_average(y, window)
        label = f"Moyenne mobile, fenêtre={window}"

    else:
        raise ValueError("Choix invalide.")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, alpha=0.35, label="Données brutes")
    plt.plot(x, y_filtered, label="Données filtrées")
    plt.xlabel("Fréquence")
    plt.ylabel("Amplitude")
    plt.title(label)
    plt.legend()
    plt.tight_layout()
    plt.show()

    ok = input("Le filtre est-il acceptable ? [o/n] : ").strip().lower()

    if ok != "o":
        print("Arrêt du programme.")
        raise SystemExit

    return y_filtered, label


def lorentzian(x, A, x0, gamma, y0):
    return y0 + A / (1 + ((x - x0) / (gamma / 2))**2)


def gaussian(x, A, x0, sigma, y0):
    return y0 + A * np.exp(-((x - x0)**2) / (2 * sigma**2))


def polynomial(x, *coeffs):
    return np.polyval(coeffs, x)


def fit_curves(x, y):
    n = int(input("\nCombien de courbes veux-tu fitter ? : "))

    results = []
    fit_total = np.zeros_like(y)

    for i in range(n):
        print(f"\nFit {i+1}")
        print("1 - Lorentzienne")
        print("2 - Gaussienne")
        print("3 - Polynomiale")

        choice = input("Type de fit [1/2/3] : ").strip()

        xmin = float(input("Fréquence minimale de la région à fitter : "))
        xmax = float(input("Fréquence maximale de la région à fitter : "))

        mask = (x >= xmin) & (x <= xmax)
        x_fit = x[mask]
        y_fit = y[mask]

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
                maxfev=10000
            )

            y_model = lorentzian(x, *popt)
            A, x0, gamma, y0 = popt

            results.append({
                "type": "lorentzienne",
                "centre": x0,
                "amplitude": A,
                "FWHM": abs(gamma),
                "offset": y0
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
                maxfev=10000
            )

            y_model = gaussian(x, *popt)
            A, x0, sigma, y0 = popt
            fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(sigma)

            results.append({
                "type": "gaussienne",
                "centre": x0,
                "amplitude": A,
                "FWHM": fwhm,
                "sigma": sigma,
                "offset": y0
            })

        elif choice == "3":
            order = int(input("Ordre du polynôme : "))

            coeffs = np.polyfit(x_fit, y_fit, order)
            y_model = polynomial(x, *coeffs)

            results.append({
                "type": f"polynôme ordre {order}",
                "centre": np.nan,
                "amplitude": np.nan,
                "FWHM": np.nan,
                "coefficients": coeffs
            })

        else:
            raise ValueError("Choix invalide.")

        fit_total += y_model

    return results, fit_total


def save_figure():
    answer = input("\nSauvegarder le graphique ? [o/n] : ").strip().lower()

    if answer == "o":
        Tk().withdraw()
        folder = askdirectory(title="Choisir le dossier de sauvegarde")
        name = input("Nom du fichier sans extension : ").strip()

        path_png = Path(folder) / f"{name}.png"
        path_pdf = Path(folder) / f"{name}.pdf"

        plt.savefig(path_png, dpi=300)
        plt.savefig(path_pdf)

        print(f"Graphique sauvegardé :")
        print(path_png)
        print(path_pdf)


def save_table(results):
    df = pd.DataFrame(results)

    print("\nTableau des résultats :")
    print(df)

    answer = input("\nSauvegarder le tableau ? [o/n] : ").strip().lower()

    if answer == "o":
        Tk().withdraw()
        folder = askdirectory(title="Choisir le dossier de sauvegarde")
        name = input("Nom du fichier sans extension : ").strip()

        path = Path(folder) / f"{name}.csv"
        df.to_csv(path, index=False)

        print(f"Tableau sauvegardé : {path}")


def save_filtered_data(x, y_filtered):
    answer = input("\nSauvegarder les données filtrées en CSV ? [o/n] : ").strip().lower()

    if answer == "o":
        Tk().withdraw()
        folder = askdirectory(title="Choisir le dossier de sauvegarde")
        name = input("Nom du fichier sans extension : ").strip()

        path = Path(folder) / f"{name}.csv"

        df = pd.DataFrame({
            "Frequence": x,
            "Amplitude_filtree": y_filtered
        })

        df.to_csv(path, index=False)
        print(f"Données filtrées sauvegardées : {path}")


def main():
    path = choose_file()
    print(f"\nFichier choisi : {path}")

    x, y = load_data(path)

    y_filtered, filter_label = apply_filter(x, y)

    results, fit_total = fit_curves(x, y_filtered)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_filtered, label="Données filtrées", linewidth=1.5)
    plt.plot(x, fit_total, label="Fit total", linewidth=2)

    plt.xlabel("Fréquence")
    plt.ylabel("Amplitude")
    plt.title("Données filtrées avec fits")
    plt.legend()
    plt.tight_layout()
    plt.show()

    save_figure()
    save_table(results)
    save_filtered_data(x, y_filtered)


if __name__ == "__main__":
    main()