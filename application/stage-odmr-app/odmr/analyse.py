"""Logique pure des filtres et des fits (lorentzien, gaussien, polynomial)."""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Filtres
# ---------------------------------------------------------------------------
def moyenne_mobile(y, fenetre):
    kernel = np.ones(fenetre) / fenetre
    return np.convolve(y, kernel, mode="same")


def filtrer(y, methode, **params):
    """Applique un filtre à `y`.

    methode : "aucun", "savgol" (params: fenetre, ordre) ou
              "moyenne_mobile" (params: fenetre).
    Retourne (y_filtre, label_descriptif).
    """
    if methode == "aucun":
        return np.asarray(y, dtype=float), "Aucun filtre"

    if methode == "savgol":
        fenetre = int(params["fenetre"])
        ordre = int(params["ordre"])
        if fenetre % 2 == 0:
            fenetre += 1  # la fenêtre doit être impaire
        if ordre >= fenetre:
            raise ValueError("L'ordre polynomial doit être inférieur à la fenêtre.")
        y_f = savgol_filter(y, window_length=fenetre, polyorder=ordre)
        return y_f, f"Savitzky-Golay (fenêtre={fenetre}, ordre={ordre})"

    if methode == "moyenne_mobile":
        fenetre = int(params["fenetre"])
        y_f = moyenne_mobile(y, fenetre)
        return y_f, f"Moyenne mobile (fenêtre={fenetre})"

    raise ValueError(f"Méthode de filtre inconnue : {methode}")


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------
def lorentzienne(x, A, x0, gamma, y0):
    return y0 + A / (1 + ((x - x0) / (gamma / 2)) ** 2)


def gaussienne(x, A, x0, sigma, y0):
    return y0 + A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def polynomiale(x, *coeffs):
    return np.polyval(coeffs, x)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def fitter_region(x, y, type_fit, xmin, xmax, ordre=2, creux=False):
    """Ajuste un modèle sur la région [xmin, xmax].

    type_fit : "lorentzienne", "gaussienne" ou "polynomiale".
    creux    : True si le pic est un creux (dip ODMR) plutôt qu'une bosse.

    Retourne (dict_resultats, y_modele_sur_tout_x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x >= xmin) & (x <= xmax)
    if mask.sum() < 4:
        raise ValueError("Région trop étroite : moins de 4 points à fitter.")
    x_fit, y_fit = x[mask], y[mask]

    if type_fit == "polynomiale":
        coeffs = np.polyfit(x_fit, y_fit, int(ordre))
        y_model = polynomiale(x, *coeffs)
        resultat = {
            "type": f"polynôme ordre {int(ordre)}",
            "centre": np.nan,
            "amplitude": np.nan,
            "FWHM": np.nan,
            "coefficients": list(coeffs),
        }
        return resultat, y_model

    # Estimations initiales communes aux modèles à pic
    baseline = np.median(y_fit)
    if creux:
        A0 = y_fit.min() - baseline           # amplitude négative
        x0 = x_fit[np.argmin(y_fit)]
    else:
        A0 = y_fit.max() - baseline
        x0 = x_fit[np.argmax(y_fit)]
    largeur0 = (xmax - xmin) / 5

    if type_fit == "lorentzienne":
        popt, pcov = curve_fit(
            lorentzienne, x_fit, y_fit,
            p0=[A0, x0, largeur0, baseline], maxfev=10000,
        )
        A, x0, gamma, y0 = popt
        incert = np.sqrt(np.diag(pcov))
        resultat = {
            "type": "lorentzienne",
            "centre": x0,
            "incert_centre": incert[1],
            "amplitude": A,
            "FWHM": abs(gamma),
            "offset": y0,
        }
        return resultat, lorentzienne(x, *popt)

    if type_fit == "gaussienne":
        popt, pcov = curve_fit(
            gaussienne, x_fit, y_fit,
            p0=[A0, x0, largeur0, baseline], maxfev=10000,
        )
        A, x0, sigma, y0 = popt
        incert = np.sqrt(np.diag(pcov))
        fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(sigma)
        resultat = {
            "type": "gaussienne",
            "centre": x0,
            "incert_centre": incert[1],
            "amplitude": A,
            "FWHM": fwhm,
            "sigma": sigma,
            "offset": y0,
        }
        return resultat, gaussienne(x, *popt)

    raise ValueError(f"Type de fit inconnu : {type_fit}")
