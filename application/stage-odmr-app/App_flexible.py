"""
Application ODMR — Streamlit

Outils :
1. Traitement de sweeps temps -> fréquence
2. Filtre & Fit flexible :
   - plusieurs séries
   - choix manuel des colonnes
   - filtres Savitzky-Golay / moyenne mobile
   - somme de Lorentziennes ou de Gaussiennes
   - fit polynomial
   - personnalisation des axes, couleurs, épaisseurs et légendes
   - export CSV et PNG

Lancement :
    streamlit run app_odmr_flexible.py
"""

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# Dépendances déjà présentes dans ton projet pour la page 1
from odmr.traitement import traiter_sweeps, detecter_pic_principal


st.set_page_config(
    page_title="Analyse ODMR",
    page_icon="🔬",
    layout="wide",
)

# ============================================================
# STYLE GLOBAL
# ============================================================

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
})

DEFAULT_COLORS = [
    "black", "red", "blue", "green", "orange",
    "purple", "brown", "magenta", "cyan", "teal"
]

FIT_COLORS = [
    "darkred", "navy", "darkgreen", "darkorange",
    "indigo", "maroon", "teal", "darkmagenta"
]


# ============================================================
# UTILITAIRES GÉNÉRAUX
# ============================================================

def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def fig_png_bytes(fig, dpi=300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def charger_voltages(donnees_brutes: bytes) -> np.ndarray:
    texte = donnees_brutes.decode("utf-8", errors="ignore")

    if "," in texte and "\t" not in texte:
        texte = texte.replace(",", " ")

    valeurs = np.array(texte.split(), dtype=float)

    if valeurs.size == 0:
        raise ValueError("Aucune valeur numérique trouvée.")

    return valeurs


def lire_tableau_flexible(uploaded_file) -> pd.DataFrame:
    """
    Essaie plusieurs méthodes de lecture afin d'accepter :
    - CSV avec virgules, points-virgules ou tabulations
    - TXT à colonnes séparées par espaces
    - fichiers avec ou sans en-tête
    """
    raw = uploaded_file.getvalue()

    essais = [
        dict(sep=None, engine="python"),
        dict(sep=","),
        dict(sep=";"),
        dict(sep="\t"),
        dict(sep=r"\s+", engine="python"),
    ]

    for kwargs in essais:
        for header in ("infer", None):
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    header=header,
                    comment="#",
                    **kwargs
                )

                if df.shape[1] >= 2:
                    # Convertir les colonnes numériques possibles
                    converted = df.copy()
                    numeric_count = 0

                    for col in converted.columns:
                        s = (
                            converted[col]
                            .astype(str)
                            .str.replace(",", ".", regex=False)
                        )
                        num = pd.to_numeric(s, errors="coerce")

                        if num.notna().sum() >= max(3, int(0.5 * len(num))):
                            converted[col] = num
                            numeric_count += 1

                    if numeric_count >= 2:
                        converted = converted.dropna(how="all")
                        return converted
            except Exception:
                pass

    raise ValueError(
        "Impossible de détecter au moins deux colonnes numériques."
    )


def nom_colonne(col) -> str:
    return str(col)


def safe_slider_bounds(x: np.ndarray):
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))

    if xmin == xmax:
        xmax = xmin + 1.0

    return xmin, xmax


# ============================================================
# FILTRES
# ============================================================

def moyenne_mobile(y: np.ndarray, fenetre: int) -> np.ndarray:
    if fenetre < 1:
        raise ValueError("La fenêtre doit être positive.")

    serie = pd.Series(y)
    return (
        serie.rolling(
            window=fenetre,
            center=True,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )


def appliquer_filtre(
    y: np.ndarray,
    methode: str,
    fenetre: int = 11,
    ordre: int = 2,
):
    if methode == "Aucun":
        return y.copy(), "Aucun filtre"

    if methode == "Savitzky-Golay":
        if fenetre % 2 == 0:
            fenetre += 1

        if fenetre >= len(y):
            fenetre = len(y) - 1 if len(y) % 2 == 0 else len(y)

        if fenetre % 2 == 0:
            fenetre -= 1

        if fenetre < 3:
            raise ValueError("Série trop courte pour Savitzky-Golay.")

        if ordre >= fenetre:
            raise ValueError(
                "L'ordre polynomial doit être inférieur à la fenêtre."
            )

        return (
            savgol_filter(y, window_length=fenetre, polyorder=ordre),
            f"Savitzky-Golay, fenêtre={fenetre}, ordre={ordre}",
        )

    if methode == "Moyenne mobile":
        if fenetre >= len(y):
            raise ValueError(
                "La fenêtre doit être plus petite que la série."
            )

        return (
            moyenne_mobile(y, fenetre),
            f"Moyenne mobile, fenêtre={fenetre}",
        )

    raise ValueError("Méthode de filtre inconnue.")


# ============================================================
# MODÈLES DE FIT
# ============================================================

def lorentzienne(x, amplitude, centre, fwhm):
    fwhm = np.maximum(np.abs(fwhm), np.finfo(float).eps)
    return amplitude / (1.0 + 4.0 * ((x - centre) / fwhm) ** 2)


def gaussienne(x, amplitude, centre, sigma):
    sigma = np.maximum(np.abs(sigma), np.finfo(float).eps)
    return amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def somme_lorentziennes(x, offset, pente, *params):
    y = offset + pente * x

    for i in range(0, len(params), 3):
        amplitude, centre, fwhm = params[i:i + 3]
        y = y + lorentzienne(x, amplitude, centre, fwhm)

    return y


def somme_gaussiennes(x, offset, pente, *params):
    y = offset + pente * x

    for i in range(0, len(params), 3):
        amplitude, centre, sigma = params[i:i + 3]
        y = y + gaussienne(x, amplitude, centre, sigma)

    return y


@dataclass
class FitOutput:
    resultats: pd.DataFrame
    y_total: np.ndarray
    y_composantes: list
    masque: np.ndarray
    type_fit: str


def effectuer_fit_somme(
    x,
    y,
    xmin,
    xmax,
    type_fit,
    pics_initiaux,
    offset_initial,
    pente_initiale,
    afficher_composantes=False,
):
    masque = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
    x_fit = x[masque]
    y_fit = y[masque]

    if len(x_fit) < max(6, 3 * len(pics_initiaux) + 2):
        raise ValueError(
            "Pas assez de points dans la région sélectionnée."
        )

    p0 = [offset_initial, pente_initiale]

    bornes_basses = [-np.inf, -np.inf]
    bornes_hautes = [np.inf, np.inf]

    largeur_region = max(xmax - xmin, np.finfo(float).eps)

    for pic in pics_initiaux:
        amplitude = float(pic["amplitude"])
        centre = float(pic["centre"])
        largeur = max(abs(float(pic["largeur"])), largeur_region / 100000)

        p0.extend([amplitude, centre, largeur])

        bornes_basses.extend([-np.inf, xmin, largeur_region / 100000])
        bornes_hautes.extend([np.inf, xmax, largeur_region * 2])

    if type_fit == "Somme de Lorentziennes":
        modele = somme_lorentziennes
        largeur_nom = "FWHM"
    elif type_fit == "Somme de Gaussiennes":
        modele = somme_gaussiennes
        largeur_nom = "sigma"
    else:
        raise ValueError("Type de fit inconnu.")

    popt, pcov = curve_fit(
        modele,
        x_fit,
        y_fit,
        p0=p0,
        bounds=(bornes_basses, bornes_hautes),
        maxfev=200000,
    )

    erreurs = np.sqrt(np.diag(pcov))

    y_total = np.full_like(x, np.nan, dtype=float)
    y_total[masque] = modele(x_fit, *popt)

    offset, pente = popt[:2]
    params = popt[2:]

    composantes = []
    lignes = []

    for i in range(len(pics_initiaux)):
        amplitude, centre, largeur = params[3 * i: 3 * i + 3]

        e_amplitude, e_centre, e_largeur = erreurs[
            2 + 3 * i: 2 + 3 * i + 3
        ]

        if type_fit == "Somme de Lorentziennes":
            fwhm = abs(largeur)
        else:
            fwhm = (
                2 * np.sqrt(2 * np.log(2)) * abs(largeur)
            )

        composante = np.full_like(x, np.nan, dtype=float)
        composante[masque] = (
            offset
            + pente * x_fit
            + (
                lorentzienne(x_fit, amplitude, centre, largeur)
                if type_fit == "Somme de Lorentziennes"
                else gaussienne(x_fit, amplitude, centre, largeur)
            )
        )

        composantes.append(
            {
                "label": f"Composante {i + 1}",
                "y": composante,
            }
        )

        lignes.append({
            "pic": i + 1,
            "type_fit_global": type_fit,
            "amplitude": amplitude,
            "erreur_amplitude": e_amplitude,
            "centre": centre,
            "erreur_centre": e_centre,
            largeur_nom: abs(largeur),
            f"erreur_{largeur_nom}": e_largeur,
            "FWHM": fwhm,
            "offset_global": offset,
            "pente_globale": pente,
            "xmin_fit": xmin,
            "xmax_fit": xmax,
        })

    return FitOutput(
        resultats=pd.DataFrame(lignes),
        y_total=y_total,
        y_composantes=composantes if afficher_composantes else [],
        masque=masque,
        type_fit=type_fit,
    )


def effectuer_fit_polynomial(x, y, xmin, xmax, ordre):
    masque = np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
    x_fit = x[masque]
    y_fit = y[masque]

    if len(x_fit) <= ordre:
        raise ValueError("Pas assez de points pour cet ordre polynomial.")

    coeffs = np.polyfit(x_fit, y_fit, ordre)

    y_total = np.full_like(x, np.nan, dtype=float)
    y_total[masque] = np.polyval(coeffs, x_fit)

    ligne = {
        "type_fit_global": f"Polynôme ordre {ordre}",
        "xmin_fit": xmin,
        "xmax_fit": xmax,
    }

    for i, coefficient in enumerate(coeffs):
        puissance = ordre - i
        ligne[f"coefficient_x^{puissance}"] = coefficient

    return FitOutput(
        resultats=pd.DataFrame([ligne]),
        y_total=y_total,
        y_composantes=[],
        masque=masque,
        type_fit=f"Polynôme ordre {ordre}",
    )


# ============================================================
# CONFIGURATION GRAPHIQUE
# ============================================================

def controles_axes(prefix, x, y):
    st.markdown("#### Axes et apparence générale")

    c1, c2, c3 = st.columns(3)

    with c1:
        titre = st.text_input(
            "Titre",
            value="Données filtrées et fit",
            key=f"{prefix}_titre",
        )

        xlabel = st.text_input(
            "Nom de l'axe x",
            value="Fréquence",
            key=f"{prefix}_xlabel",
        )

        ylabel = st.text_input(
            "Nom de l'axe y",
            value="Amplitude",
            key=f"{prefix}_ylabel",
        )

    xmin_auto, xmax_auto = safe_slider_bounds(x)
    ymin_auto = float(np.nanmin(y))
    ymax_auto = float(np.nanmax(y))

    if ymin_auto == ymax_auto:
        ymax_auto = ymin_auto + 1.0

    with c2:
        limites_x_auto = st.checkbox(
            "Limites x automatiques",
            value=True,
            key=f"{prefix}_xauto",
        )

        xmin = st.number_input(
            "xmin",
            value=xmin_auto,
            format="%.8g",
            disabled=limites_x_auto,
            key=f"{prefix}_xmin",
        )

        xmax = st.number_input(
            "xmax",
            value=xmax_auto,
            format="%.8g",
            disabled=limites_x_auto,
            key=f"{prefix}_xmax",
        )

    with c3:
        limites_y_auto = st.checkbox(
            "Limites y automatiques",
            value=True,
            key=f"{prefix}_yauto",
        )

        ymin = st.number_input(
            "ymin",
            value=ymin_auto,
            format="%.8g",
            disabled=limites_y_auto,
            key=f"{prefix}_ymin",
        )

        ymax = st.number_input(
            "ymax",
            value=ymax_auto,
            format="%.8g",
            disabled=limites_y_auto,
            key=f"{prefix}_ymax",
        )

    c4, c5, c6, c7 = st.columns(4)

    with c4:
        largeur_fig = st.number_input(
            "Largeur figure",
            min_value=4.0,
            max_value=30.0,
            value=12.0,
            step=0.5,
            key=f"{prefix}_figw",
        )

    with c5:
        hauteur_fig = st.number_input(
            "Hauteur figure",
            min_value=3.0,
            max_value=20.0,
            value=6.0,
            step=0.5,
            key=f"{prefix}_figh",
        )

    with c6:
        grille = st.checkbox(
            "Afficher la grille",
            value=True,
            key=f"{prefix}_grid",
        )

    with c7:
        afficher_legende = st.checkbox(
            "Afficher la légende",
            value=True,
            key=f"{prefix}_legend",
        )

    return {
        "titre": titre,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "xlim": None if limites_x_auto else (xmin, xmax),
        "ylim": None if limites_y_auto else (ymin, ymax),
        "figsize": (largeur_fig, hauteur_fig),
        "grille": grille,
        "afficher_legende": afficher_legende,
    }


def appliquer_axes(ax, config):
    ax.set_title(config["titre"])
    ax.set_xlabel(config["xlabel"])
    ax.set_ylabel(config["ylabel"])

    if config["xlim"] is not None:
        ax.set_xlim(*config["xlim"])

    if config["ylim"] is not None:
        ax.set_ylim(*config["ylim"])

    if config["grille"]:
        ax.grid(True, linestyle="--", alpha=0.35)

    if config["afficher_legende"]:
        ax.legend()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔬 Analyse ODMR")

page = st.sidebar.radio(
    "Outil",
    [
        "1 · Traitement de sweeps",
        "2 · Filtre & Fit flexible",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Les données restent dans la session Streamlit. "
    "Télécharge les résultats avant de fermer."
)


# ============================================================
# PAGE 1 — TRAITEMENT DE SWEEPS
# ============================================================

if page.startswith("1"):
    st.title("Traitement de sweeps ODMR — temps → fréquence")

    st.markdown(
        "L'outil regroupe les échantillons associés à chaque fréquence, "
        "calcule chaque sweep, puis produit la moyenne et l'erreur standard."
    )

    fichiers = st.file_uploader(
        "Fichiers de données brutes",
        type=None,
        accept_multiple_files=True,
        key="sweeps_files",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        n_freq = st.number_input(
            "Nombre de fréquences",
            min_value=2,
            value=200,
            step=1,
        )

    with c2:
        f_debut = st.number_input(
            "Fréquence de départ (GHz)",
            value=1.5,
            format="%.6f",
        )

    with c3:
        f_fin = st.number_input(
            "Fréquence de fin (GHz)",
            value=2.5,
            format="%.6f",
        )

    if st.button(
        "Lancer le traitement",
        type="primary",
        disabled=not fichiers,
    ):
        donnees = []
        noms = []

        for f in fichiers:
            try:
                v = charger_voltages(f.getvalue())
                donnees.append(v)
                noms.append(f.name)
                st.caption(f"{f.name} : {v.size} points")
            except Exception as exc:
                st.warning(f"{f.name} : {exc}")

        if not donnees:
            st.error("Aucun fichier lisible.")
            st.stop()

        try:
            res = traiter_sweeps(
                donnees,
                noms,
                int(n_freq),
                f_debut,
                f_fin,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        for msg in res.messages:
            st.info(msg)

        st.success(
            f"{res.n_sweeps} sweep(s) traité(s). "
            f"Réduction de bruit attendue ≈ √N = "
            f"{res.facteur_reduction_bruit:.2f}"
        )

        fig1, ax1 = plt.subplots(figsize=(10, 5))

        couleurs = plt.cm.viridis(
            np.linspace(0, 1, res.n_sweeps)
        )

        for courbe, nom, couleur in zip(
            res.sweeps,
            res.noms,
            couleurs,
        ):
            ax1.plot(
                res.axe_frequence,
                courbe,
                color=couleur,
                linewidth=0.9,
                alpha=0.85,
                label=nom if res.n_sweeps <= 12 else None,
            )

        ax1.set_title(
            f"Tous les sweeps ({res.n_sweeps} acquisitions)"
        )
        ax1.set_xlabel("Fréquence (GHz)")
        ax1.set_ylabel("Amplitude")

        if res.n_sweeps <= 12:
            ax1.legend(fontsize=7)

        ax1.grid(True, alpha=0.25)
        fig1.tight_layout()

        pic = detecter_pic_principal(
            res.axe_frequence,
            res.moyenne,
        )

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        ax2.errorbar(
            res.axe_frequence,
            res.moyenne,
            yerr=res.erreur_moyenne,
            fmt="-",
            color="black",
            linewidth=1.3,
            ecolor="tab:red",
            elinewidth=0.6,
            capsize=1.5,
            errorevery=max(1, int(n_freq) // 50),
            label=(
                f"Moyenne de {res.n_sweeps} sweeps "
                f"(± σ/√N)"
            ),
        )

        if pic:
            freq_pic, amp_pic, _ = pic
            ax2.plot(
                freq_pic,
                amp_pic,
                "v",
                color="tab:blue",
                markersize=9,
                label=f"Pic : {freq_pic:.6f} GHz",
            )

        ax2.set_title("Moyenne ODMR")
        ax2.set_xlabel("Fréquence (GHz)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend()
        fig2.tight_layout()

        cg1, cg2 = st.columns(2)

        with cg1:
            st.pyplot(fig1)

        with cg2:
            st.pyplot(fig2)

        df_moyenne = pd.DataFrame({
            "Frequence_GHz": res.axe_frequence,
            "Amplitude_Moyenne": res.moyenne,
            "Erreur_Standard": res.erreur_moyenne,
        })

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(
            zip_buffer,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(
                "00_Moyenne_Globale.csv",
                df_moyenne.to_csv(index=False),
            )

            for courbe, nom in zip(res.sweeps, res.noms):
                base = nom.rsplit(".", 1)[0]

                df_sweep = pd.DataFrame({
                    "Frequence_GHz": res.axe_frequence,
                    "Amplitude": courbe,
                })

                archive.writestr(
                    f"freq_{base}.csv",
                    df_sweep.to_csv(index=False),
                )

            archive.writestr(
                "01_tous_les_sweeps.png",
                fig_png_bytes(fig1),
            )

            archive.writestr(
                "02_moyenne.png",
                fig_png_bytes(fig2),
            )

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "⬇️ Moyenne globale (CSV)",
                csv_bytes(df_moyenne),
                "00_Moyenne_Globale.csv",
                "text/csv",
            )

        with d2:
            st.download_button(
                "⬇️ Tout télécharger (ZIP)",
                zip_buffer.getvalue(),
                "traitement_odmr.zip",
                "application/zip",
            )


# ============================================================
# PAGE 2 — FILTRE & FIT FLEXIBLE
# ============================================================

else:
    st.title("Filtre & Fit flexible")

    st.markdown(
        """
        Cette page permet de charger plusieurs séries, choisir les colonnes,
        appliquer un filtre et effectuer un **fit simultané sous forme d'une
        somme de pics**. Le graphique final n'affiche par défaut que le fit
        total, et non les courbes de fit séparées.
        """
    )

    fichiers = st.file_uploader(
        "Fichiers CSV ou TXT",
        type=["csv", "txt", "dat"],
        accept_multiple_files=True,
        key="fit_files",
    )

    if not fichiers:
        st.info("Charge au moins un fichier pour commencer.")
        st.stop()

    series = []

    for i, fichier in enumerate(fichiers):
        st.markdown("---")
        st.subheader(f"Série {i + 1} — {fichier.name}")

        try:
            df = lire_tableau_flexible(fichier)
        except Exception as exc:
            st.error(f"Lecture impossible : {exc}")
            continue

        colonnes = list(df.columns)

        c1, c2, c3 = st.columns(3)

        with c1:
            col_x = st.selectbox(
                "Colonne x",
                colonnes,
                index=0,
                format_func=nom_colonne,
                key=f"colx_{i}",
            )

        with c2:
            col_y = st.selectbox(
                "Colonne y",
                colonnes,
                index=min(1, len(colonnes) - 1),
                format_func=nom_colonne,
                key=f"coly_{i}",
            )

        with c3:
            nom_serie = st.text_input(
                "Nom de la série",
                value=fichier.name.rsplit(".", 1)[0],
                key=f"serie_name_{i}",
            )

        x = pd.to_numeric(
            df[col_x].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).to_numpy(dtype=float)

        y = pd.to_numeric(
            df[col_y].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        ).to_numpy(dtype=float)

        valide = np.isfinite(x) & np.isfinite(y)
        x = x[valide]
        y = y[valide]

        ordre_tri = np.argsort(x)
        x = x[ordre_tri]
        y = y[ordre_tri]

        st.markdown("#### Filtre")

        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            methode = st.selectbox(
                "Méthode",
                ["Aucun", "Savitzky-Golay", "Moyenne mobile"],
                key=f"filter_method_{i}",
            )

        with fc2:
            fenetre_max = max(3, min(len(y) - 1, 501))

            fenetre_defaut = min(21, fenetre_max)

            if fenetre_defaut % 2 == 0:
                fenetre_defaut -= 1

            fenetre = st.number_input(
                "Fenêtre",
                min_value=3,
                max_value=max(3, fenetre_max),
                value=max(3, fenetre_defaut),
                step=2 if methode == "Savitzky-Golay" else 1,
                disabled=(methode == "Aucun"),
                key=f"filter_window_{i}",
            )

        with fc3:
            ordre = st.number_input(
                "Ordre Savitzky-Golay",
                min_value=1,
                max_value=10,
                value=2,
                disabled=(methode != "Savitzky-Golay"),
                key=f"filter_order_{i}",
            )

        try:
            y_filtre, nom_filtre = appliquer_filtre(
                y,
                methode,
                int(fenetre),
                int(ordre),
            )
        except Exception as exc:
            st.error(str(exc))
            continue

        st.caption(nom_filtre)

        st.markdown("#### Apparence de la série")

        sc1, sc2, sc3, sc4 = st.columns(4)

        with sc1:
            couleur_donnees = st.color_picker(
                "Couleur des données",
                value="#000000" if i == 0 else "#1f77b4",
                key=f"data_color_{i}",
            )

        with sc2:
            couleur_fit = st.color_picker(
                "Couleur du fit total",
                value="#8B0000",
                key=f"fit_color_{i}",
            )

        with sc3:
            epaisseur_donnees = st.number_input(
                "Épaisseur données",
                min_value=0.2,
                max_value=10.0,
                value=1.5,
                step=0.1,
                key=f"data_lw_{i}",
            )

        with sc4:
            epaisseur_fit = st.number_input(
                "Épaisseur fit",
                min_value=0.2,
                max_value=10.0,
                value=3.0,
                step=0.1,
                key=f"fit_lw_{i}",
            )

        st.markdown("#### Ajustement")

        fitc1, fitc2, fitc3 = st.columns(3)

        with fitc1:
            type_fit = st.selectbox(
                "Type de fit",
                [
                    "Somme de Lorentziennes",
                    "Somme de Gaussiennes",
                    "Polynôme",
                    "Aucun fit",
                ],
                key=f"fit_type_{i}",
            )

        xmin_data, xmax_data = safe_slider_bounds(x)

        with fitc2:
            xmin_fit = st.number_input(
                "x minimum du fit",
                value=xmin_data,
                format="%.8g",
                key=f"xmin_fit_{i}",
            )

        with fitc3:
            xmax_fit = st.number_input(
                "x maximum du fit",
                value=xmax_data,
                format="%.8g",
                key=f"xmax_fit_{i}",
            )

        fit_output = None

        if type_fit in (
            "Somme de Lorentziennes",
            "Somme de Gaussiennes",
        ):
            mc1, mc2, mc3 = st.columns(3)

            with mc1:
                n_pics = st.number_input(
                    "Nombre de pics simultanés",
                    min_value=1,
                    max_value=12,
                    value=1,
                    step=1,
                    key=f"npeaks_{i}",
                )

            masque_region = (
                np.isfinite(x)
                & np.isfinite(y_filtre)
                & (x >= xmin_fit)
                & (x <= xmax_fit)
            )

            if np.any(masque_region):
                y_region = y_filtre[masque_region]
                x_region = x[masque_region]
                offset_defaut = float(np.median(y_region))
                indice_extreme = int(
                    np.argmax(np.abs(y_region - offset_defaut))
                )
                centre_defaut = float(x_region[indice_extreme])
                amplitude_defaut = float(
                    y_region[indice_extreme] - offset_defaut
                )
            else:
                offset_defaut = float(np.median(y_filtre))
                centre_defaut = float(np.mean(x))
                amplitude_defaut = float(
                    np.max(y_filtre) - offset_defaut
                )

            with mc2:
                offset_initial = st.number_input(
                    "Offset initial",
                    value=offset_defaut,
                    format="%.8g",
                    key=f"offset0_{i}",
                )

            with mc3:
                pente_initiale = st.number_input(
                    "Pente initiale",
                    value=0.0,
                    format="%.8g",
                    key=f"slope0_{i}",
                )

            afficher_composantes = st.checkbox(
                "Afficher aussi les composantes individuelles "
                "(désactivé par défaut)",
                value=False,
                key=f"show_components_{i}",
            )

            largeur_region = max(
                float(xmax_fit - xmin_fit),
                np.finfo(float).eps,
            )

            pics_initiaux = []

            st.markdown("##### Valeurs initiales des pics")

            for j in range(int(n_pics)):
                pc1, pc2, pc3 = st.columns(3)

                centre_j_defaut = (
                    xmin_fit
                    + (j + 1)
                    * (xmax_fit - xmin_fit)
                    / (int(n_pics) + 1)
                )

                if int(n_pics) == 1:
                    centre_j_defaut = centre_defaut

                with pc1:
                    amplitude_j = st.number_input(
                        f"Amplitude initiale — pic {j + 1}",
                        value=amplitude_defaut,
                        format="%.8g",
                        key=f"amp0_{i}_{j}",
                    )

                with pc2:
                    centre_j = st.number_input(
                        f"Centre initial — pic {j + 1}",
                        value=float(centre_j_defaut),
                        format="%.8g",
                        key=f"center0_{i}_{j}",
                    )

                with pc3:
                    largeur_j = st.number_input(
                        (
                            f"FWHM initiale — pic {j + 1}"
                            if type_fit == "Somme de Lorentziennes"
                            else f"Sigma initial — pic {j + 1}"
                        ),
                        min_value=largeur_region / 100000,
                        value=max(
                            largeur_region / (5 * int(n_pics)),
                            largeur_region / 100000,
                        ),
                        format="%.8g",
                        key=f"width0_{i}_{j}",
                    )

                pics_initiaux.append({
                    "amplitude": amplitude_j,
                    "centre": centre_j,
                    "largeur": largeur_j,
                })

            try:
                fit_output = effectuer_fit_somme(
                    x=x,
                    y=y_filtre,
                    xmin=float(xmin_fit),
                    xmax=float(xmax_fit),
                    type_fit=type_fit,
                    pics_initiaux=pics_initiaux,
                    offset_initial=float(offset_initial),
                    pente_initiale=float(pente_initiale),
                    afficher_composantes=afficher_composantes,
                )

                st.success("Fit calculé.")

                st.dataframe(
                    fit_output.resultats,
                    use_container_width=True,
                )

            except Exception as exc:
                st.warning(
                    "Le fit n'a pas convergé avec ces paramètres : "
                    f"{exc}"
                )

        elif type_fit == "Polynôme":
            ordre_poly = st.number_input(
                "Ordre du polynôme",
                min_value=0,
                max_value=12,
                value=2,
                key=f"poly_order_{i}",
            )

            try:
                fit_output = effectuer_fit_polynomial(
                    x,
                    y_filtre,
                    float(xmin_fit),
                    float(xmax_fit),
                    int(ordre_poly),
                )

                st.success("Fit polynomial calculé.")

                st.dataframe(
                    fit_output.resultats,
                    use_container_width=True,
                )

            except Exception as exc:
                st.warning(str(exc))

        series.append({
            "nom": nom_serie,
            "x": x,
            "y_brut": y,
            "y_filtre": y_filtre,
            "filtre": nom_filtre,
            "couleur_donnees": couleur_donnees,
            "couleur_fit": couleur_fit,
            "epaisseur_donnees": epaisseur_donnees,
            "epaisseur_fit": epaisseur_fit,
            "fit": fit_output,
        })

    if not series:
        st.error("Aucune série exploitable.")
        st.stop()

    st.markdown("---")
    st.header("Graphique final")

    x_global = np.concatenate([s["x"] for s in series])
    y_global = np.concatenate([s["y_filtre"] for s in series])

    config_axes = controles_axes(
        "final",
        x_global,
        y_global,
    )

    fig, ax = plt.subplots(figsize=config_axes["figsize"])

    for serie in series:
        ax.plot(
            serie["x"],
            serie["y_filtre"],
            color=serie["couleur_donnees"],
            linewidth=serie["epaisseur_donnees"],
            label=f"{serie['nom']} — données filtrées",
        )

        if serie["fit"] is not None:
            ax.plot(
                serie["x"],
                serie["fit"].y_total,
                color=serie["couleur_fit"],
                linewidth=serie["epaisseur_fit"],
                linestyle="--",
                label=f"{serie['nom']} — fit total",
            )

            for k, composante in enumerate(
                serie["fit"].y_composantes
            ):
                ax.plot(
                    serie["x"],
                    composante["y"],
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.7,
                    label=(
                        f"{serie['nom']} — "
                        f"{composante['label']}"
                    ),
                )

    appliquer_axes(ax, config_axes)
    fig.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Les contrôles Streamlit sont réactifs : modifie les paramètres "
        "du fit, les limites, les couleurs ou les axes et le graphique "
        "est recalculé immédiatement."
    )

    st.markdown("---")
    st.header("Téléchargements")

    resultats_tous = []
    donnees_export = []

    for serie in series:
        df_serie = pd.DataFrame({
            "serie": serie["nom"],
            "x": serie["x"],
            "y_brut": serie["y_brut"],
            "y_filtre": serie["y_filtre"],
        })

        if serie["fit"] is not None:
            df_serie["fit_total"] = serie["fit"].y_total

            df_res = serie["fit"].resultats.copy()
            df_res.insert(0, "serie", serie["nom"])
            resultats_tous.append(df_res)

        donnees_export.append(df_serie)

    df_donnees = pd.concat(
        donnees_export,
        ignore_index=True,
    )

    if resultats_tous:
        df_resultats = pd.concat(
            resultats_tous,
            ignore_index=True,
            sort=False,
        )
    else:
        df_resultats = pd.DataFrame()

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            "⬇️ Données filtrées et fits",
            csv_bytes(df_donnees),
            "donnees_filtrees_et_fits.csv",
            "text/csv",
        )

    with d2:
        st.download_button(
            "⬇️ Tableau des paramètres",
            csv_bytes(df_resultats),
            "parametres_fits.csv",
            "text/csv",
            disabled=df_resultats.empty,
        )

    with d3:
        st.download_button(
            "⬇️ Graphique PNG",
            fig_png_bytes(fig),
            "graphique_fits.png",
            "image/png",
        )
