"""Application ODMR — interface Streamlit.

Deux outils :
  1. Traitement de sweeps : voltage(temps) -> voltage(fréquence), moyenne, SEM.
  2. Filtre & Fit : lissage puis ajustement lorentzien / gaussien / polynomial.

Lancement local :  streamlit run app.py
"""

import io
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from odmr.traitement import traiter_sweeps, detecter_pic_principal
from odmr.analyse import filtrer, fitter_region

st.set_page_config(page_title="Analyse ODMR", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🔬 Analyse ODMR")
page = st.sidebar.radio(
    "Outil",
    ["1 · Traitement de sweeps", "2 · Filtre & Fit"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Prototype pour l'équipe — les données ne quittent pas la session. "
    "Ajoutez vos futurs scripts comme nouvelles pages dans `app.py`."
)


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def charger_voltages(donnees_brutes: bytes) -> np.ndarray:
    """Charge un tableau 1D de voltages depuis des octets bruts.

    Gère : valeurs séparées par tabulations, espaces, retours à la ligne
    ou virgules ; une seule ligne géante (ex. fichiers Sweep0) ou une
    colonne classique ; avec ou sans extension de fichier.
    """
    texte = donnees_brutes.decode("utf-8", errors="ignore")
    if "," in texte and "\t" not in texte:
        texte = texte.replace(",", " ")
    valeurs = np.array(texte.split(), dtype=float)
    if valeurs.size == 0:
        raise ValueError("aucune valeur numérique trouvée dans le fichier.")
    return valeurs


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def fig_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


# ===========================================================================
# PAGE 1 — Traitement de sweeps
# ===========================================================================
if page.startswith("1"):
    st.title("Traitement de sweeps ODMR")
    st.markdown(
        "Chaque fichier contient des **voltages en fonction du temps** pendant un "
        "balayage de fréquence micro-onde. L'outil regroupe les échantillons par "
        "fréquence, les moyenne, puis calcule la **moyenne de tous les sweeps** "
        "avec son erreur standard (bruit réduit d'un facteur ~√N)."
    )

    fichiers = st.file_uploader(
        "Fichiers de données brutes (voltages en ligne ou en colonne, "
        "avec ou sans extension — ex. Sweep0, Sweep1...)",
        type=None,  # accepte les fichiers sans extension
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_freq = st.number_input("Nombre de fréquences testées", min_value=2,
                                 value=200, step=1)
    with col2:
        f_debut = st.number_input("Fréquence de départ (GHz)", value=2.5,
                                  format="%.4f")
    with col3:
        f_fin = st.number_input("Fréquence de fin (GHz)", value=3.5,
                                format="%.4f")

    if st.button("Lancer le traitement", type="primary", disabled=not fichiers):
        donnees, noms = [], []
        for f in fichiers:
            try:
                v = charger_voltages(f.getvalue())
                donnees.append(v)
                noms.append(f.name)
                st.caption(f"📄 {f.name} : {v.size:,} points lus".replace(",", " "))
            except Exception as e:
                st.warning(f"Lecture impossible de **{f.name}** : {e}")

        if not donnees:
            st.error("Aucun fichier lisible.")
            st.stop()

        try:
            res = traiter_sweeps(donnees, noms, int(n_freq), f_debut, f_fin)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        for msg in res.messages:
            st.info(msg)

        st.success(
            f"{res.n_sweeps} sweep(s) traité(s) — réduction du bruit attendue : "
            f"facteur ~√{res.n_sweeps} = {res.facteur_reduction_bruit:.2f}"
        )

        # --- Graphique 1 : tous les sweeps ---
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        couleurs = plt.cm.viridis(np.linspace(0, 1, res.n_sweeps))
        for courbe, nom, c in zip(res.sweeps, res.noms, couleurs):
            ax1.plot(res.axe_frequence, courbe, color=c, lw=0.8, alpha=0.8,
                     label=nom if res.n_sweeps <= 12 else None)
        ax1.set_title(f"Tous les sweeps ({res.n_sweeps} acquisitions)")
        ax1.set_xlabel("Fréquence (GHz)")
        ax1.set_ylabel("Amplitude (V)")
        if res.n_sweeps <= 12:
            ax1.legend(fontsize=7)
        fig1.tight_layout()

        # --- Graphique 2 : moyenne + SEM + pic ---
        pic = detecter_pic_principal(res.axe_frequence, res.moyenne)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.errorbar(
            res.axe_frequence, res.moyenne, yerr=res.erreur_moyenne,
            fmt="-", color="black", lw=1.2, ecolor="tab:red",
            elinewidth=0.6, capsize=1.5,
            errorevery=max(1, int(n_freq) // 50),
            label=f"Moyenne de {res.n_sweeps} sweeps (± σ/√N)",
        )
        if pic:
            freq_pic, amp_pic, _ = pic
            ax2.plot(freq_pic, amp_pic, "v", color="tab:blue", ms=10,
                     label=f"Pic : {freq_pic:.4f} GHz")
        ax2.set_title(
            f"Moyenne ODMR — bruit réduit d'un facteur "
            f"~{res.facteur_reduction_bruit:.1f}"
        )
        ax2.set_xlabel("Fréquence (GHz)")
        ax2.set_ylabel("Amplitude (V)")
        ax2.grid(True, ls="--", alpha=0.5)
        ax2.legend()
        fig2.tight_layout()

        cg1, cg2 = st.columns(2)
        with cg1:
            st.pyplot(fig1)
        with cg2:
            st.pyplot(fig2)

        if pic:
            st.metric("Pic principal", f"{pic[0]:.4f} GHz",
                      f"{pic[1]:.4f} V")

        # --- Téléchargements ---
        df_moy = pd.DataFrame({
            "Frequence_GHz": res.axe_frequence,
            "Amplitude_Moyenne": res.moyenne,
            "Erreur_Standard": res.erreur_moyenne,
        })

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("00_Moyenne_Globale.csv", df_moy.to_csv(index=False))
            for courbe, nom in zip(res.sweeps, res.noms):
                base = nom.rsplit(".", 1)[0]
                df = pd.DataFrame({
                    "Frequence_GHz": res.axe_frequence,
                    "Amplitude": courbe,
                })
                z.writestr(f"freq_{base}.csv", df.to_csv(index=False))
            z.writestr("01_tous_les_sweeps.png", fig_png_bytes(fig1))
            z.writestr("02_moyenne_bruit_reduit.png", fig_png_bytes(fig2))

        cd1, cd2 = st.columns(2)
        with cd1:
            st.download_button(
                "⬇️ Moyenne globale (CSV)", csv_bytes(df_moy),
                "00_Moyenne_Globale.csv", "text/csv",
            )
        with cd2:
            st.download_button(
                "⬇️ Tout (CSV + PNG) en ZIP", zip_buf.getvalue(),
                "traitement_odmr.zip", "application/zip",
            )


# ===========================================================================
# PAGE 2 — Filtre & Fit
# ===========================================================================
else:
    st.title("Filtre & Fit")
    st.markdown(
        "Charge un CSV **voltage(fréquence)** (par exemple la moyenne produite "
        "par l'outil 1), applique un filtre optionnel, puis ajuste une ou "
        "plusieurs courbes (lorentzienne, gaussienne, polynomiale)."
    )

    fichier = st.file_uploader("Fichier CSV", type=["csv", "txt"])
    if fichier is None:
        st.info("Charge un fichier pour commencer.")
        st.stop()

    df = pd.read_csv(fichier)
    colonnes = list(df.columns)

    c1, c2 = st.columns(2)
    with c1:
        col_x = st.selectbox("Colonne fréquence", colonnes, index=0)
    with c2:
        col_y = st.selectbox("Colonne amplitude", colonnes,
                             index=min(1, len(colonnes) - 1))

    x = df[col_x].to_numpy(dtype=float)
    y = df[col_y].to_numpy(dtype=float)

    # --- Filtre ---
    st.subheader("Filtre")
    methode = st.radio(
        "Méthode", ["Aucun", "Savitzky-Golay", "Moyenne mobile"],
        horizontal=True,
    )
    try:
        if methode == "Savitzky-Golay":
            cf1, cf2 = st.columns(2)
            with cf1:
                fenetre = st.slider("Fenêtre (impaire)", 3,
                                    max(5, len(y) // 2), 51, step=2)
            with cf2:
                ordre = st.slider("Ordre polynomial", 1, 5, 2)
            y_f, label_filtre = filtrer(y, "savgol", fenetre=fenetre, ordre=ordre)
        elif methode == "Moyenne mobile":
            fenetre = st.slider("Fenêtre", 2, max(4, len(y) // 2), 20)
            y_f, label_filtre = filtrer(y, "moyenne_mobile", fenetre=fenetre)
        else:
            y_f, label_filtre = filtrer(y, "aucun")
    except ValueError as e:
        st.error(str(e))
        st.stop()

    fig_f, ax_f = plt.subplots(figsize=(10, 4))
    ax_f.plot(x, y, alpha=0.35, label="Données brutes")
    if methode != "Aucun":
        ax_f.plot(x, y_f, label="Données filtrées")
    ax_f.set_xlabel("Fréquence")
    ax_f.set_ylabel("Amplitude")
    ax_f.set_title(label_filtre)
    ax_f.legend()
    fig_f.tight_layout()
    st.pyplot(fig_f)

    # --- Fits ---
    st.subheader("Ajustements")
    n_fits = st.number_input("Nombre de courbes à fitter", 0, 10, 1)

    configs = []
    for i in range(int(n_fits)):
        with st.expander(f"Fit {i + 1}", expanded=(i == 0)):
            ct, cr = st.columns([1, 2])
            with ct:
                type_fit = st.selectbox(
                    "Type", ["Lorentzienne", "Gaussienne", "Polynomiale"],
                    key=f"type_{i}",
                )
                creux = False
                ordre_poly = 2
                if type_fit == "Polynomiale":
                    ordre_poly = st.number_input("Ordre", 0, 10, 2,
                                                 key=f"ordre_{i}")
                else:
                    creux = st.checkbox("Le pic est un creux (dip ODMR)",
                                        key=f"creux_{i}")
            with cr:
                xmin, xmax = st.slider(
                    "Région à fitter",
                    float(x.min()), float(x.max()),
                    (float(x.min()), float(x.max())),
                    key=f"region_{i}",
                )
            configs.append({
                "type": type_fit.lower(), "xmin": xmin, "xmax": xmax,
                "ordre": ordre_poly, "creux": creux,
            })

    if int(n_fits) > 0 and st.button("Lancer les fits", type="primary"):
        resultats = []
        fit_total = np.zeros_like(y_f)
        courbes_fits = []
        erreurs = False

        for i, cfg in enumerate(configs):
            try:
                res, y_model = fitter_region(
                    x, y_f, cfg["type"], cfg["xmin"], cfg["xmax"],
                    ordre=cfg["ordre"], creux=cfg["creux"],
                )
            except Exception as e:
                st.error(f"Fit {i + 1} ({cfg['type']}) : {e}")
                erreurs = True
                continue
            resultats.append(res)
            fit_total += y_model
            courbes_fits.append((y_model, f"Fit {i + 1} : {res['type']}"))

        if resultats:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, y_f, label="Données filtrées", lw=1.5)
            for y_model, lab in courbes_fits:
                ax.plot(x, y_model, "--", lw=1, alpha=0.8, label=lab)
            if len(courbes_fits) > 1:
                ax.plot(x, fit_total, label="Fit total", lw=2)
            ax.set_xlabel("Fréquence")
            ax.set_ylabel("Amplitude")
            ax.set_title("Données filtrées avec fits")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)

            df_res = pd.DataFrame(resultats)
            st.dataframe(df_res, width="stretch")

            df_filtre = pd.DataFrame({
                "Frequence": x, "Amplitude_filtree": y_f,
            })
            cd1, cd2, cd3 = st.columns(3)
            with cd1:
                st.download_button("⬇️ Résultats des fits (CSV)",
                                   csv_bytes(df_res), "resultats_fits.csv",
                                   "text/csv")
            with cd2:
                st.download_button("⬇️ Données filtrées (CSV)",
                                   csv_bytes(df_filtre), "donnees_filtrees.csv",
                                   "text/csv")
            with cd3:
                st.download_button("⬇️ Graphique (PNG)",
                                   fig_png_bytes(fig, dpi=300),
                                   "graphique_fits.png", "image/png")
        elif erreurs:
            st.warning("Aucun fit n'a abouti — ajuste les régions ou le filtre.")
