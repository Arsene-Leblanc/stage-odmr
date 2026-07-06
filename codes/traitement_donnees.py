#!/usr/bin/env python3
"""
Traitement de sweeps ODMR
=========================

Chaque fichier du dossier source contient des voltages enregistrés en fonction
du TEMPS pendant un balayage (sweep) de la fréquence micro-onde.

Ce script :
  1. S'exécute avec '''python "name".py''' dans le terminal ;
  2. Demande depuis quel sous dossier dans /data il doit aller chercher les données brutes et dans quel sous dossier mettre les données traitées ;
  3. convertit chaque acquisition voltage(temps) -> voltage(fréquence) en
     regroupant les échantillons par fréquence puis en les moyennant. Le graphique n'est pas continu et c'est normal ;
  4. sauvegarde chaque sweep converti en CSV ;
  5. produit deux graphiques :
       - tous les sweeps superposés, chacun d'une couleur différente ;
       - la moyenne de tous les sweeps (bruit réduit d'un facteur ~sqrt(N)),
         avec une bande d'incertitude (erreur standard de la moyenne).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def trouver_racine_repo(nom_repo="stage-odmr"):
    """Remonte l'arborescence pour trouver le chemin absolu du dépôt racine."""
    chemin_courant = os.path.abspath(__file__)
    while True:
        parent, dossier = os.path.split(chemin_courant)
        if dossier == nom_repo:
            return chemin_courant
        if not dossier:  # racine du système atteinte sans trouver le repo
            break
        chemin_courant = parent
    return os.path.dirname(os.path.abspath(__file__))


def lister_fichiers_donnees(dossier_entree):
    """Retourne la liste triée des fichiers de données valides du dossier.

    On ignore les dossiers, les fichiers cachés, et les fichiers déjà générés
    par un run précédent (freq_*, 00_Moyenne*).
    """
    fichiers_valides = []
    for chemin in sorted(glob.glob(os.path.join(dossier_entree, "*"))):
        if os.path.isdir(chemin):
            continue
        nom = os.path.basename(chemin)
        if nom.startswith(".") or nom.startswith("freq_") or nom.startswith("00_Moyenne"):
            continue
        fichiers_valides.append(chemin)
    return fichiers_valides


def convertir_temps_vers_frequence(voltages, n_freq, f_debut, f_fin):
    """Convertit un tableau voltage(temps) 1D en voltage(fréquence).

    Le balayage teste `n_freq` fréquences réparties linéairement entre
    `f_debut` et `f_fin`. Les échantillons temporels sont donc regroupés en
    `n_freq` paquets consécutifs, et chaque paquet est moyenné pour donner
    un point de la courbe voltage(fréquence).

    Retourne (axe_frequence, voltage_par_frequence).
    """
    voltages = np.asarray(voltages, dtype=float).flatten()
    n_total = voltages.size

    if n_freq <= 0:
        raise ValueError("Le nombre de fréquences doit être strictement positif.")
    if n_total < n_freq:
        raise ValueError(
            f"Fichier trop court : {n_total} points pour {n_freq} fréquences demandées."
        )

    # Nombre d'échantillons par fréquence (durée du palier)
    ech_par_freq = n_total // n_freq
    n_utilisable = ech_par_freq * n_freq

    # Si le total n'est pas un multiple exact, on tronque les points en trop.
    reste = n_total - n_utilisable
    if reste:
        print(
            f"    [Info] {n_total} points non divisibles par {n_freq} : "
            f"{reste} point(s) en fin de fichier ignoré(s)."
        )

    # (n_freq, ech_par_freq) puis moyenne sur les échantillons de chaque palier
    bloc = voltages[:n_utilisable].reshape(n_freq, ech_par_freq)
    voltage_par_frequence = bloc.mean(axis=1)

    axe_frequence = np.linspace(f_debut, f_fin, n_freq)
    return axe_frequence, voltage_par_frequence


# ---------------------------------------------------------------------------
# Traitement principal
# ---------------------------------------------------------------------------
def traiter_dossier(dossier_entree, dossier_sortie, n_freq, f_debut, f_fin,
                    afficher=True):
    """Traite tout le dossier et génère les CSV + les deux graphiques."""

    os.makedirs(dossier_sortie, exist_ok=True)

    fichiers = lister_fichiers_donnees(dossier_entree)
    if not fichiers:
        print(f"\n[Erreur] Aucun fichier de données trouvé dans '{dossier_entree}'.")
        return

    print(f"\n{len(fichiers)} fichier(s) à traiter.\n")
    print("--- Conversion voltage(temps) -> voltage(fréquence) ---")

    axe_frequence = None
    sweeps = []        # liste des courbes voltage(fréquence)
    noms_sweeps = []   # noms associés (pour la légende / les CSV)

    for chemin in fichiers:
        nom = os.path.basename(chemin)
        try:
            voltages = np.loadtxt(chemin)
        except Exception as e:
            print(f"  [Ignoré] Lecture impossible de {nom} : {e}")
            continue

        try:
            freqs, volt_freq = convertir_temps_vers_frequence(
                voltages, n_freq, f_debut, f_fin
            )
        except ValueError as e:
            print(f"  [Ignoré] {nom} : {e}")
            continue

        if axe_frequence is None:
            axe_frequence = freqs

        sweeps.append(volt_freq)
        noms_sweeps.append(nom)

        # Sauvegarde du sweep converti
        nom_sortie = f"freq_{os.path.splitext(nom)[0]}.csv"
        pd.DataFrame(
            {"Frequence_GHz": freqs, "Amplitude": volt_freq}
        ).to_csv(os.path.join(dossier_sortie, nom_sortie), index=False)
        print(f"  Traité : {nom}  ->  {nom_sortie}")

    if not sweeps:
        print("\n[Erreur] Aucun sweep exploitable. Arrêt.")
        return

    sweeps = np.array(sweeps)          # forme (n_fichiers, n_freq)
    n_sweeps = sweeps.shape[0]

    # --- Moyenne et réduction du bruit ---
    moyenne = sweeps.mean(axis=0)
    ecart_type = sweeps.std(axis=0, ddof=1) if n_sweeps > 1 else np.zeros(n_freq)
    erreur_moyenne = ecart_type / np.sqrt(n_sweeps)   # SEM ~ 1/sqrt(N)

    # Sauvegarde de la moyenne
    df_moyenne = pd.DataFrame({
        "Frequence_GHz": axe_frequence,
        "Amplitude_Moyenne": moyenne,
        "Erreur_Standard": erreur_moyenne,
    })
    chemin_moyenne = os.path.join(dossier_sortie, "00_Moyenne_Globale.csv")
    df_moyenne.to_csv(chemin_moyenne, index=False)
    print(f"\nMoyenne sauvegardée : {chemin_moyenne}")
    print(f"Réduction du bruit attendue : facteur ~sqrt({n_sweeps}) "
          f"= {np.sqrt(n_sweeps):.2f}")

    # -----------------------------------------------------------------------
    # Graphique 1 : tous les sweeps, une couleur par sweep
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    couleurs = plt.cm.viridis(np.linspace(0, 1, n_sweeps))
    for volt_freq, nom, couleur in zip(sweeps, noms_sweeps, couleurs):
        # Légende affichée seulement s'il n'y a pas trop de sweeps
        label = nom if n_sweeps <= 12 else None
        ax1.plot(axe_frequence, volt_freq, color=couleur, linewidth=0.8,
                 alpha=0.8, label=label)
    ax1.set_title(f"Graphique de tous les sweeps ({n_sweeps} acquisitions)")
    ax1.set_xlabel("Fréquence (GHz)")
    ax1.set_ylabel("Amplitude (V)")
    if n_sweeps <= 12:
        ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig(os.path.join(dossier_sortie, "01_tous_les_sweeps.png"), dpi=150)

    # -----------------------------------------------------------------------
    # Graphique 2 : moyenne (bruit réduit) + barres d'erreur + pic principal
    # -----------------------------------------------------------------------
    from scipy.signal import find_peaks

    # Détection de tous les pics, puis sélection du plus important (plus haut)
    indices_pics, _ = find_peaks(moyenne)
    if indices_pics.size:
        idx_pic = indices_pics[np.argmax(moyenne[indices_pics])]
        freq_pic = axe_frequence[idx_pic]
        amp_pic = moyenne[idx_pic]
        print(f"\nPic principal détecté : {amp_pic:.4f} V à {freq_pic:.4f} GHz")
    else:
        idx_pic = None
        print("\n[Info] Aucun pic détecté.")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(axe_frequence, moyenne, yerr=erreur_moyenne,
                 fmt="-", color="black", linewidth=1.2,
                 ecolor="tab:red", elinewidth=0.6, capsize=1.5,
                 errorevery=max(1, n_freq // 50),
                 label=f"Moyenne de {n_sweeps} sweeps  (± σ/√N)")

    # Marque le pic principal
    if idx_pic is not None:
        ax2.plot(freq_pic, amp_pic, "v", color="tab:blue", markersize=10,
                 label=f"Pic : {freq_pic:.3f} GHz")
        ax2.annotate(f"{freq_pic:.3f} GHz",
                     xy=(freq_pic, amp_pic),
                     xytext=(freq_pic, amp_pic + 0.05 * (moyenne.max() - moyenne.min())),
                     ha="center", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="tab:blue"))


    ax2.set_title(f"Moyenne ODMR — bruit réduit d'un facteur ~{np.sqrt(n_sweeps):.1f}")
    ax2.set_xlabel("Fréquence (GHz)")
    ax2.set_ylabel("Amplitude (V)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(dossier_sortie, "02_moyenne_bruit_reduit.png"), dpi=150)   
    print("\nGraphiques sauvegardés dans le dossier de sortie.")
    if afficher:
        plt.show()


# ---------------------------------------------------------------------------
# Bloc interactif
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Traitement de sweeps ODMR ===")

    racine_repo = trouver_racine_repo("stage-odmr")
    chemin_data_base = os.path.join(racine_repo, "data")

    entree = input("Sous-dossier source (dans data/) [test_data] : ").strip() or "test_data"
    sortie = input("Sous-dossier de sortie (dans data/) [test_data_traite] : ").strip() or "test_data_traite"
    dossier_input = os.path.join(chemin_data_base, entree)
    dossier_output = os.path.join(chemin_data_base, sortie)

    print(f"\n-> Source : {dossier_input}")
    print(f"-> Sortie : {dossier_output}\n")

    try:
        n_freq = int(input("Nombre de fréquences testées (points du sweep) : "))
        f_debut = float(input("Fréquence de départ en GHz (ex: 2.5) : "))
        f_fin = float(input("Fréquence de fin en GHz (ex: 3.5) : "))
    except ValueError:
        print("\n[Erreur] Entre des valeurs numériques valides. Relance le script.")
        raise SystemExit(1)

    print("\nLancement du traitement...")
    traiter_dossier(dossier_input, dossier_output, n_freq, f_debut, f_fin)
    print("\nTraitement terminé.")
