"""Logique pure du traitement de sweeps ODMR.

Aucune interaction console, aucun matplotlib : uniquement des fonctions
qui prennent des tableaux et retournent des résultats. C'est ce qui rend
le module réutilisable par l'app Streamlit, un notebook, ou un futur CLI.
"""

from dataclasses import dataclass, field

import numpy as np


def convertir_temps_vers_frequence(voltages, n_freq, f_debut, f_fin):
    """Convertit un tableau voltage(temps) 1D en voltage(fréquence).

    Le balayage teste `n_freq` fréquences réparties linéairement entre
    `f_debut` et `f_fin`. Les échantillons temporels sont regroupés en
    `n_freq` paquets consécutifs, chacun moyenné pour donner un point.

    Retourne (axe_frequence, voltage_par_frequence, n_points_ignores).
    """
    voltages = np.asarray(voltages, dtype=float).flatten()
    n_total = voltages.size

    if n_freq <= 0:
        raise ValueError("Le nombre de fréquences doit être strictement positif.")
    if n_total < n_freq:
        raise ValueError(
            f"Fichier trop court : {n_total} points pour {n_freq} fréquences demandées."
        )

    ech_par_freq = n_total // n_freq
    n_utilisable = ech_par_freq * n_freq
    reste = n_total - n_utilisable

    bloc = voltages[:n_utilisable].reshape(n_freq, ech_par_freq)
    voltage_par_frequence = bloc.mean(axis=1)
    axe_frequence = np.linspace(f_debut, f_fin, n_freq)

    return axe_frequence, voltage_par_frequence, reste


@dataclass
class ResultatSweeps:
    """Résultat du traitement d'un lot de sweeps."""
    axe_frequence: np.ndarray
    sweeps: np.ndarray                 # forme (n_sweeps, n_freq)
    noms: list = field(default_factory=list)
    moyenne: np.ndarray = None
    erreur_moyenne: np.ndarray = None  # SEM = sigma / sqrt(N)
    messages: list = field(default_factory=list)

    @property
    def n_sweeps(self):
        return self.sweeps.shape[0]

    @property
    def facteur_reduction_bruit(self):
        return float(np.sqrt(self.n_sweeps))


def traiter_sweeps(donnees_brutes, noms, n_freq, f_debut, f_fin):
    """Traite une liste de tableaux voltage(temps).

    `donnees_brutes` : liste de tableaux 1D (un par fichier).
    `noms`           : noms de fichiers associés (même longueur).

    Retourne un ResultatSweeps. Les fichiers illisibles ou trop courts
    sont ignorés avec un message explicatif dans `messages`.
    """
    axe = None
    sweeps, noms_ok, messages = [], [], []

    for voltages, nom in zip(donnees_brutes, noms):
        try:
            freqs, volt_freq, reste = convertir_temps_vers_frequence(
                voltages, n_freq, f_debut, f_fin
            )
        except ValueError as e:
            messages.append(f"[Ignoré] {nom} : {e}")
            continue

        if reste:
            messages.append(
                f"[Info] {nom} : {reste} point(s) en fin de fichier ignoré(s) "
                f"(total non multiple de {n_freq})."
            )

        if axe is None:
            axe = freqs
        sweeps.append(volt_freq)
        noms_ok.append(nom)

    if not sweeps:
        raise ValueError("Aucun sweep exploitable dans les fichiers fournis.")

    sweeps = np.array(sweeps)
    n = sweeps.shape[0]
    moyenne = sweeps.mean(axis=0)
    ecart_type = sweeps.std(axis=0, ddof=1) if n > 1 else np.zeros(n_freq)
    sem = ecart_type / np.sqrt(n)

    return ResultatSweeps(
        axe_frequence=axe,
        sweeps=sweeps,
        noms=noms_ok,
        moyenne=moyenne,
        erreur_moyenne=sem,
        messages=messages,
    )


def detecter_pic_principal(axe_frequence, courbe):
    """Retourne (freq_pic, amp_pic, index) du pic le plus haut, ou None."""
    from scipy.signal import find_peaks

    indices, _ = find_peaks(courbe)
    if indices.size == 0:
        return None
    idx = indices[np.argmax(courbe[indices])]
    return float(axe_frequence[idx]), float(courbe[idx]), int(idx)
