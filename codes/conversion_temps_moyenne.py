import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trouver_racine_repo(nom_repo="stage-odmr"):
    """
    Remonte l'arborescence des dossiers à partir du script actuel 
    pour trouver le chemin absolu du dépôt racine.
    """
    chemin_courant = os.path.abspath(__file__)
    while True:
        parent, dossier = os.path.split(chemin_courant)
        if dossier == nom_repo:
            return chemin_courant
        if not dossier:  # Arrivé à la racine du système de fichiers sans trouver
            break
        chemin_courant = parent
    # Fallback si exécuté en dehors du repo : prend le dossier du script
    return os.path.dirname(os.path.abspath(__file__))

def traitement_frequence_et_moyenne(dossier_entree, dossier_sortie, freq_debut, freq_fin):
    # Vérification et création du dossier de sortie
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
        print(f"\n[Info] Le dossier de sortie '{dossier_sortie}' a été créé.")
    else:
        print(f"\n[Info] Le dossier de sortie '{dossier_sortie}' existe déjà.")
    
    # Récupère absolument tous les fichiers dans le dossier source
    tous_fichiers = glob.glob(os.path.join(dossier_entree, "*"))
    fichiers_valides = []
    
    for f in tous_fichiers:
        if os.path.isdir(f):
            continue
        nom_base = os.path.basename(f)
        # On ignore les fichiers cachés système ou les fichiers déjà générés par un run précédent
        if nom_base.startswith('.') or nom_base.startswith('freq_') or nom_base.startswith('00_Moyenne'):
            continue
        fichiers_valides.append(f)
    
    if not fichiers_valides:
        print(f"\n[Erreur] Aucun fichier de données trouvé dans '{dossier_entree}'.")
        print("Vérifie que tes fichiers s'y trouvent bien.")
        return

    toutes_les_donnees = []
    
    print("\n--- Étape 1 : Transformation de l'axe X ---")
    for fichier in fichiers_valides:
        nom_base = os.path.basename(fichier)
        
        try:
            # np.loadtxt est robuste pour les fichiers 1D (avec ou sans extension)
            amplitudes = np.loadtxt(fichier).flatten()
        except Exception as e:
            print(f"Erreur lors de la lecture de {nom_base}: {e}")
            continue
            
        nb_points = len(amplitudes)
        
        # Création du vecteur de fréquence linéaire
        axe_frequence = np.linspace(freq_debut, freq_fin, nb_points)
        
        # Création du DataFrame
        df_nouveau = pd.DataFrame({
            'Frequence_GHz': axe_frequence,
            'Amplitude': amplitudes
        })
        
        # On force l'extension .csv en sortie pour la propreté du dossier d'analyse
        nom_sortie = f"freq_{nom_base}" if nom_base.endswith('.csv') else f"freq_{nom_base}.csv"
        chemin_sortie = os.path.join(dossier_sortie, nom_sortie)
        df_nouveau.to_csv(chemin_sortie, index=False)
        
        toutes_les_donnees.append(amplitudes)
        print(f"Traité : {nom_base} ({nb_points} points) -> {nom_sortie}")

    print("\n--- Étape 2 : Calcul de la moyenne globale et Affichage ---")
    if toutes_les_donnees:
        taille_min = min(len(d) for d in toutes_les_donnees)
        donnees_tronquees = np.array([d[:taille_min] for d in toutes_les_donnees])
        
        # Moyenne sur l'axe horizontal
        amplitudes_moyennes = np.mean(donnees_tronquees, axis=0)
        axe_frequence_moyen = np.linspace(freq_debut, freq_fin, taille_min)
        
        # Sauvegarde du fichier moyenné
        df_moyenne = pd.DataFrame({
            'Frequence_GHz': axe_frequence_moyen,
            'Amplitude_Moyenne': amplitudes_moyennes
        })
        
        chemin_moyenne = os.path.join(dossier_sortie, "00_Moyenne_Globale.csv")
        df_moyenne.to_csv(chemin_moyenne, index=False)
        print(f"Fichier de moyenne généré : {chemin_moyenne} (basé sur {len(toutes_les_donnees)} fichiers).")
        
        # Affichage du graphique final
        print("\nGénération du graphique... Fermez la fenêtre pour terminer le script.")
        plt.figure(figsize=(10, 6))
        plt.plot(axe_frequence_moyen, amplitudes_moyennes, label='Moyenne globale (bruit réduit)', color='black', linewidth=1.5)
        plt.title(f'Moyenne des acquisitions ODMR ({len(toutes_les_donnees)} fichiers)', fontsize=18)
        plt.xlabel('Fréquence (GHz)',  fontsize=18)
        plt.ylabel('Amplitude de sortie (V)', fontsize=18)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dossier_sortie, "00_Moyenne_Globale.png"), dpi=300)
        plt.show()

# ==========================================
# Bloc d'exécution interactif
# ==========================================
if __name__ == "__main__":
    print("=== Paramétrage du traitement de données ===")
    
    # 1. Résolution dynamique et robuste du dépôt stage-odmr
    racine_repo = trouver_racine_repo("stage-odmr")
    chemin_data_base = os.path.join(racine_repo, "data")
    
    # 2. Saisie des sous-dossiers (Le défaut est maintenant configuré sur "test_data")
    input_entree = input("Entrez le nom du sous-dossier source (dans data/) [défaut: test_data] : ").strip()
    nom_dossier_in = input_entree if input_entree else "test_data"
    DOSSIER_INPUT = os.path.join(chemin_data_base, nom_dossier_in)

    input_sortie = input("Entrez le nom du sous-dossier de sortie (dans data/) [défaut: test_data_traite] : ").strip()
    nom_dossier_out = input_sortie if input_sortie else "test_data_traite"
    DOSSIER_OUTPUT = os.path.join(chemin_data_base, nom_dossier_out)
    
    print(f"\n[Vérification des chemins absolus ciblés]")
    print(f"-> Source  : {DOSSIER_INPUT}")
    print(f"-> Sortie  : {DOSSIER_OUTPUT}\n")
    
    # 3. Paramètres numériques
    try:
        FREQ_DEBUT_GHZ = float(input("Fréquence de départ en GHz (ex: 2.5) : "))
        FREQ_FIN_GHZ = float(input("Fréquence de fin en GHz (ex: 3.5) : "))
        
        input_temps = input("Temps par point en secondes [défaut: 5.00] : ").strip()
        TEMPS_PAR_POINT = float(input_temps) if input_temps else 5.00
        
        print("\nLancement du traitement...")
        traitement_frequence_et_moyenne(
            dossier_entree=DOSSIER_INPUT,
            dossier_sortie=DOSSIER_OUTPUT,
            freq_debut=FREQ_DEBUT_GHZ,
            freq_fin=FREQ_FIN_GHZ
        )
        print("\nTraitement terminé avec succès !")
        
    except ValueError:
        print("\nErreur : Tu dois entrer des valeurs numériques valides. Relance le script.")
