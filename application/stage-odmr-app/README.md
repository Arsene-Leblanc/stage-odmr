# Analyse ODMR — application web

Application Streamlit regroupant les outils d'analyse ODMR du labo :

1. **Traitement de sweeps** — convertit les acquisitions voltage(temps) en
   voltage(fréquence), moyenne tous les sweeps (bruit réduit d'un facteur ~√N)
   et détecte le pic principal. Export CSV + PNG.
2. **Filtre & Fit** — lissage (Savitzky-Golay ou moyenne mobile) puis
   ajustement lorentzien, gaussien ou polynomial sur une ou plusieurs régions.
   Gère les pics *et* les creux (dips ODMR). Export des résultats.

## Utilisation locale (sans Docker)

```bash
git clone <url-du-repo>
cd stage-odmr-app
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse http://localhost:8501.

## Utilisation avec Docker

```bash
docker build -t odmr-app .
docker run -p 8501:8501 odmr-app
```

Puis ouvrir http://localhost:8501. Pour que toute l'équipe y accède depuis
le réseau du labo, lancer le conteneur sur une machine partagée : les
collègues s'y connectent via `http://<ip-de-la-machine>:8501`.

## Structure du projet

```
stage-odmr-app/
├── app.py               # interface Streamlit (les deux pages)
├── odmr/
│   ├── traitement.py    # logique pure : conversion temps→fréquence, moyenne
│   └── analyse.py       # logique pure : filtres, modèles, fits
├── requirements.txt
├── Dockerfile
└── README.md
```

## Ajouter un nouvel outil

1. Écrire la logique scientifique dans un nouveau module `odmr/mon_outil.py`
   sous forme de **fonctions pures** (entrées : tableaux/paramètres,
   sorties : résultats). Pas de `input()`, pas de `plt.show()`.
2. Ajouter une entrée dans le `st.sidebar.radio` de `app.py` et une section
   qui appelle ces fonctions avec des widgets Streamlit
   (`st.file_uploader`, `st.number_input`, `st.pyplot`, `st.download_button`).

## Notes

- Les données uploadées restent dans la session du navigateur ; rien n'est
  écrit sur le serveur.
- Les scripts console d'origine (`traitement_donnees.py`, `Filter_fit.py`)
  restent utilisables : la logique a simplement été extraite dans `odmr/`.
