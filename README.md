# TerangaScore

**Scoring de microcrédit interprétable et analyse de survie pour les coopératives financières de l'UEMOA.**

Prototype développé par l'équipe TERANGALAB dans le cadre du hackathon national CIF, projet DigiCoop-WA+ (Thématique 02).

TerangaScore répond à deux questions complémentaires et **entièrement interprétables** :

- une **scorecard** (probabilité de défaut) : *ce client va-t-il faire défaut ?*
- une **analyse de survie** (Kaplan-Meier + Cox) : *quand* le défaut risque-t-il de survenir ?

L'interprétabilité complète répond aux exigences prudentielles de la BCEAO (décision justifiable, opposable).

## Performances

| Moteur | Indicateur | Valeur |
|---|---|---|
| Scorecard | AUC | 0,847 |
| Scorecard | Gini | 0,694 |
| Scorecard | KS | 0,542 |
| Survie (Cox) | c-index | 0,876 |

## Structure du projet

| Fichier | Rôle |
|---|---|
| `generer_base_microcredit.py` | Génère la base synthétique UEMOA (5 000 clients, 32 variables) |
| `prototypage_scorecard.py` | Pipeline WOE / IV / scorecard interprétable (from scratch) |
| `entrainer_survie.py` | Entraîne et persiste le modèle de survie (Cox) |
| `analyse_survie.py` | Kaplan-Meier, log-rank, hazard ratios |
| `terangascore_scorer.py` | Moteur de scoring partagé (appli + API) |
| `app.py` | Application de démonstration (Streamlit) |
| `api.py` | API REST sécurisée (FastAPI) |
| `journal.py` | Journalisation des décisions (SQLite, mode hors ligne) |
| `exemple_client_api.py` | Exemple d'intégration côté SI d'une caisse |
| `documentation_technique_terangascore.pdf` | Dossier technique complet |

## Démarrage

Installer les dépendances :

```bash
pip install numpy pandas scikit-learn lifelines matplotlib streamlit fastapi uvicorn joblib pyreadr requests
```

Reconstruire les artefacts (base, scorecard, modèle de survie) :

```bash
python generer_base_microcredit.py
python prototypage_scorecard.py
python entrainer_survie.py
```

Lancer l'application de démonstration :

```bash
streamlit run app.py
```

Lancer l'API REST (documentation interactive sur http://localhost:8000/docs) :

```bash
uvicorn api:app --port 8000
```

## API REST

| Méthode et route | Rôle | Accès |
|---|---|---|
| `GET /health` | Disponibilité du service | libre |
| `GET /modele/info` | Qualité du modèle | libre |
| `POST /score` | Score, décision, explication | clé d'API |
| `POST /survie` | Courbe de survie individuelle | clé d'API |
| `POST /evaluation` | Score + survie en un appel | clé d'API |
| `GET /journal/statistiques` | Suivi du portefeuille | clé d'API |

Les points d'accès de décision exigent l'en-tête `X-API-Key`. La clé par défaut de développement est `teranga-dev-key`, à surcharger en production par la variable d'environnement `TERANGASCORE_API_KEY`.

## Note sur les données

La base est **synthétique**, calibrée sur la réalité de l'UEMOA (8 pays, structure de genre, ruralité, secteurs, plafond d'usure BCEAO à 24 %). Elle ne contient aucune donnée personnelle réelle.
