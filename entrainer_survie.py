# -*- coding: utf-8 -*-
"""
TerangaScore, entrainement et sauvegarde du modele de survie (Cox)
==================================================================
Ajuste un modele de Cox (hazards proportionnels) sur la base UEMOA, puis
PERSISTE le modele et les parametres de standardisation dans un artefact
(modele_survie.pkl) pour predire la courbe de survie d'un client dans l'appli.
"""
import os
import numpy as np
import pandas as pd
import joblib
from lifelines import CoxPHFitter

DOSSIER = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DOSSIER, "base_microcredit_uemoa.csv"))
T, E = "duree_observation_mois", "evenement_defaut"

# Variables numeriques standardisees + indicatrices (memes que l'analyse)
NUM = ["taux_effort", "revenu_mensuel_fcfa", "anciennete_membre_ans",
       "epargne_moyenne_fcfa", "montant_credit_fcfa", "age",
       "nombre_credits_anterieurs"]

d = df[[T, E] + NUM + ["historique_remboursement", "type_credit", "secteur_activite"]].copy()

# Parametres de standardisation (a reutiliser pour un nouveau client)
mediane = {c: float(d[c].median()) for c in NUM}
moyenne, ecart = {}, {}
for c in NUM:
    d[c] = d[c].fillna(mediane[c])
    moyenne[c] = float(d[c].mean())
    ecart[c] = float(d[c].std())
    d[c] = (d[c] - moyenne[c]) / ecart[c]

# Indicatrices
d["hist_Mauvais"] = (d["historique_remboursement"] == "Mauvais").astype(int)
d["hist_Moyen"] = (d["historique_remboursement"] == "Moyen").astype(int)
d["hist_Nouveau"] = (d["historique_remboursement"] == "Nouveau").astype(int)
d["credit_solidaire"] = (d["type_credit"] == "Groupe/solidaire").astype(int)
d["secteur_agricole"] = d["secteur_activite"].isin(["Agriculture", "Elevage"]).astype(int)
d = d.drop(columns=["historique_remboursement", "type_credit", "secteur_activite"])

cph = CoxPHFitter(penalizer=0.05)
cph.fit(d, duration_col=T, event_col=E)

# Ordre des covariables attendu par le modele (hors T/E)
COVARIABLES = [c for c in d.columns if c not in (T, E)]

artefact_survie = dict(
    cph=cph, NUM=NUM, moyenne=moyenne, ecart=ecart, mediane=mediane,
    COVARIABLES=COVARIABLES, c_index=float(cph.concordance_index_),
    survie_mediane_portefeuille=float(df.loc[df[E] == 1, T].median()),
)
joblib.dump(artefact_survie, os.path.join(DOSSIER, "modele_survie.pkl"))

print("=" * 60)
print("Modele de survie entraine et sauvegarde")
print("=" * 60)
print(f"c-index = {cph.concordance_index_:.3f}")
print(f"Covariables : {COVARIABLES}")
print("Artefact : modele_survie.pkl")
