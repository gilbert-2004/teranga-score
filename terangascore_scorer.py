# -*- coding: utf-8 -*-
"""Moteur de scoring TerangaScore (partage par l'appli Streamlit et l'API)."""
import os
import numpy as np
import pandas as pd
import joblib

def load_artefact(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modele_terangascore.pkl")
    return joblib.load(path)

def load_artefact_survie(path=None):
    """Charge le modele de survie (Cox). Retourne None s'il n'existe pas."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modele_survie.pkl")
    return joblib.load(path) if os.path.exists(path) else None

def _ligne_covariables_survie(f, arts):
    """Construit la ligne de covariables standardisees pour un client."""
    NUM, moyenne, ecart, mediane = arts["NUM"], arts["moyenne"], arts["ecart"], arts["mediane"]
    lig = {}
    for c in NUM:
        val = f.get(c)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = mediane[c]
        lig[c] = (val - moyenne[c]) / ecart[c]
    hist = str(f.get("historique_remboursement"))
    lig["hist_Mauvais"] = int(hist == "Mauvais")
    lig["hist_Moyen"] = int(hist == "Moyen")
    lig["hist_Nouveau"] = int(hist == "Nouveau")
    lig["credit_solidaire"] = int(str(f.get("type_credit")) == "Groupe/solidaire")
    lig["secteur_agricole"] = int(str(f.get("secteur_activite")) in ("Agriculture", "Elevage"))
    return pd.DataFrame([[lig[c] for c in arts["COVARIABLES"]]], columns=arts["COVARIABLES"])

def courbe_survie_client(features, arts):
    """Predit la fonction de survie d'un client (probabilite de NE PAS avoir
    fait defaut, mois par mois). Retourne (mois, probabilites, survie a 12 mois)."""
    x = _ligne_covariables_survie(features, arts)
    sf = arts["cph"].predict_survival_function(x)
    mois = sf.index.values.astype(float)
    proba = sf.iloc[:, 0].values.astype(float)
    s12 = float(np.interp(12, mois, proba)) if len(mois) else float("nan")
    return mois, proba, s12

def _classe_num(val, bornes):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "Manquant"
    return str(pd.cut([val], bins=bornes, include_lowest=True)[0])

def calc_derivees(f):
    """Calcule les variables derivees a partir des saisies brutes."""
    montant = f["montant_credit_fcfa"]; taux = f["taux_interet_annuel_pct"]
    duree = f["duree_credit_mois"]; revenu = f["revenu_mensuel_fcfa"]
    epargne = f["epargne_moyenne_fcfa"]
    cout_total = montant * (1 + (taux/100) * duree/12)
    mensualite = cout_total / duree if duree else np.nan
    return {
        "mensualite_fcfa": mensualite,
        "taux_effort": (mensualite/revenu) if revenu else np.nan,
        "ratio_epargne_credit": (epargne/montant) if montant else np.nan,
    }

def scorer_client(raw, art):
    """Retourne score, proba de defaut, decision et le detail des points."""
    f = dict(raw)
    f.update(calc_derivees(f))
    if str(f.get("possede_mobile_money")) == "Non":
        for v in ["volume_mm_mensuel_fcfa","frequence_mm_mensuelle","anciennete_mobile_money_mois"]:
            f[v] = np.nan
    SEL, binning, beta, b0 = art["SEL"], art["binning"], art["beta"], art["b0"]
    F, OFF, n = art["FACTOR"], art["OFFSET"], len(art["SEL"])
    points, woe, classes = {}, {}, {}
    for v in SEL:
        typ, bornes, wmap, iv = binning[v]
        val = f.get(v, np.nan)
        if typ == "num":
            cls = _classe_num(val, bornes)
        else:
            cls = "Manquant" if val is None else str(val)
        w = wmap.get(cls, 0.0)
        woe[v] = w; classes[v] = cls
        points[v] = -F*beta[v]*w - F*b0/n + OFF/n
    logit = b0 + sum(beta[v]*woe[v] for v in SEL)
    proba = 1/(1+np.exp(-logit))
    score = OFF - F*logit
    decision = "Accorde" if score >= art["seuil"] else "Refuse"
    detail = (pd.DataFrame({"variable": list(points.keys()),
                            "classe": [classes[v] for v in points],
                            "points": [round(points[v],1) for v in points]})
              .sort_values("points"))
    return dict(score=round(float(score)), proba_defaut=float(proba),
                decision=decision, detail=detail, features=f)
