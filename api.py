# -*- coding: utf-8 -*-
"""
TerangaScore, API REST (interoperabilite + securite + mode hors ligne)
======================================================================
Expose le moteur de scoring et l'analyse de survie via HTTP/JSON, pour que le
systeme d'information d'une caisse (SFD) puisse obtenir un score sans connaitre
les details du modele. Reutilise exactement le meme moteur que l'appli Streamlit.

Securite     : les points d'acces de decision exigent une cle d'API (entete X-API-Key).
Mode hors ligne : chaque decision est journalisee localement (SQLite), sans reseau.

Lancer :   uvicorn api:app --port 8000
Docs      : http://localhost:8000/docs   (Swagger interactif)
Cle par defaut (dev) : teranga-dev-key   (surchargeable par la variable TERANGASCORE_API_KEY)
"""
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from terangascore_scorer import (load_artefact, load_artefact_survie,
                                 scorer_client, courbe_survie_client)
import journal

# Chargement unique des artefacts au demarrage
ART = load_artefact()
ART_SURVIE = load_artefact_survie()
journal.init_db()

# Cle d'API : variable d'environnement, sinon cle de developpement
API_KEY = os.environ.get("TERANGASCORE_API_KEY", "teranga-dev-key")
_entete_cle = APIKeyHeader(name="X-API-Key", auto_error=False)


def verifier_cle(cle: str = Security(_entete_cle)):
    """Dependance de securite : refuse l'appel si la cle est absente ou fausse."""
    if cle != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Cle d'API absente ou invalide (entete X-API-Key).")
    return True


app = FastAPI(
    title="TerangaScore API",
    version="1.1.0",
    description="Scoring de microcredit interpretable et analyse de survie pour les "
                "cooperatives financieres de l'UEMOA. Equipe TERANGALAB, projet DigiCoop-WA+. "
                "Les points d'acces de decision exigent l'entete X-API-Key.",
)

# ----------------------------------------------------------------------
# Schemas d'entree / sortie
# ----------------------------------------------------------------------
class Client(BaseModel):
    """Profil brut d'un demandeur de credit."""
    pays: str = Field(..., examples=["Senegal"])
    zone: str = Field(..., examples=["Urbaine"])
    sexe: str = Field(..., examples=["Femme"])
    age: float = Field(..., ge=18, le=100, examples=[42])
    situation_matrimoniale: str = Field(..., examples=["Mariee"])
    personnes_a_charge: float = Field(..., ge=0, examples=[2])
    niveau_instruction: str = Field(..., examples=["Secondaire"])
    secteur_activite: str = Field(..., examples=["Commerce"])
    statut_activite: str = Field(..., examples=["Formel"])
    anciennete_activite_ans: float = Field(..., ge=0, examples=[8])
    revenu_mensuel_fcfa: float = Field(..., gt=0, examples=[350000])
    anciennete_membre_ans: float = Field(..., ge=0, examples=[6])
    nombre_credits_anterieurs: float = Field(..., ge=0, examples=[4])
    historique_remboursement: str = Field(..., examples=["Bon"])
    epargne_moyenne_fcfa: float = Field(..., ge=0, examples=[250000])
    type_credit: str = Field(..., examples=["Individuel"])
    possede_garant: str = Field(..., examples=["Oui"])
    objet_credit: str = Field(..., examples=["Fonds de roulement"])
    montant_credit_fcfa: float = Field(..., gt=0, examples=[400000])
    duree_credit_mois: float = Field(..., gt=0, examples=[12])
    taux_interet_annuel_pct: float = Field(..., ge=0, le=24, examples=[18.0])
    possede_mobile_money: str = Field(..., examples=["Oui"])
    volume_mm_mensuel_fcfa: Optional[float] = Field(None, examples=[300000])
    frequence_mm_mensuelle: Optional[float] = Field(None, examples=[20])
    anciennete_mobile_money_mois: Optional[float] = Field(None, examples=[36])


class LigneExplication(BaseModel):
    variable: str
    classe: str
    points: float


class ReponseScore(BaseModel):
    score: int
    proba_defaut: float
    decision: str
    seuil: int
    explication: List[LigneExplication]


class PointSurvie(BaseModel):
    mois: float
    survie: float


class ReponseSurvie(BaseModel):
    survie_12_mois: float
    risque_defaut_12_mois: float
    c_index: float
    courbe: List[PointSurvie]


class ReponseComplete(BaseModel):
    score: ReponseScore
    survie: Optional[ReponseSurvie]


# ----------------------------------------------------------------------
# Fonctions internes
# ----------------------------------------------------------------------
def _bloc_score(raw: dict) -> ReponseScore:
    r = scorer_client(raw, ART)
    explication = [
        LigneExplication(variable=row["variable"], classe=row["classe"], points=row["points"])
        for _, row in r["detail"].iterrows()
    ]
    return ReponseScore(
        score=r["score"], proba_defaut=round(r["proba_defaut"], 4),
        decision=r["decision"], seuil=int(ART["seuil"]), explication=explication,
    )


def _bloc_survie(raw: dict) -> Optional[ReponseSurvie]:
    if ART_SURVIE is None:
        return None
    r = scorer_client(raw, ART)
    mois, proba, s12 = courbe_survie_client(r["features"], ART_SURVIE)
    courbe = [PointSurvie(mois=float(m), survie=round(float(p), 4)) for m, p in zip(mois, proba)]
    return ReponseSurvie(
        survie_12_mois=round(s12, 4), risque_defaut_12_mois=round(1 - s12, 4),
        c_index=round(ART_SURVIE["c_index"], 3), courbe=courbe,
    )


# ----------------------------------------------------------------------
# Points d'acces publics (systeme)
# ----------------------------------------------------------------------
@app.get("/health", tags=["Systeme"])
def health():
    """Verifie que le service repond (acces libre)."""
    return {"statut": "ok", "service": "TerangaScore API", "version": app.version}


@app.get("/modele/info", tags=["Systeme"])
def modele_info():
    """Qualite et parametres du modele en production (acces libre)."""
    return {
        "auc": round(ART["auc"], 3), "gini": round(ART["gini"], 3), "ks": round(ART["ks"], 3),
        "seuil_octroi": int(ART["seuil"]),
        "variables_retenues": len(ART["SEL"]),
        "survie_disponible": ART_SURVIE is not None,
        "c_index_survie": round(ART_SURVIE["c_index"], 3) if ART_SURVIE else None,
    }


# ----------------------------------------------------------------------
# Points d'acces proteges (decision) : exigent la cle d'API
# ----------------------------------------------------------------------
@app.post("/score", response_model=ReponseScore, tags=["Decision"])
def score(client: Client, _=Security(verifier_cle)):
    """Score, probabilite de defaut, decision et explication ligne par ligne."""
    try:
        rep = _bloc_score(client.model_dump())
        journal.enregistrer(client.model_dump(), rep.score, rep.proba_defaut, rep.decision)
        return rep
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur de scoring : {e}")


@app.post("/survie", response_model=ReponseSurvie, tags=["Decision"])
def survie(client: Client, _=Security(verifier_cle)):
    """Courbe de survie individuelle (probabilite de non-defaut mois par mois)."""
    bloc = _bloc_survie(client.model_dump())
    if bloc is None:
        raise HTTPException(status_code=503, detail="Modele de survie indisponible (lancer entrainer_survie.py).")
    return bloc


@app.post("/evaluation", response_model=ReponseComplete, tags=["Decision"])
def evaluation(client: Client, _=Security(verifier_cle)):
    """Evaluation complete : score + decision + courbe de survie en un seul appel."""
    raw = client.model_dump()
    try:
        bloc_score = _bloc_score(raw)
        bloc_survie = _bloc_survie(raw)
        s12 = bloc_survie.survie_12_mois if bloc_survie else None
        journal.enregistrer(raw, bloc_score.score, bloc_score.proba_defaut, bloc_score.decision, s12)
        return ReponseComplete(score=bloc_score, survie=bloc_survie)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur d'evaluation : {e}")


@app.get("/journal/statistiques", tags=["Journal"])
def journal_statistiques(_=Security(verifier_cle)):
    """Statistiques du portefeuille journalise localement (mode hors ligne)."""
    return journal.statistiques()


@app.get("/journal/dernieres", tags=["Journal"])
def journal_dernieres(limite: int = 20, _=Security(verifier_cle)):
    """Dernieres decisions enregistrees (piste d'audit)."""
    return journal.dernieres(limite)
