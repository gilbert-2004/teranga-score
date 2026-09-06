# -*- coding: utf-8 -*-
"""
Exemple d'integration : comment le SI d'une caisse (SFD) appelle TerangaScore.
Prerequis : l'API tourne (uvicorn api:app --port 8000).
"""
import requests

BASE = "http://localhost:8000"
CLE = "teranga-dev-key"          # cle d'API (surchargeable cote serveur par TERANGASCORE_API_KEY)
ENTETES = {"X-API-Key": CLE}

# Profil transmis par le systeme d'information de la caisse
client = dict(
    pays="Senegal", zone="Urbaine", sexe="Femme", age=42,
    situation_matrimoniale="Mariee", personnes_a_charge=2,
    niveau_instruction="Secondaire", secteur_activite="Commerce",
    statut_activite="Formel", anciennete_activite_ans=8,
    revenu_mensuel_fcfa=350000, anciennete_membre_ans=6,
    nombre_credits_anterieurs=4, historique_remboursement="Bon",
    epargne_moyenne_fcfa=250000, type_credit="Individuel",
    possede_garant="Oui", objet_credit="Fonds de roulement",
    montant_credit_fcfa=400000, duree_credit_mois=12,
    taux_interet_annuel_pct=18.0, possede_mobile_money="Oui",
    volume_mm_mensuel_fcfa=300000, frequence_mm_mensuelle=20,
    anciennete_mobile_money_mois=36,
)

# Un seul appel : decision + trajectoire de risque
rep = requests.post(f"{BASE}/evaluation", json=client, headers=ENTETES, timeout=10).json()

s = rep["score"]
print(f"Score        : {s['score']}  (seuil d'octroi : {s['seuil']})")
print(f"Proba defaut : {s['proba_defaut']*100:.2f}%")
print(f"Decision     : {s['decision'].upper()}")
print("\nTrois principaux facteurs (points) :")
for ligne in sorted(s["explication"], key=lambda x: -x["points"])[:3]:
    print(f"   + {ligne['variable']:24} {ligne['classe']:20} {ligne['points']:>6} pts")

v = rep.get("survie")
if v:
    print(f"\nSurvie a 12 mois : {v['survie_12_mois']*100:.0f}%  "
          f"(risque de defaut : {v['risque_defaut_12_mois']*100:.0f}%)")
