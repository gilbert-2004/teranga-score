# -*- coding: utf-8 -*-
"""
TerangaScore, generateur de base synthetique de microcredit (contexte UEMOA)
============================================================================
Genere une base realiste de demandes de microcredit dans les cooperatives
financieres (SFD) de l'UEMOA, avec :
  - variables socio-demographiques, activite economique, relation cooperative,
    caracteristiques du credit, donnees alternatives (mobile money) ;
  - une cible 'defaut' construite a partir d'un modele de risque latent
    (donc APPRENABLE : un modele de scoring peut y retrouver du signal) ;
  - un parametrage explicite calibre sur la realite UEMOA (modifiable en tete).

Sorties (dans le meme dossier) :
  - base_microcredit_uemoa.csv     (la base principale)
  - dictionnaire_variables.csv     (dictionnaire des variables)
  - resume_base.txt                (statistiques de controle)
"""
import numpy as np
import pandas as pd
import os

# ============================================================
# 0. PARAMETRAGE (calibre sur la realite de l'UEMOA)
# ============================================================
SEED = 42
N = 5000                     # nombre de clients
TAUX_DEFAUT_CIBLE = 0.11     # taux de defaut vise (~ portefeuille a risque SFD)

rng = np.random.default_rng(SEED)
DOSSIER = os.path.dirname(os.path.abspath(__file__))

# Repartitions (parts), calibrees sur le profil des SFD de l'UEMOA
PAYS = {"Senegal":0.20,"Cote d'Ivoire":0.20,"Mali":0.15,"Burkina Faso":0.15,
        "Benin":0.10,"Togo":0.10,"Niger":0.07,"Guinee-Bissau":0.03}
SEXE = {"Femme":0.62,"Homme":0.38}                     # forte part de femmes en microfinance
ZONE = {"Rural":0.52,"Urbain":0.48}
INSTRUCTION = {"Aucun":0.35,"Primaire":0.33,"Secondaire":0.22,"Superieur":0.10}
MATRIMONIAL = {"Marie(e)":0.66,"Celibataire":0.20,"Veuf(ve)":0.09,"Divorce(e)":0.05}
SECTEUR = {"Commerce":0.38,"Agriculture":0.22,"Elevage":0.08,"Artisanat":0.12,
           "Transport":0.05,"Services":0.09,"Transformation agroalimentaire":0.06}
TYPE_CREDIT = {"Groupe/solidaire":0.55,"Individuel":0.45}
OBJET = {"Fonds de roulement":0.45,"Intrants agricoles":0.15,"Stock":0.15,
         "Equipement":0.12,"Investissement":0.13}

# Effets multiplicatifs sur le revenu
MULT_SECTEUR = {"Commerce":1.10,"Agriculture":0.75,"Elevage":0.85,"Artisanat":0.90,
                "Transport":1.15,"Services":1.25,"Transformation agroalimentaire":1.00}
MULT_INSTR = {"Aucun":0.85,"Primaire":0.95,"Secondaire":1.10,"Superieur":1.35}

def tirage(d, n):
    return rng.choice(list(d.keys()), size=n, p=np.array(list(d.values())))

# ============================================================
# 1. SOCIO-DEMOGRAPHIE
# ============================================================
pays = tirage(PAYS, N)
sexe = tirage(SEXE, N)
zone = tirage(ZONE, N)
instruction = tirage(INSTRUCTION, N)
matrimonial = tirage(MATRIMONIAL, N)
age = np.clip(rng.normal(38, 11, N), 18, 70).round().astype(int)
personnes_charge = np.clip(rng.poisson(3, N), 0, 12)

# ============================================================
# 2. ACTIVITE ECONOMIQUE ET REVENU
# ============================================================
secteur = tirage(SECTEUR, N)
anciennete_activite = np.clip(rng.exponential(6, N), 0, 40).round(1)
statut_activite = np.where(rng.random(N) < 0.82, "Informel", "Formel")  # informel majoritaire

mult_sec = np.array([MULT_SECTEUR[s] for s in secteur])
mult_ins = np.array([MULT_INSTR[i] for i in instruction])
bruit_rev = rng.lognormal(0, 0.45, N)
revenu_mensuel = (110000 * mult_sec * mult_ins
                  * (1 + 0.03*anciennete_activite) * bruit_rev)
revenu_mensuel = np.clip(revenu_mensuel, 30000, 1500000).round(-2).astype(int)

# ============================================================
# 3. RELATION AVEC LA COOPERATIVE (SFD)
# ============================================================
anciennete_membre = np.clip(rng.exponential(4, N), 0, 25).round(1)
# nombre de credits anterieurs, croissant avec l'anciennete de membre
nombre_credits_ant = np.clip(rng.poisson(0.5 + 0.6*anciennete_membre), 0, 15)
# epargne moyenne, liee au revenu et a l'anciennete de membre
epargne_moyenne = (revenu_mensuel * (0.15 + 0.06*anciennete_membre)
                   * rng.lognormal(0, 0.5, N))
epargne_moyenne = np.clip(epargne_moyenne, 0, 3000000).round(-2).astype(int)

# historique de remboursement (uniquement si credits anterieurs)
hist = np.array(["Nouveau"]*N, dtype=object)
a_deja = nombre_credits_ant > 0
u = rng.random(N)
hist[a_deja & (u < 0.70)] = "Bon"
hist[a_deja & (u >= 0.70) & (u < 0.92)] = "Moyen"
hist[a_deja & (u >= 0.92)] = "Mauvais"

type_credit = tirage(TYPE_CREDIT, N)
a_garant = np.where(rng.random(N) < 0.6, "Oui", "Non")

# ============================================================
# 4. CARACTERISTIQUES DU CREDIT DEMANDE
# ============================================================
objet = tirage(OBJET, N)
# montant lie au revenu (2 a 6 mois de revenu), borne reglementaire realiste
montant_credit = (revenu_mensuel * rng.uniform(1.5, 6.0, N) * rng.lognormal(0,0.3,N))
montant_credit = np.clip(montant_credit, 50000, 2000000).round(-3).astype(int)
duree_credit = rng.choice([3,6,9,12,18,24], size=N, p=[0.10,0.25,0.15,0.30,0.12,0.08])
# taux effectif annuel, sous le plafond d'usure BCEAO pour les SFD (24%)
taux_interet = np.clip(rng.normal(21, 2.0, N), 12, 24).round(1)

# echeance mensuelle (interet simple annualise) et taux d'effort
cout_total = montant_credit * (1 + (taux_interet/100) * duree_credit/12)
mensualite = (cout_total / duree_credit).round(-2).astype(int)
taux_effort = (mensualite / revenu_mensuel).round(3)              # part du revenu
ratio_epargne_credit = (epargne_moyenne / montant_credit).round(3)

# ============================================================
# 5. DONNEES ALTERNATIVES (MOBILE MONEY)  -> angle d'innovation
# ============================================================
p_mm = np.clip(0.50 + 0.20*(zone=="Urbain") - 0.004*(age-38)
               + 0.10*(instruction=="Superieur"), 0.1, 0.95)
possede_mm = rng.random(N) < p_mm
volume_mm = np.where(possede_mm,
                     np.clip(revenu_mensuel*rng.uniform(0.2,1.5,N)*rng.lognormal(0,0.5,N),
                             0, 5000000), np.nan)
freq_mm = np.where(possede_mm, np.clip(rng.poisson(12, N), 0, 120), np.nan)
anciennete_mm = np.where(possede_mm, np.clip(rng.exponential(18, N),0,120).round(0), np.nan)

df = pd.DataFrame({
    "id_client": [f"TS{100000+i}" for i in range(N)],
    "pays": pays, "zone": zone, "sexe": sexe, "age": age,
    "situation_matrimoniale": matrimonial, "personnes_a_charge": personnes_charge,
    "niveau_instruction": instruction, "secteur_activite": secteur,
    "statut_activite": statut_activite, "anciennete_activite_ans": anciennete_activite,
    "revenu_mensuel_fcfa": revenu_mensuel,
    "anciennete_membre_ans": anciennete_membre,
    "nombre_credits_anterieurs": nombre_credits_ant,
    "historique_remboursement": hist,
    "epargne_moyenne_fcfa": epargne_moyenne,
    "type_credit": type_credit, "possede_garant": a_garant,
    "objet_credit": objet, "montant_credit_fcfa": montant_credit,
    "duree_credit_mois": duree_credit, "taux_interet_annuel_pct": taux_interet,
    "mensualite_fcfa": mensualite, "taux_effort": taux_effort,
    "ratio_epargne_credit": ratio_epargne_credit,
    "possede_mobile_money": np.where(possede_mm,"Oui","Non"),
    "volume_mm_mensuel_fcfa": volume_mm,
    "frequence_mm_mensuelle": freq_mm,
    "anciennete_mobile_money_mois": anciennete_mm,
})

# ============================================================
# 6. CIBLE : DEFAUT (modele de risque latent, donc apprenable)
# ============================================================
def z(x): return (x - np.nanmean(x)) / np.nanstd(x)   # standardisation

hist_risk = df["historique_remboursement"].map(
    {"Bon":-0.8,"Moyen":0.3,"Mauvais":1.6,"Nouveau":0.4}).values
sect_risk = df["secteur_activite"].map(
    {"Agriculture":0.5,"Elevage":0.4,"Commerce":0.0,"Artisanat":0.1,
     "Transport":0.2,"Services":-0.2,"Transformation agroalimentaire":0.0}).values
groupe = (df["type_credit"]=="Groupe/solidaire").astype(float).values
garant = (df["possede_garant"]=="Oui").astype(float).values
mm_actif = np.where(df["possede_mobile_money"]=="Oui",
                    z(np.nan_to_num(df["volume_mm_mensuel_fcfa"].values, nan=0)), -0.3)

lin = (
      1.30*z(df["taux_effort"].values)                 # + effort = + risque (moteur principal)
    + 0.90*hist_risk
    - 0.55*z(df["revenu_mensuel_fcfa"].values)
    - 0.45*z(df["anciennete_membre_ans"].values)
    - 0.35*z(df["nombre_credits_anterieurs"].values.astype(float))
    - 0.40*z(df["epargne_moyenne_fcfa"].values)
    + 0.45*sect_risk
    - 0.50*groupe                                       # solidarite reduit le defaut
    - 0.30*garant
    - 0.35*mm_actif                                     # donnee alternative protectrice
    + 0.30*z((df["age"].values<28).astype(float))       # jeunes un peu plus risques
    + 0.25*z(df["montant_credit_fcfa"].values)
    + rng.normal(0, 0.6, N)                             # bruit irreductible
)
# calibration numerique de l'intercept pour atteindre exactement le taux cible
from scipy.optimize import brentq
def _ecart(b):
    return (1/(1+np.exp(-(b + lin)))).mean() - TAUX_DEFAUT_CIBLE
intercept = brentq(_ecart, -20, 20)
proba_defaut = 1/(1+np.exp(-(intercept + lin)))
df["defaut"] = (rng.random(N) < proba_defaut).astype(int)

# ============================================================
# 7. VALEURS MANQUANTES REALISTES (pour l'apurement)
# ============================================================
for col, taux in [("revenu_mensuel_fcfa",0.03),("niveau_instruction",0.02),
                  ("epargne_moyenne_fcfa",0.02),("anciennete_activite_ans",0.02)]:
    idx = rng.random(N) < taux
    df.loc[idx, col] = np.nan

# ============================================================
# 8. SAUVEGARDES
# ============================================================
csv_path = os.path.join(DOSSIER, "base_microcredit_uemoa.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# dictionnaire des variables
dico = [
 ("id_client","Identifiant unique du client","texte"),
 ("pays","Pays UEMOA de la cooperative","categoriel"),
 ("zone","Zone de residence (Urbain/Rural)","categoriel"),
 ("sexe","Sexe du client","categoriel"),
 ("age","Age en annees","numerique"),
 ("situation_matrimoniale","Situation matrimoniale","categoriel"),
 ("personnes_a_charge","Nombre de personnes a charge","numerique"),
 ("niveau_instruction","Niveau d'instruction","categoriel"),
 ("secteur_activite","Secteur d'activite economique","categoriel"),
 ("statut_activite","Formel ou informel","categoriel"),
 ("anciennete_activite_ans","Anciennete dans l'activite (annees)","numerique"),
 ("revenu_mensuel_fcfa","Revenu mensuel estime (FCFA)","numerique"),
 ("anciennete_membre_ans","Anciennete comme membre de la cooperative (annees)","numerique"),
 ("nombre_credits_anterieurs","Nombre de credits deja obtenus","numerique"),
 ("historique_remboursement","Qualite de remboursement passee","categoriel"),
 ("epargne_moyenne_fcfa","Solde d'epargne moyen (FCFA)","numerique"),
 ("type_credit","Credit individuel ou groupe/solidaire","categoriel"),
 ("possede_garant","Presence d'un garant/caution","categoriel"),
 ("objet_credit","Objet du credit demande","categoriel"),
 ("montant_credit_fcfa","Montant du credit demande (FCFA)","numerique"),
 ("duree_credit_mois","Duree du credit (mois)","numerique"),
 ("taux_interet_annuel_pct","Taux d'interet annuel (%)","numerique"),
 ("mensualite_fcfa","Echeance mensuelle estimee (FCFA)","numerique"),
 ("taux_effort","Mensualite / revenu mensuel","numerique"),
 ("ratio_epargne_credit","Epargne / montant du credit","numerique"),
 ("possede_mobile_money","Utilise le mobile money","categoriel"),
 ("volume_mm_mensuel_fcfa","Volume mensuel de transactions mobile money (FCFA)","numerique"),
 ("frequence_mm_mensuelle","Nombre de transactions mobile money par mois","numerique"),
 ("anciennete_mobile_money_mois","Anciennete du compte mobile money (mois)","numerique"),
 ("defaut","Cible : 1 = defaut de remboursement, 0 = bon payeur","binaire"),
]
pd.DataFrame(dico, columns=["variable","description","type"]).to_csv(
    os.path.join(DOSSIER,"dictionnaire_variables.csv"), index=False, encoding="utf-8-sig")

# resume de controle
with open(os.path.join(DOSSIER,"resume_base.txt"),"w",encoding="utf-8") as f:
    f.write("BASE MICROCREDIT UEMOA, resume de controle\n")
    f.write("="*50+"\n")
    f.write(f"Nombre de clients : {len(df)}\n")
    f.write(f"Nombre de variables : {df.shape[1]}\n")
    f.write(f"Taux de defaut : {df['defaut'].mean()*100:.1f}%\n\n")
    f.write("Repartition par sexe :\n"+df['sexe'].value_counts(normalize=True).round(3).to_string()+"\n\n")
    f.write("Repartition par secteur :\n"+df['secteur_activite'].value_counts(normalize=True).round(3).to_string()+"\n\n")
    f.write("Revenu mensuel (FCFA) :\n"+df['revenu_mensuel_fcfa'].describe().round(0).to_string()+"\n\n")
    f.write("Montant credit (FCFA) :\n"+df['montant_credit_fcfa'].describe().round(0).to_string()+"\n\n")
    f.write("Taux de defaut par historique :\n"+df.groupby('historique_remboursement')['defaut'].mean().round(3).to_string()+"\n")

print("OK. Base generee :", len(df), "clients,", df.shape[1], "variables.")
print("Taux de defaut :", round(df['defaut'].mean()*100,1), "%")
print("Fichiers dans :", DOSSIER)
