# -*- coding: utf-8 -*-
"""
TerangaScore, analyse de survie du risque de credit (time-to-default)
=====================================================================
Complement de la scorecard : au lieu de "va-t-il faire defaut ?" (binaire),
on modelise "QUAND" le defaut survient, en gerant la CENSURE (credits arrives
a terme sans defaut).

  1. Kaplan-Meier : courbe de survie globale et par groupe (historique)
  2. Test du log-rank (les groupes ont-ils des survies differentes ?)
  3. Modele de Cox (hazards proportionnels) : effet de chaque variable sur
     le risque instantane de defaut (hazard ratio), interpretable.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

DOSSIER = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DOSSIER, "base_microcredit_uemoa.csv"))
T, E = "duree_observation_mois", "evenement_defaut"

# ============================================================
# 1-2. KAPLAN-MEIER (global + par historique) et log-rank
# ============================================================
fig, ax = plt.subplots(figsize=(9,5.5))
kmf = KaplanMeierFitter()
kmf.fit(df[T], df[E], label="Ensemble du portefeuille")
kmf.plot_survival_function(ax=ax, color="#00578A", lw=2.5)

couleurs = {"Bon":"#1E8449","Moyen":"#B8860B","Nouveau":"#1D7AAE","Mauvais":"#C0392B"}
for grp, col in couleurs.items():
    m = df["historique_remboursement"]==grp
    if m.sum() > 20:
        k = KaplanMeierFitter()
        k.fit(df.loc[m,T], df.loc[m,E], label=f"Historique : {grp}")
        k.plot_survival_function(ax=ax, color=col, lw=1.6, ci_show=False)
ax.set_title("Courbe de survie du credit (probabilite de NE PAS avoir fait defaut)",
             color="#00578A", fontweight="bold")
ax.set_xlabel("Mois depuis l'octroi"); ax.set_ylabel("Probabilite de survie")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(DOSSIER,"courbe_survie.png"), dpi=130)

logrank = multivariate_logrank_test(df[T], df["historique_remboursement"], df[E])

# ============================================================
# 3. MODELE DE COX (hazards proportionnels)
# ============================================================
num = ["taux_effort","revenu_mensuel_fcfa","anciennete_membre_ans","epargne_moyenne_fcfa",
       "montant_credit_fcfa","age","nombre_credits_anterieurs"]
d = df[[T,E]+num+["historique_remboursement","type_credit","secteur_activite"]].copy()
# imputation simple des numeriques, puis standardisation
for c in num:
    d[c] = d[c].fillna(d[c].median())
    d[c] = (d[c]-d[c].mean())/d[c].std()
# categoriels -> indicatrices
d["hist_Mauvais"] = (d["historique_remboursement"]=="Mauvais").astype(int)
d["hist_Moyen"]   = (d["historique_remboursement"]=="Moyen").astype(int)
d["hist_Nouveau"] = (d["historique_remboursement"]=="Nouveau").astype(int)
d["credit_solidaire"] = (d["type_credit"]=="Groupe/solidaire").astype(int)
d["secteur_agricole"] = d["secteur_activite"].isin(["Agriculture","Elevage"]).astype(int)
d = d.drop(columns=["historique_remboursement","type_credit","secteur_activite"])

cph = CoxPHFitter(penalizer=0.05)
cph.fit(d, duration_col=T, event_col=E)

hr = cph.summary[["coef","exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].copy()
hr.columns = ["coef","hazard_ratio","HR_bas_95","HR_haut_95","p_value"]
hr = hr.sort_values("hazard_ratio", ascending=False).round(3)
hr.to_csv(os.path.join(DOSSIER,"hazard_ratios.csv"), encoding="utf-8-sig")

# ============================================================
# RAPPORT
# ============================================================
print("="*62)
print("TERANGASCORE, analyse de survie du risque de credit")
print("="*62)
print(f"Clients : {len(df)} | evenements (defauts) : {int(df[E].sum())} | "
      f"censures : {int((df[E]==0).sum())} ({(df[E]==0).mean()*100:.0f}%)")
print(f"Temps median au defaut : {int(df.loc[df[E]==1,T].median())} mois\n")

print(">> Kaplan-Meier, survie a quelques echeances :")
for t in [3,6,12]:
    print(f"   a {t:>2} mois : {kmf.survival_function_at_times(t).iloc[0]*100:5.1f}% des credits sains")
print(f"\n>> Test du log-rank (survie differente selon l'historique) :")
print(f"   p-value = {logrank.p_value:.2e}  ->",
      "difference tres significative" if logrank.p_value<0.001 else "difference significative" if logrank.p_value<0.05 else "non significative")

print(f"\n>> Modele de Cox, concordance (c-index) = {cph.concordance_index_:.3f}")
print("   (analogue de l'AUC pour la survie ; >0.7 correct, >0.8 bon)\n")
print(">> Hazard ratios (HR>1 = risque accru, HR<1 = protecteur) :")
print(hr.to_string())
print("\nFichiers ecrits : courbe_survie.png, hazard_ratios.csv")
