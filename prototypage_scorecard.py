# -*- coding: utf-8 -*-
"""
TerangaScore, prototype de scorecard de credit INTERPRETABLE (from scratch)
===========================================================================
Pipeline pedagogique complet, sans boite noire :
  1. Chargement et separation train/test
  2. Apurement (les valeurs manquantes deviennent une classe a part entiere)
  3. WOE (Weight of Evidence) et IV (Information Value) par variable
  4. Selection des variables par pouvoir predictif (IV)
  5. Regression logistique sur les variables transformees en WOE
  6. Mise en points (scorecard) : Factor / Offset / PDO
  7. Evaluation : AUC, Gini, KS
  8. Sauvegardes : table IV, scorecard (points par classe), clients scores

Convention : cible defaut=1 (mauvais), 0=bon. Un score ELEVE = client SUR.
"""
import numpy as np, pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DOSSIER = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(DOSSIER, "base_microcredit_uemoa.csv"))

TARGET = "defaut"
DROP = ["id_client"]
NUM = ["age","personnes_a_charge","anciennete_activite_ans","revenu_mensuel_fcfa",
       "anciennete_membre_ans","nombre_credits_anterieurs","epargne_moyenne_fcfa",
       "montant_credit_fcfa","duree_credit_mois","taux_interet_annuel_pct",
       "mensualite_fcfa","taux_effort","ratio_epargne_credit","volume_mm_mensuel_fcfa",
       "frequence_mm_mensuelle","anciennete_mobile_money_mois"]
CAT = ["pays","zone","sexe","situation_matrimoniale","niveau_instruction",
       "secteur_activite","statut_activite","historique_remboursement",
       "type_credit","possede_garant","objet_credit","possede_mobile_money"]

# 1. Separation train / test (le binning se calcule sur TRAIN seulement, anti-fuite)
Xtr, Xte, ytr, yte = train_test_split(df.drop(columns=[TARGET]+DROP), df[TARGET],
                                      test_size=0.30, random_state=42, stratify=df[TARGET])
tr = Xtr.copy(); tr[TARGET]=ytr.values
te = Xte.copy(); te[TARGET]=yte.values

# ------------------------------------------------------------------
# 2-3. Binning + WOE/IV (appris sur TRAIN)
# ------------------------------------------------------------------
def bins_numeriques(serie, q=5):
    """Bornes de classes par quantiles (sur les valeurs non manquantes du train)."""
    vals = serie.dropna()
    bornes = np.unique(np.quantile(vals, np.linspace(0,1,q+1)))
    bornes[0]=-np.inf; bornes[-1]=np.inf
    return bornes

def classe_num(serie, bornes):
    c = pd.cut(serie, bins=bornes, include_lowest=True).astype(str)
    return c.where(serie.notna(), "Manquant")

def woe_iv(classe_tr, y):
    """Calcule WOE par classe et IV de la variable (lissage +0.5)."""
    d = pd.DataFrame({"c":classe_tr.values, "y":y.values})
    g = d.groupby("c")["y"].agg(["count","sum"])
    g["bad"]=g["sum"]; g["good"]=g["count"]-g["bad"]
    tot_bad=g["bad"].sum(); tot_good=g["good"].sum()
    g["p_bad"]=(g["bad"]+0.5)/(tot_bad+0.5*len(g))
    g["p_good"]=(g["good"]+0.5)/(tot_good+0.5*len(g))
    g["woe"]=np.log(g["p_good"]/g["p_bad"])
    g["iv"]=(g["p_good"]-g["p_bad"])*g["woe"]
    return g["woe"].to_dict(), float(g["iv"].sum())

# Apprentissage du binning et des WOE sur TRAIN
binning = {}   # var -> (type, bornes/None, woe_map, iv)
for v in NUM:
    bornes = bins_numeriques(tr[v])
    ctr = classe_num(tr[v], bornes)
    wmap, iv = woe_iv(ctr, tr[TARGET])
    binning[v] = ("num", bornes, wmap, iv)
for v in CAT:
    ctr = tr[v].astype(str).fillna("Manquant")
    wmap, iv = woe_iv(ctr, tr[TARGET])
    binning[v] = ("cat", None, wmap, iv)

# Table IV
iv_table = (pd.DataFrame([(v, binning[v][3]) for v in NUM+CAT], columns=["variable","IV"])
            .sort_values("IV", ascending=False).reset_index(drop=True))
def force(iv):
    return ("inutile" if iv<0.02 else "faible" if iv<0.1 else "moyen"
            if iv<0.3 else "fort" if iv<0.5 else "suspect (>0.5)")
iv_table["pouvoir_predictif"]=iv_table["IV"].apply(force)
iv_table.to_csv(os.path.join(DOSSIER,"table_IV.csv"), index=False, encoding="utf-8-sig")

# 4. Selection des variables (IV > 0.02, on ecarte le bruit)
SEL = iv_table.loc[iv_table["IV"]>0.02, "variable"].tolist()

# Fonction : transformer un jeu en WOE (avec le binning appris sur TRAIN)
def to_woe(data):
    out = pd.DataFrame(index=data.index)
    for v in SEL:
        typ, bornes, wmap, _ = binning[v]
        c = classe_num(data[v], bornes) if typ=="num" else data[v].astype(str).fillna("Manquant")
        out[v] = c.map(wmap).fillna(0.0)   # classe inconnue -> WOE neutre 0
    return out

Wtr, Wte = to_woe(tr), to_woe(te)

# 5. Regression logistique sur les WOE
modele = LogisticRegression(max_iter=1000)
modele.fit(Wtr, tr[TARGET])
beta = dict(zip(SEL, modele.coef_[0])); b0 = modele.intercept_[0]

# 6. Mise en points (scorecard)
PDO, SCORE_REF, ODDS_REF = 20, 600, 50        # 600 pts pour odds bon:mauvais = 50, +20 pts double les odds
FACTOR = PDO/np.log(2)
OFFSET = SCORE_REF - FACTOR*np.log(ODDS_REF)
n = len(SEL)

# Table scorecard : points par classe. points = -F*beta*woe - F*b0/n + OFFSET/n
lignes=[]
for v in SEL:
    _,_,wmap,_ = binning[v]
    for classe, woe in wmap.items():
        pts = -FACTOR*beta[v]*woe - FACTOR*b0/n + OFFSET/n
        lignes.append((v, classe, round(woe,4), round(pts,1)))
scorecard = pd.DataFrame(lignes, columns=["variable","classe","WOE","points"])
scorecard.to_csv(os.path.join(DOSSIER,"scorecard_points.csv"), index=False, encoding="utf-8-sig")

# 7. Scorer le test et evaluer
def scorer(data, W):
    logit = b0 + W[SEL].values @ np.array([beta[v] for v in SEL])
    p_def = 1/(1+np.exp(-logit))
    score = OFFSET - FACTOR*logit            # score eleve = sur
    return p_def, score
p_def, score = scorer(te, Wte)

auc = roc_auc_score(te[TARGET], p_def)
gini = 2*auc-1
# KS : ecart max des cumules de score entre bons et mauvais
dsc = pd.DataFrame({"score":score,"y":te[TARGET].values}).sort_values("score")
cum_bad = (dsc["y"]==1).cumsum()/ (dsc["y"]==1).sum()
cum_good= (dsc["y"]==0).cumsum()/ (dsc["y"]==0).sum()
ks = float((cum_bad-cum_good).abs().max())

# Sauvegarde des clients scores (test)
res = te.copy()
res["proba_defaut"]=p_def.round(4)
res["score"]=score.round(0)
res["decision"]=np.where(res["score"]>=550,"Accorde","Refuse")   # seuil illustratif
res.to_csv(os.path.join(DOSSIER,"clients_scores_test.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------
# 8. RAPPORT
# ------------------------------------------------------------------
print("="*60)
print("TERANGASCORE, prototype de scorecard")
print("="*60)
print(f"Train : {len(tr)} clients | Test : {len(te)} clients")
print(f"Variables retenues (IV>0.02) : {len(SEL)}\n")
print(">> Pouvoir predictif (IV) des variables :")
print(iv_table.to_string(index=False))
print("\n>> Performance sur le TEST :")
print(f"   AUC  = {auc:.3f}")
print(f"   Gini = {gini:.3f}")
print(f"   KS   = {ks:.3f}")
print(f"\n   (repere : AUC>0.7 correct, >0.8 bon ; Gini>0.4 correct ; KS>0.3 correct)")
print("\n>> Exemple de scorecard (extrait, variable la plus predictive) :")
top = iv_table.iloc[0]["variable"]
print(scorecard[scorecard["variable"]==top].to_string(index=False))
print("\nFichiers ecrits : table_IV.csv, scorecard_points.csv, clients_scores_test.csv")

# ------------------------------------------------------------------
# 9. SAUVEGARDE DE L'ARTEFACT (pour l'appli Streamlit)
# ------------------------------------------------------------------
import joblib
RAW_NUM_INPUT = ["age","personnes_a_charge","anciennete_activite_ans","revenu_mensuel_fcfa",
                 "anciennete_membre_ans","nombre_credits_anterieurs","epargne_moyenne_fcfa",
                 "montant_credit_fcfa","duree_credit_mois","taux_interet_annuel_pct",
                 "volume_mm_mensuel_fcfa","frequence_mm_mensuelle","anciennete_mobile_money_mois"]
form_meta = {}
for v in CAT:
    form_meta[v] = ("cat", sorted(df[v].dropna().astype(str).unique().tolist()))
for v in RAW_NUM_INPUT:
    s = df[v].dropna()
    form_meta[v] = ("num", float(s.min()), float(s.median()), float(s.max()))
artefact = dict(SEL=SEL, binning=binning, beta=beta, b0=float(b0),
                FACTOR=float(FACTOR), OFFSET=float(OFFSET), seuil=550,
                form_meta=form_meta, RAW_NUM_INPUT=RAW_NUM_INPUT, CAT=CAT,
                auc=float(auc), gini=float(gini), ks=float(ks))
joblib.dump(artefact, os.path.join(DOSSIER, "modele_terangascore.pkl"))
print("Artefact sauvegarde : modele_terangascore.pkl")
