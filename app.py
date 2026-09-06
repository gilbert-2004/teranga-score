# -*- coding: utf-8 -*-
"""
TerangaScore, appli de demonstration (Streamlit)
Saisir un profil client -> score, decision et explication transparente.
Lancer :  streamlit run app.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from terangascore_scorer import (load_artefact, scorer_client,
                                  load_artefact_survie, courbe_survie_client)

st.set_page_config(page_title="TerangaScore", page_icon="🛡️", layout="wide")

NAVY = "#00578A"
st.markdown(f"""
<style>
.big-title {{font-size:2.2rem; font-weight:800; color:{NAVY}; margin-bottom:0}}
.tag {{color:#6B5A2C; font-style:italic; margin-top:0}}
.block {{background:#EAF3F8; border-left:5px solid {NAVY}; padding:10px 14px; border-radius:6px}}
</style>
""", unsafe_allow_html=True)

art = load_artefact()
art_survie = load_artefact_survie()

# En-tete
st.markdown('<p class="big-title">🛡️ TerangaScore</p>', unsafe_allow_html=True)
st.markdown('<p class="tag">La confiance qui se calcule. Scoring de microcredit interpretable pour les cooperatives de l\'UEMOA.</p>', unsafe_allow_html=True)

# Barre laterale : qualite du modele
with st.sidebar:
    st.header("Modele")
    st.metric("AUC", f"{art['auc']:.3f}")
    st.metric("Gini", f"{art['gini']:.3f}")
    st.metric("KS", f"{art['ks']:.3f}")
    st.caption(f"Seuil d'octroi : score >= {art['seuil']}")
    st.caption(f"{len(art['SEL'])} variables retenues (IV > 0.02)")

fm = art["form_meta"]
def widget(v, label=None):
    label = label or v
    kind = fm[v][0]
    if kind == "cat":
        return st.selectbox(label, fm[v][1])
    else:
        _, mn, med, mx = fm[v]
        return st.number_input(label, min_value=float(mn), max_value=float(mx),
                               value=float(med), step=1.0)

st.subheader("Profil du demandeur")
with st.form("profil"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Identite**")
        pays = widget("pays","Pays"); zone = widget("zone","Zone")
        sexe = widget("sexe","Sexe"); age = widget("age","Age")
        matri = widget("situation_matrimoniale","Situation matrimoniale")
        charge = widget("personnes_a_charge","Personnes a charge")
        instr = widget("niveau_instruction","Niveau d'instruction")
    with c2:
        st.markdown("**Activite et cooperative**")
        secteur = widget("secteur_activite","Secteur d'activite")
        statut = widget("statut_activite","Statut de l'activite")
        anc_act = widget("anciennete_activite_ans","Anciennete activite (ans)")
        revenu = widget("revenu_mensuel_fcfa","Revenu mensuel (FCFA)")
        anc_mem = widget("anciennete_membre_ans","Anciennete membre (ans)")
        nb_cred = widget("nombre_credits_anterieurs","Credits anterieurs")
        hist = st.selectbox("Historique de remboursement", ["Nouveau","Bon","Moyen","Mauvais"])
        epargne = widget("epargne_moyenne_fcfa","Epargne moyenne (FCFA)")
    with c3:
        st.markdown("**Credit demande et mobile money**")
        type_cred = widget("type_credit","Type de credit")
        garant = widget("possede_garant","Garant / caution")
        objet = widget("objet_credit","Objet du credit")
        montant = widget("montant_credit_fcfa","Montant demande (FCFA)")
        duree = st.selectbox("Duree (mois)", [3,6,9,12,18,24], index=3)
        taux = st.slider("Taux d'interet annuel (%)", 12.0, 24.0, 21.0, 0.5)
        mm = st.selectbox("Utilise le mobile money", ["Oui","Non"])
        vol_mm = st.number_input("Volume mobile money mensuel (FCFA)", 0.0, 5000000.0, 200000.0, 1000.0)
        freq_mm = st.number_input("Transactions mobile money / mois", 0.0, 120.0, 12.0, 1.0)
        anc_mm = st.number_input("Anciennete mobile money (mois)", 0.0, 120.0, 24.0, 1.0)
    submit = st.form_submit_button("Calculer le score", type="primary")

if submit:
    raw = dict(pays=pays, zone=zone, sexe=sexe, age=age, situation_matrimoniale=matri,
               personnes_a_charge=charge, niveau_instruction=instr, secteur_activite=secteur,
               statut_activite=statut, anciennete_activite_ans=anc_act, revenu_mensuel_fcfa=revenu,
               anciennete_membre_ans=anc_mem, nombre_credits_anterieurs=nb_cred,
               historique_remboursement=hist, epargne_moyenne_fcfa=epargne, type_credit=type_cred,
               possede_garant=garant, objet_credit=objet, montant_credit_fcfa=montant,
               duree_credit_mois=duree, taux_interet_annuel_pct=taux, possede_mobile_money=mm,
               volume_mm_mensuel_fcfa=vol_mm, frequence_mm_mensuelle=freq_mm,
               anciennete_mobile_money_mois=anc_mm)
    r = scorer_client(raw, art)

    st.divider()
    g1, g2, g3 = st.columns([1,1,2])
    g1.metric("Score TerangaScore", r["score"])
    g2.metric("Probabilite de defaut", f"{r['proba_defaut']*100:.1f}%")
    if r["decision"] == "Accorde":
        g3.success(f"### Decision : CREDIT ACCORDE  (score >= {art['seuil']})")
    else:
        g3.error(f"### Decision : CREDIT REFUSE  (score < {art['seuil']})")

    pct = float(np.clip((r["score"]-300)/(750-300), 0, 1))
    st.progress(pct, text=f"Score {r['score']} / 750")

    st.subheader("Explication de la decision (transparence BCEAO)")
    st.markdown('<div class="block">Chaque caracteristique du client rapporte des points. '
                'La somme des points donne le score. Voici la contribution de chaque variable.</div>',
                unsafe_allow_html=True)
    det = r["detail"].copy()
    base = art["OFFSET"]/len(art["SEL"]) - art["FACTOR"]*art["b0"]/len(art["SEL"])
    det["effet"] = det["points"] - base           # >0 favorable, <0 penalisant
    det = det.sort_values("effet")
    fig, ax = plt.subplots(figsize=(8, 6))
    couleurs = ["#C0392B" if e < 0 else "#1E8449" for e in det["effet"]]
    ax.barh(det["variable"], det["effet"], color=couleurs)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Effet sur le score (points, vert = favorable, rouge = penalisant)")
    ax.set_title("Contribution de chaque variable", color=NAVY, fontweight="bold")
    st.pyplot(fig)

    cc1, cc2 = st.columns(2)
    cc1.markdown("**Principaux facteurs favorables**")
    cc1.table(det.sort_values("effet", ascending=False).head(4)[["variable","classe","effet"]]
              .round(1).reset_index(drop=True))
    cc2.markdown("**Principaux facteurs de risque**")
    cc2.table(det.sort_values("effet").head(4)[["variable","classe","effet"]]
              .round(1).reset_index(drop=True))

    # --- Courbe de survie individuelle (modele de Cox) ---
    if art_survie is not None:
        st.divider()
        st.subheader("Trajectoire de risque dans le temps (analyse de survie)")
        st.markdown('<div class="block">La scorecard dit <b>si</b> le client risque de faire defaut. '
                    "L'analyse de survie dit <b>quand</b> : voici la probabilite que ce client "
                    "n'ait PAS fait defaut au fil des mois.</div>", unsafe_allow_html=True)
        mois, proba, s12 = courbe_survie_client(r["features"], art_survie)

        sv1, sv2 = st.columns([2, 1])
        with sv1:
            figs, axs = plt.subplots(figsize=(8, 4.5))
            axs.plot(mois, proba*100, color=NAVY, lw=2.5)
            axs.fill_between(mois, proba*100, 100, color=NAVY, alpha=0.07)
            axs.axvline(12, color="#B8860B", ls="--", lw=1)
            axs.annotate(f"{s12*100:.0f}% a 12 mois", (12, s12*100),
                         textcoords="offset points", xytext=(8, 10), color="#6B5A2C")
            axs.set_ylim(min(80, proba.min()*100 - 2), 100.5)
            axs.set_xlabel("Mois depuis l'octroi")
            axs.set_ylabel("Probabilite de survie du credit (%)")
            axs.set_title("Courbe de survie individuelle", color=NAVY, fontweight="bold")
            axs.grid(alpha=0.3)
            st.pyplot(figs)
        with sv2:
            st.metric("Survie a 12 mois", f"{s12*100:.0f}%")
            st.metric("Risque de defaut a 12 mois", f"{(1-s12)*100:.0f}%")
            st.caption(f"Modele de Cox, c-index = {art_survie['c_index']:.3f}. "
                       "Une survie elevee et plate indique un client durablement sain ; "
                       "une chute rapide signale un risque precoce.")
    else:
        st.caption("Astuce : lancez `python entrainer_survie.py` pour activer la courbe de survie individuelle.")
else:
    st.info("Renseignez le profil du demandeur puis cliquez sur **Calculer le score**.")
