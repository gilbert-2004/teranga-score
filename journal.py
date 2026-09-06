# -*- coding: utf-8 -*-
"""
TerangaScore, journal des decisions (mode hors ligne, SQLite)
=============================================================
Chaque decision de scoring est tracee localement dans une base SQLite, sans
dependance reseau. Permet a une caisse sur un poste modeste et sans connexion
permanente de garder une piste d'audit (exigence de tracabilite BCEAO) et de
produire des statistiques de portefeuille.
"""
import os
import json
import sqlite3
from datetime import datetime

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(DOSSIER, "journal_decisions.db")


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    """Cree la table si besoin (idempotent)."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                horodatage    TEXT    NOT NULL,
                score         INTEGER NOT NULL,
                proba_defaut  REAL    NOT NULL,
                decision      TEXT    NOT NULL,
                survie_12m    REAL,
                pays          TEXT,
                secteur       TEXT,
                montant       REAL,
                client_json   TEXT    NOT NULL
            )
        """)


def enregistrer(client: dict, score: int, proba: float, decision: str, survie_12m=None) -> int:
    """Enregistre une decision et retourne son identifiant local."""
    init_db()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO decisions
               (horodatage, score, proba_defaut, decision, survie_12m, pays, secteur, montant, client_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(timespec="seconds"), int(score), float(proba), decision,
             None if survie_12m is None else float(survie_12m),
             client.get("pays"), client.get("secteur_activite"),
             client.get("montant_credit_fcfa"), json.dumps(client, ensure_ascii=False)),
        )
        return cur.lastrowid


def statistiques() -> dict:
    """Statistiques agregees du portefeuille journalise."""
    init_db()
    with _conn() as c:
        n = c.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        if not n:
            return {"total": 0, "accordes": 0, "refuses": 0, "taux_acceptation": None,
                    "montant_total_accorde": 0, "score_moyen": None}
        acc = c.execute("SELECT COUNT(*) FROM decisions WHERE decision='Accorde'").fetchone()[0]
        mnt = c.execute("SELECT COALESCE(SUM(montant),0) FROM decisions WHERE decision='Accorde'").fetchone()[0]
        sco = c.execute("SELECT AVG(score) FROM decisions").fetchone()[0]
        return {
            "total": n, "accordes": acc, "refuses": n - acc,
            "taux_acceptation": round(acc / n, 3),
            "montant_total_accorde": round(mnt, 0),
            "score_moyen": round(sco, 1),
        }


def dernieres(limite: int = 20) -> list:
    """Renvoie les dernieres decisions (sans le JSON brut complet)."""
    init_db()
    with _conn() as c:
        rows = c.execute(
            """SELECT id, horodatage, score, proba_defaut, decision, survie_12m,
                      pays, secteur, montant
               FROM decisions ORDER BY id DESC LIMIT ?""", (limite,)).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Base de journalisation prete : {DB}")
    print("Statistiques actuelles :", statistiques())
