"""
Modèle de script d'entraînement de DatafoncIA.

Mode d'emploi (détaillé dans le README, étape 5) :

    1. Copier ce fichier :      cp training/train_template.py training/train_dv3f.py
    2. Adapter les rubriques ci-dessous (SCHEMA, TABLES_DOCUMENTEES, DOCUMENTATION, EXEMPLES)
    3. Lancer l'entraînement :  uv run python training/train_dv3f.py
    4. Tester dans l'application, puis compléter DOCUMENTATION / EXEMPLES et relancer

Le script reconstruit toute la mémoire à chaque lancement : ce fichier est donc la
référence de ce que sait l'assistant. Il se relance sans risque autant de fois que nécessaire.
"""

import sys
from pathlib import Path

# Le code de l'assistant (agent Vanna 2) est dans app/core.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from core import CeremaVanna  # noqa: E402

# ---------------------------------------------------------------------------
# 1. La base
# ---------------------------------------------------------------------------

# Schéma PostgreSQL interrogé par l'assistant. La structure des tables (colonnes,
# types) et les commentaires de colonnes sont lus automatiquement dans la base.
SCHEMA = "dv3f_2026_1"

# Tables dont la documentation des variables est récupérée sur
# doc-datafoncier.cerema.fr (laisser la liste vide pour une base hors Datafoncier).
TABLES_DOCUMENTEES = ["mutation", "local"]

# ---------------------------------------------------------------------------
# 2. La documentation métier
# ---------------------------------------------------------------------------
# Ce que la structure de la base ne dit pas : vocabulaire, règles de calcul,
# filtres à appliquer, pièges. Un texte par idée, en nommant les tables et les
# colonnes concernées. Les textes ci-dessous sont des exemples à adapter.
DOCUMENTATION = [
    """
    Une mutation est une transaction immobilière (vente, adjudication, échange...).
    Chaque ligne de la table dv3f_2026_1.mutation correspond à une mutation.
    """,
    """
    Pour filtrer les mutations d'une commune, utiliser la colonne l_codinsee de la table
    dv3f_2026_1.mutation : c'est un tableau de codes INSEE, à tester avec
    '62041' = ANY(l_codinsee). Le code INSEE d'Arras est 62041.
    Pour filtrer sur un département, utiliser la colonne coddep (exemple : coddep = '62').
    """,
    """
    Le prix d'une mutation est la colonne valeurfonc de la table dv3f_2026_1.mutation.
    Pour un prix au m², diviser valeurfonc par la surface bâtie sbati en écartant les
    lignes où sbati vaut 0 ou NULL. Préférer la médiane à la moyenne pour les prix.
    """,
]

# ---------------------------------------------------------------------------
# 3. Les exemples question / SQL
# ---------------------------------------------------------------------------
# Des questions types avec la requête SQL validée qui y répond. C'est la source
# d'entraînement la plus efficace : l'assistant s'inspire des exemples les plus
# proches de la question posée. Chaque requête est vérifiée sur la base avant
# d'être enregistrée ; une requête invalide est rejetée et signalée.
# Les exemples ci-dessous sont à adapter et à compléter.
EXEMPLES = [
    {
        "question": "Nombre de mutations par année dans le Pas-de-Calais",
        "sql": """
            SELECT anneemut, count(*) AS nb_mutations
            FROM dv3f_2026_1.mutation
            WHERE coddep = '62'
            GROUP BY anneemut
            ORDER BY anneemut
        """,
    },
    {
        "question": "Prix médian des mutations sur Arras par année",
        "sql": """
            SELECT anneemut,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY valeurfonc) AS prix_median
            FROM dv3f_2026_1.mutation
            WHERE '62041' = ANY(l_codinsee)
            GROUP BY anneemut
            ORDER BY anneemut
        """,
    },
]

# ---------------------------------------------------------------------------
# 4. Questions de contrôle (facultatif)
# ---------------------------------------------------------------------------
# Posées à l'assistant en fin d'entraînement pour voir le SQL qu'il produit.
# Chaque question consomme des appels au LLM : laisser la liste vide pour sauter l'étape.
QUESTIONS_DE_CONTROLE = [
    # "Nombre de mutations par année sur Arras",
]


def main() -> int:
    cerema = CeremaVanna()

    # La mémoire est reconstruite entièrement à partir de ce fichier
    cerema.clean()
    success = cerema.train_complete_dataset(
        schema=SCHEMA,
        doc_tables=TABLES_DOCUMENTEES,
        documentation=DOCUMENTATION,
        examples=EXEMPLES,
    )

    training_data = cerema.get_training_data()
    print(f"\n{len(training_data)} éléments en mémoire :")
    print(training_data["training_data_type"].value_counts().to_string())

    for question in QUESTIONS_DE_CONTROLE:
        answer = cerema.ask(question)
        print(f"\nQuestion : {question}")
        print(f"SQL ({answer.attempts} tentative(s)) :\n{answer.sql}")
        if answer.error:
            print(f"Erreur : {answer.error}")
        elif answer.df is not None:
            print(answer.df.head(10).to_string())

    if not success:
        print(
            "\n⚠️ Entraînement incomplet : voir les messages « Erreur » ou « rejeté » ci-dessus"
        )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
