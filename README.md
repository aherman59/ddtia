# DDTIA

Assistant d'analyse de données foncières pour la DDTM 62 : l'utilisateur pose une
question en français, l'assistant génère la requête SQL, l'exécute sur la base
PostgreSQL (données DV3F), affiche le tableau de résultat, un graphique et un résumé.

Le projet s'appuie sur [Vanna AI](https://vanna.ai) **2.x**, un LLM au choix
([Albert API](https://albert.api.etalab.gouv.fr) de la DINUM ou
[Mistral](https://mistral.ai)) et une application [Streamlit](https://streamlit.io).

## Fonctionnement

```
Question ──► Agent Vanna 2 ──► LLM (Albert API ou Mistral)
                 │  ▲
                 │  └─ contexte : mémoire ChromaDB (structure des tables,
                 │     documentation des variables, exemples question/SQL)
                 ├─► outil run_sql        ──► PostgreSQL (lecture seule)
                 └─► outil visualize_data ──► graphique Plotly
```

- **Auto-correction** : si la requête SQL échoue, l'erreur est renvoyée au LLM qui
  corrige et réessaie (3 tentatives au maximum).
- **Lecture seule** : seules les requêtes `SELECT` / `WITH` sont acceptées, et elles
  sont exécutées dans une transaction PostgreSQL en lecture seule.
- **Conversation** : l'agent garde le fil de la conversation, on peut donc poser des
  questions de suivi (« et pour Béthune ? »). Le bouton *Nouvelle conversation* repart de zéro.

## Organisation du dépôt

Le code est dans le dossier `ddtia/` ; les chemins ci-dessous sont relatifs à ce dossier.

| Chemin | Rôle |
| --- | --- |
| `app/core.py` | Cœur du projet : agent Vanna 2, services LLM (Albert, Mistral), outil SQL, mémoire, fonctions d'entraînement (`CeremaVanna`) |
| `app/app.py` | Interface Streamlit |
| `app/vanna_calls.py` | Lien entre Streamlit et `core.py` |
| `training/train_template.py` | Modèle de script d'entraînement, à copier et à adapter (étape 4) |
| `training/notebook_marimo_test.py` | Notebook marimo pour tester l'assistant et consulter sa mémoire |
| `justfile` | Raccourcis de commandes (`just app`, `just train`, ...) |
| `.env.dist` | Modèle de fichier de configuration |

## Prérequis

- **Python 3.11 ou plus** (installé automatiquement par `uv` si besoin) ;
- **[uv](https://docs.astral.sh/uv/)** pour gérer l'environnement et les dépendances ;
- **[just](https://github.com/casey/just)** (facultatif) pour les raccourcis de commandes ;
- une **base PostgreSQL** et le client `psql` ;
- une **clé d'API pour le LLM** : [Albert API](https://albert.api.etalab.gouv.fr)
  ou Mistral ([console.mistral.ai](https://console.mistral.ai)) ;
- un accès internet (API du LLM, documentation Datafoncier et téléchargement
  du modèle d'embedding de ChromaDB, environ 80 Mo, à la première utilisation).

Le projet est conçu pour tourner sur le Datalab (Onyxia / SSP Cloud), dans un service
VSCode-python auquel est associé un service PostgreSQL : les variables `PG*`
y sont déjà définies. Il fonctionne aussi sur un poste local (voir l'étape 3).

## Installation

Toutes les commandes se lancent depuis le sous-dossier `ddtia/` du dépôt, qui contient
`pyproject.toml` (d'où le `cd ddtia/ddtia` après le clonage).

### 1. Récupérer le projet

```bash
git clone https://github.com/aherman59/ddtia.git
cd ddtia/ddtia
```

### 2. Installer les outils et les dépendances

```bash
# uv (si absent)
curl -LsSf https://astral.sh/uv/install.sh | sh        # Linux / macOS
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# just (facultatif)
uv tool install rust-just

# environnement virtuel + dépendances (crée .venv à partir de uv.lock)
uv sync
```

### 3. Configurer l'environnement

```bash
cp .env.dist .env
```

Puis renseigner `.env` :

| Variable | Obligatoire | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | non | `albert` ou `mistral`. Défaut : `albert` si `ALBERT_API_KEY` est définie, sinon `mistral` |
| `ALBERT_API_KEY` | si Albert | Clé d'API Albert |
| `ALBERT_MODEL` | non | Identifiant ou alias du modèle Albert (défaut : `openweight-code`, voir ci-dessous) |
| `ALBERT_BASE_URL` | non | Défaut : `https://albert.api.etalab.gouv.fr/v1` |
| `MISTRAL_API_KEY` | si Mistral | Clé d'API Mistral |
| `MISTRAL_MODEL` | non | Modèle Mistral (défaut : `codestral-latest`) |
| `CHROMA_DIR` | oui | Dossier de la mémoire ChromaDB (ex. `/home/onyxia/work/chroma`). Défaut : `./chroma` |
| `CHROMA_COLLECTION` | non | Nom de la collection ChromaDB (défaut : `ddtia`) |
| `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` | hors Datalab | Connexion PostgreSQL. Déjà définies sur le Datalab |
| `DDTIA_WORK_DIR` | non | Dossier des fichiers temporaires de l'agent (défaut : dossier temporaire du système) |

**Choix du modèle.** Quel que soit le fournisseur, le modèle doit gérer l'appel d'outils
(*function calling*), sans quoi l'agent ne peut pas exécuter de SQL.

Avec Albert, le modèle par défaut est `openweight-code` (modèle spécialisé code, adapté à
la génération de SQL). `ALBERT_MODEL` accepte l'identifiant ou l'un des alias d'un modèle
de type `text-generation`. Pour consulter la liste :

```bash
curl -s https://albert.api.etalab.gouv.fr/v1/models -H "Authorization: Bearer $ALBERT_API_KEY"
```

Modèles Albert testés avec l'agent (appel d'outils, correction SQL, graphique) en
novembre 2026 : `openweight-code` (qwen3-coder-30b), `openweight-large` (gpt-oss-120b),
`openweight-medium` (mistral-small-3.2-24b), `openweight-small` (ministral-3-8b),
`deepseek-v4-flash` et `gemma-4-31b-it`. Un alias `openweight-*` peut pointer vers un
autre modèle au fil du temps ; pour figer le modèle, utiliser son identifiant.

### 4. Entraîner l'assistant

L'assistant ne connaît la base que par ce qu'on lui apprend. L'entraînement remplit sa
mémoire (ChromaDB, dossier `CHROMA_DIR`) à partir de cinq sources :

| Source | Origine | À écrire ? |
| --- | --- | --- |
| Structure des tables (colonnes, types) | lue dans PostgreSQL | non |
| Commentaires des colonnes | lus dans PostgreSQL | non |
| Documentation des variables | [doc-datafoncier.cerema.fr](https://doc-datafoncier.cerema.fr) | non |
| Documentation métier | script d'entraînement | **oui** |
| Exemples question / SQL | script d'entraînement | **oui** |

Les trois premières sont automatiques. Les deux dernières se rédigent dans un script
d'entraînement, créé à partir du modèle `training/train_template.py`.

#### a. Créer son script à partir du modèle

```bash
cp training/train_template.py training/train_dv3f.py
```

Ce script devient la référence de ce que sait l'assistant : le versionner avec le projet.

#### b. Décrire la base

En haut du script :

```python
SCHEMA = "dv3f_2026_1"                     # schéma PostgreSQL interrogé
TABLES_DOCUMENTEES = ["mutation", "local"]  # tables documentées sur doc-datafoncier
```

#### c. Rédiger la documentation métier

`DOCUMENTATION` est une liste de textes libres. Y mettre ce que la structure de la base
ne dit pas : vocabulaire, règles de calcul, filtres à appliquer, pièges.

```python
DOCUMENTATION = [
    """
    Pour filtrer les mutations d'une commune, utiliser la colonne l_codinsee de la table
    dv3f_2025_1.mutation : c'est un tableau de codes INSEE, à tester avec
    '62041' = ANY(l_codinsee). Le code INSEE d'Arras est 62041.
    """,
]
```

Conseils :

- un texte par idée, court : l'assistant ne reçoit que les textes les plus proches de la
  question posée ;
- nommer la table et la colonne avec leur nom exact (`schema.table`, `colonne`) ;
- employer les mots des utilisateurs (« prix », « vente », « maison »...) à côté des noms
  techniques, c'est ce qui permet de retrouver le texte.

#### d. Écrire des exemples question / SQL

`EXEMPLES` est une liste de questions types avec la requête validée qui y répond. C'est
la source la plus efficace : l'assistant s'inspire des exemples les plus proches.

```python
EXEMPLES = [
    {
        "question": "Nombre de mutations par année dans le Pas-de-Calais",
        "sql": """
            SELECT anneemut, count(*) AS nb_mutations
            FROM dv3f_2025_1.mutation
            WHERE coddep = '62'
            GROUP BY anneemut
            ORDER BY anneemut
        """,
    },
]
```

Conseils :

- formuler la question comme le ferait un utilisateur ;
- n'enregistrer que des requêtes vérifiées, dont le résultat a été contrôlé ;
- couvrir les cas difficiles (jointures, filtres par commune, médianes, typologie de
  biens) plutôt que de multiplier les variantes d'une même requête ;
- préfixer les tables par leur schéma.

Chaque requête est vérifiée sur la base (`EXPLAIN`) avant d'être enregistrée : une requête
invalide est rejetée et signalée, sans bloquer le reste de l'entraînement.

#### e. Lancer l'entraînement

```bash
uv run python training/train_dv3f.py
# ou : just train training/train_dv3f.py
```

Le script vide la mémoire, recharge toutes les sources, puis affiche le nombre d'éléments
enregistrés par type. Il se termine en erreur si une phase a échoué ou si un exemple a été
rejeté : lire alors les lignes « Erreur » et « rejeté » de la sortie.

Le script n'a pas besoin du LLM, sauf si `QUESTIONS_DE_CONTROLE` est renseignée : ces
questions sont alors posées à l'assistant en fin d'entraînement, et le SQL produit est affiché.

#### f. Tester et améliorer

Poser des questions dans l'application (étape 5) ou dans le notebook marimo, qui affiche
aussi le contenu de la mémoire :

```bash
just notebook
# ou : uv run marimo edit training/notebook_marimo_test.py --host 0.0.0.0 --port 5000
```

Quand une réponse est fausse :

1. regarder le SQL produit (onglet *SQL* de l'application) ;
2. si l'assistant se trompe de colonne, de filtre ou de règle de calcul, ajouter un texte
   dans `DOCUMENTATION` ;
3. si la logique de la requête est mauvaise, ajouter dans `EXEMPLES` la question avec la
   bonne requête ;
4. relancer le script d'entraînement, puis reposer la question dans une nouvelle conversation.

Comme le script reconstruit toute la mémoire, il se relance sans risque. L'entraînement
est aussi à refaire à chaque changement de schéma ou de millésime (penser à mettre à jour
`SCHEMA` et le nom du schéma dans les textes et les requêtes).

### 5. Lancer l'application

```bash
just app
# ou : uv run streamlit run app/app.py --server.address 0.0.0.0 --server.port 8000
```

L'application est disponible sur le port 8000 (`http://localhost:8000` en local ; sur le
Datalab, ouvrir le port 8000 du service).

## Utiliser l'assistant depuis Python

```python
import sys
sys.path.insert(0, "app")
from core import CeremaVanna

cerema = CeremaVanna()
reponse = cerema.ask("Nombre de mutations par année sur Arras")

reponse.sql       # requête SQL exécutée
reponse.df        # résultat (DataFrame pandas)
reponse.figure    # graphique Plotly (ou None)
reponse.summary   # résumé rédigé par le LLM
reponse.error     # message d'erreur si la question n'a pas abouti
reponse.attempts  # nombre de requêtes SQL exécutées

cerema.get_training_data()   # contenu de la mémoire

# Ajout ponctuel à la mémoire (perdu au prochain entraînement complet :
# pour le conserver, l'ajouter au script d'entraînement)
cerema.train(question="...", sql="SELECT ...")
cerema.train(documentation="La variable ... correspond à ...")
```

## Dépannage

| Symptôme | Piste |
| --- | --- |
| `ALBERT_API_KEY` / `MISTRAL_API_KEY n'est pas définie` | Fichier `.env` absent ou incomplet (étape 3) |
| Erreur 404 du LLM (visible en mode debug) | `ALBERT_MODEL` ne correspond à aucun modèle : vérifier avec `/v1/models` |
| L'assistant répond par du texte ou du SQL sans jamais afficher de résultat | Le modèle ne gère pas l'appel d'outils : en choisir un autre |
| `Connexion PostgreSQL non configurée` | Variables `PG*` absentes : les ajouter dans `.env` hors Datalab |
| Réponses hors sujet, tables ou colonnes inventées | Mémoire vide ou `CHROMA_DIR` différent entre l'entraînement et l'application : refaire l'étape 5 |
| Erreur 401 ou 429 du LLM (visible en mode debug) | Clé invalide ou quota dépassé |
| Premier lancement long | Téléchargement du modèle d'embedding de ChromaDB |

Le *mode debug* de la barre latérale affiche le message d'erreur complet et le détail
des tentatives SQL.
