"""
Cœur de DatafoncIA, basé sur Vanna 2.x (architecture « agent »).

Vanna 2 remplace l'ancienne classe `VannaBase` (generate_sql / run_sql / train...)
par un `Agent` qui dialogue avec le LLM et appelle des outils :

- `run_sql`        : exécute le SQL généré sur PostgreSQL (lecture seule) ;
- `visualize_data` : construit un graphique Plotly à partir du résultat.

Le LLM est au choix Albert API (DINUM) ou Mistral (variable `LLM_PROVIDER`).

Les erreurs SQL sont renvoyées au LLM, qui corrige sa requête et réessaie
(auto-correction, `MAX_SQL_ATTEMPTS` tentatives au maximum).

Les données d'entraînement (structure de la base, documentation, exemples
question/SQL) sont stockées dans une mémoire ChromaDB (`ChromaAgentMemory`) et
injectées dans le prompt système à chaque question.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type

import dotenv
import pandas as pd
import requests
from vanna import Agent, AgentConfig
from vanna.capabilities.sql_runner import RunSqlToolArgs, SqlRunner
from vanna.components import (
    ChartComponent,
    DataFrameComponent,
    NotificationComponent,
    RichTextComponent,
    SimpleTextComponent,
    StatusCardComponent,
    UiComponent,
)
from vanna.core.agent.config import DEFAULT_UI_FEATURES, UiFeature, UiFeatures
from vanna.core.enhancer import LlmContextEnhancer
from vanna.core.llm import LlmRequest, LlmResponse, LlmService, LlmStreamChunk
from vanna.core.registry import ToolRegistry
from vanna.core.system_prompt import SystemPromptBuilder
from vanna.core.tool import Tool, ToolCall, ToolContext, ToolResult, ToolSchema
from vanna.core.user import RequestContext, User, UserResolver
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.integrations.local import LocalFileSystem
from vanna.integrations.openai import OpenAILlmService
from vanna.tools import VisualizeDataTool

logger = logging.getLogger("ddtia")

# ---------------------------------------------------------------------------
# Configuration (fichier .env + variables d'environnement)
# ---------------------------------------------------------------------------
dotenv.load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "codestral-latest")

# Albert API (DINUM) : API compatible OpenAI
ALBERT_API_KEY = os.getenv("ALBERT_API_KEY")
ALBERT_BASE_URL = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1").rstrip("/")
# Identifiant ou alias d'un modèle listé par GET /v1/models (il doit gérer l'appel d'outils)
ALBERT_MODEL = os.getenv("ALBERT_MODEL") or "openweight-code"

# Fournisseur du LLM : "albert" ou "mistral".
# Par défaut : Albert si ALBERT_API_KEY est définie, sinon Mistral.
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or ("albert" if ALBERT_API_KEY else "mistral")).lower()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ddtia")

# Dossier de travail : résultats CSV échangés entre run_sql et visualize_data
WORK_DIR = os.getenv("DDTIA_WORK_DIR") or str(Path(tempfile.gettempdir()) / "ddtia")

# Variables PG (fournies automatiquement par le Datalab / à définir dans .env)
PGHOST = os.getenv("PGHOST")
PGDATABASE = os.getenv("PGDATABASE")
PGUSER = os.getenv("PGUSER")
PGPORT = os.getenv("PGPORT")
PGPASSWORD = os.getenv("PGPASSWORD")

DOC_DATAFONCIER_TRAINING_URL = "https://doc-datafoncier.cerema.fr/api/variables/"

MAX_SQL_ATTEMPTS = 3  # nombre maximal de requêtes SQL en échec par question
N_CONTEXT_DOCS = 10  # éléments de documentation injectés dans le prompt
N_CONTEXT_EXAMPLES = 10  # exemples question/SQL injectés dans le prompt
MAX_LLM_PREVIEW_CHARS = 2000  # taille de l'aperçu du résultat renvoyé au LLM
MAX_COMPONENT_ROWS = 100  # lignes recopiées dans le composant d'interface

RUN_SQL_TOOL_NAME = "run_sql"

# Erreurs techniques (LLM...) de la requête en cours, pour le mode debug.
_request_errors: ContextVar[Optional[List[str]]] = ContextVar(
    "ddtia_request_errors", default=None
)


def run_sync(coro):
    """Exécute une coroutine depuis du code synchrone.

    Fonctionne aussi lorsqu'une boucle asyncio tourne déjà dans le thread
    courant (notebook marimo) : la coroutine est alors lancée dans un thread dédié.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


# ---------------------------------------------------------------------------
# Aide à l'auto-correction SQL
# ---------------------------------------------------------------------------
COLUMN_NOT_FOUND_PATTERNS = [
    r'column\s+"?([a-zA-Z0-9_.]+)"?\s+does\s+not\s+exist',
    r'unknown\s+column\s+"?([a-zA-Z0-9_.]+)"?',
    r'la\s+colonne\s+«?\s*"?([a-zA-Z0-9_.]+)"?\s*»?\s+n\'existe\s+pas',
]

TABLE_NOT_FOUND_PATTERNS = [
    r'relation\s+"?([a-zA-Z0-9_.]+)"?\s+does\s+not\s+exist',
    r'table\s+"?([a-zA-Z0-9_.]+)"?\s+does\s+not\s+exist',
    r'unknown\s+table\s+"?([a-zA-Z0-9_.]+)"?',
    r'la\s+relation\s+«?\s*"?([a-zA-Z0-9_.]+)"?\s*»?\s+n\'existe\s+pas',
]


def _extract_missing_element(err_msg: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extrait le nom de la colonne ou table manquante depuis le message d'erreur
    Returns: (element_type, element_name) où element_type = 'column' ou 'table'
    """
    for pat in COLUMN_NOT_FOUND_PATTERNS:
        m = re.search(pat, err_msg, flags=re.IGNORECASE)
        if m:
            return ("column", m.group(1))

    for pat in TABLE_NOT_FOUND_PATTERNS:
        m = re.search(pat, err_msg, flags=re.IGNORECASE)
        if m:
            return ("table", m.group(1))

    return (None, None)


def _repair_hint(err_msg: str) -> str:
    """Consigne de correction renvoyée au LLM avec l'erreur SQL."""
    element_type, missing = _extract_missing_element(err_msg)

    if element_type == "column":
        return (
            f"ERREUR DÉTECTÉE: La colonne '{missing}' n'existe pas dans le schéma.\n"
            "INSTRUCTION: Corrige la requête SQL pour utiliser une colonne réellement présente.\n"
            "Vérifie le schéma de la base de données et utilise les noms de colonnes exacts "
            "(au besoin, interroge information_schema.columns).\n"
            "Conserve l'intention de la question puis rappelle run_sql."
        )
    if element_type == "table":
        return (
            f"ERREUR DÉTECTÉE: La table '{missing}' n'existe pas dans le schéma.\n"
            "INSTRUCTION: Corrige la requête SQL pour utiliser une table réellement présente, "
            "préfixée par son schéma (au besoin, interroge information_schema.tables).\n"
            "Conserve l'intention de la question puis rappelle run_sql."
        )
    return (
        f"ERREUR SQL DÉTECTÉE: {err_msg}\n"
        "INSTRUCTION: Corrige la requête SQL pour résoudre cette erreur puis rappelle run_sql."
    )


def _clean_sql(sql: str) -> str:
    """Nettoie le SQL produit par le LLM (balises markdown, échappements Mistral)."""
    sql = sql.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", sql, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        sql = fenced.group(1).strip()
    # Certains modèles échappent les underscores ("\_")
    sql = sql.replace("\\_", "_")
    return sql.rstrip().rstrip(";").rstrip()


def _is_read_only(sql: str) -> bool:
    """Vrai si la requête commence par SELECT ou WITH (hors commentaires)."""
    without_comments = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    without_comments = re.sub(r"--[^\n]*", " ", without_comments)
    return bool(re.match(r"\s*\(*\s*(select|with)\b", without_comments, flags=re.IGNORECASE))


# ---------------------------------------------------------------------------
# LLM : Mistral (Vanna 2 ne fournit plus d'intégration Mistral native)
# ---------------------------------------------------------------------------
class MistralLlmService(LlmService):
    """Service LLM Vanna 2 s'appuyant sur l'API Mistral (avec appel d'outils)."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from mistralai import Mistral

        api_key = api_key or MISTRAL_API_KEY
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY n'est pas définie (fichier .env ou variable d'environnement)"
            )
        self.model = model or MISTRAL_MODEL
        self._client = Mistral(api_key=api_key)

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        payload = self._build_payload(request)
        try:
            # Client synchrone dans un thread : indépendant de la boucle asyncio courante
            resp = await asyncio.to_thread(self._client.chat.complete, **payload)
        except Exception as e:
            errors = _request_errors.get()
            if errors is not None:
                errors.append(f"Erreur LLM Mistral ({type(e).__name__}) : {e}")
            raise

        if not resp or not resp.choices:
            return LlmResponse(content=None, tool_calls=None, finish_reason=None)

        choice = resp.choices[0]
        message = choice.message
        tool_calls = self._extract_tool_calls(message)

        usage = None
        if getattr(resp, "usage", None):
            usage = {
                "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
            }

        finish_reason = getattr(choice, "finish_reason", None)
        return LlmResponse(
            content=self._content_to_text(getattr(message, "content", None)),
            tool_calls=tool_calls or None,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage,
        )

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        # Pas de streaming token par token : la réponse complète est émise en un bloc.
        response = await self.send_request(request)
        yield LlmStreamChunk(
            content=response.content,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason or "stop",
        )

    async def validate_tools(self, tools: List[ToolSchema]) -> List[str]:
        return [f"Nom d'outil invalide : {t.name!r}" for t in tools if not t.name]

    # -- helpers ---------------------------------------------------------
    def _build_payload(self, request: LlmRequest) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        tool_names: Dict[str, str] = {}  # id d'appel -> nom de l'outil
        for m in request.messages:
            msg: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.role == "assistant" and m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
                for tc in m.tool_calls:
                    tool_names[tc.id] = tc.name
            elif m.role == "tool":
                msg["tool_call_id"] = m.tool_call_id
                msg["name"] = tool_names.get(m.tool_call_id or "", "tool")
            messages.append(msg)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _content_to_text(content: Any) -> Optional[str]:
        """Le contenu Mistral est une chaîne ou une liste de blocs."""
        if content is None or isinstance(content, str):
            return content or None
        parts = []
        for chunk in content:
            text = getattr(chunk, "text", None)
            if text is None and isinstance(chunk, dict):
                text = chunk.get("text")
            if text:
                parts.append(text)
        return "".join(parts) or None

    @staticmethod
    def _extract_tool_calls(message: Any) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            raw_args = getattr(fn, "arguments", None) or {}
            if isinstance(raw_args, str):
                try:
                    loaded = json.loads(raw_args)
                    args = loaded if isinstance(loaded, dict) else {"args": loaded}
                except Exception:
                    args = {"_raw": raw_args}
            else:
                args = dict(raw_args)
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", None) or uuid.uuid4().hex[:9],
                    name=fn.name,
                    arguments=args,
                )
            )
        return tool_calls


# ---------------------------------------------------------------------------
# LLM : Albert API (compatible OpenAI)
# ---------------------------------------------------------------------------
class AlbertLlmService(OpenAILlmService):
    """Service LLM Vanna 2 pour Albert API, via le client OpenAI (API compatible).

    Le modèle retenu doit gérer l'appel d'outils (function calling).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        api_key = api_key or ALBERT_API_KEY
        if not api_key:
            raise ValueError(
                "ALBERT_API_KEY n'est pas définie (fichier .env ou variable d'environnement)"
            )
        base_url = (base_url or ALBERT_BASE_URL).rstrip("/")
        model = model or ALBERT_MODEL
        super().__init__(model=model, api_key=api_key, base_url=base_url)
        logger.info("LLM : Albert API, modèle %s", self.model)

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        payload = self._build_payload(request)
        try:
            # Client synchrone dans un thread, pour ne pas bloquer la boucle asyncio
            resp = await asyncio.to_thread(
                self._client.chat.completions.create, **payload, stream=False
            )
        except Exception as e:
            errors = _request_errors.get()
            if errors is not None:
                errors.append(f"Erreur LLM Albert ({type(e).__name__}) : {e}")
            raise

        if not resp.choices:
            return LlmResponse(content=None, tool_calls=None, finish_reason=None)

        choice = resp.choices[0]
        tool_calls = self._extract_tool_calls_from_message(choice.message)
        return LlmResponse(
            content=getattr(choice.message, "content", None) or None,
            tool_calls=tool_calls or None,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        # Pas de streaming token par token : la réponse complète est émise en un bloc.
        response = await self.send_request(request)
        yield LlmStreamChunk(
            content=response.content,
            tool_calls=response.tool_calls,
            finish_reason=response.finish_reason or "stop",
        )

    def _build_payload(self, request: LlmRequest) -> Dict[str, Any]:
        payload = super()._build_payload(request)
        payload["temperature"] = request.temperature
        return payload


def build_llm_service() -> LlmService:
    """Construit le service LLM selon LLM_PROVIDER (albert ou mistral)."""
    if LLM_PROVIDER == "albert":
        return AlbertLlmService()
    if LLM_PROVIDER == "mistral":
        return MistralLlmService()
    raise ValueError(f"LLM_PROVIDER inconnu : {LLM_PROVIDER!r} (valeurs possibles : albert, mistral)")


# ---------------------------------------------------------------------------
# Accès PostgreSQL (lecture seule)
# ---------------------------------------------------------------------------
class PostgresReadOnlyRunner(SqlRunner):
    """Exécute le SQL sur PostgreSQL dans une transaction en lecture seule."""

    def __init__(self, host=None, database=None, user=None, password=None, port=None):
        self.connection_params = {
            "host": host or PGHOST,
            "dbname": database or PGDATABASE,
            "user": user or PGUSER,
            "password": password or PGPASSWORD,
            "port": port or PGPORT or 5432,
        }

    def run_sql_sync(self, sql: str) -> pd.DataFrame:
        import psycopg2

        missing = [k for k in ("host", "dbname", "user") if not self.connection_params[k]]
        if missing:
            raise RuntimeError(
                "Connexion PostgreSQL non configurée : définir PGHOST, PGDATABASE, PGUSER, "
                "PGPASSWORD et PGPORT (variables d'environnement ou fichier .env)"
            )

        conn = psycopg2.connect(**self.connection_params)
        try:
            conn.set_session(readonly=True)
            with conn.cursor() as cursor:
                cursor.execute(sql)
                if cursor.description is None:
                    return pd.DataFrame()
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
            return pd.DataFrame.from_records(rows, columns=columns, coerce_float=True)
        finally:
            conn.close()

    async def run_sql(self, args: RunSqlToolArgs, context: ToolContext) -> pd.DataFrame:
        return await asyncio.to_thread(self.run_sql_sync, args.sql)


class CeremaRunSqlTool(Tool[RunSqlToolArgs]):
    """Outil `run_sql` : exécution en lecture seule + aide à l'auto-correction.

    Variante de `vanna.tools.RunSqlTool` qui :
    - accepte les requêtes `WITH ... SELECT` (CTE) et refuse toute écriture ;
    - plafonne le nombre de requêtes en échec par question (`MAX_SQL_ATTEMPTS`) ;
    - joint le SQL et le DataFrame complet au composant d'interface (`data`).
    """

    def __init__(self, sql_runner: SqlRunner, file_system: LocalFileSystem):
        self.sql_runner = sql_runner
        self.file_system = file_system

    @property
    def name(self) -> str:
        return RUN_SQL_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Exécute une requête SQL PostgreSQL en lecture seule (SELECT ou WITH ... SELECT) "
            "sur la base de données foncières et renvoie le résultat."
        )

    def get_args_schema(self) -> Type[RunSqlToolArgs]:
        return RunSqlToolArgs

    def _failure(self, sql: str, error: str, llm_message: str) -> ToolResult:
        return ToolResult(
            success=False,
            result_for_llm=llm_message,
            ui_component=UiComponent(
                rich_component=NotificationComponent(
                    level="error",
                    message=error,
                    data={"sql": sql, "error": error},
                ),
                simple_component=SimpleTextComponent(text=error),
            ),
            # C'est `error` que l'agent renvoie au LLM en cas d'échec
            error=llm_message,
            metadata={"error_type": "sql_error", "sql": sql},
        )

    async def execute(self, context: ToolContext, args: RunSqlToolArgs) -> ToolResult:
        sql = _clean_sql(args.sql)
        failures = context.metadata.get("sql_failures", 0)

        if failures >= MAX_SQL_ATTEMPTS:
            # Requête non exécutée (et non comptée comme tentative côté interface)
            error = (
                f"Nombre maximal de tentatives SQL atteint ({MAX_SQL_ATTEMPTS}). "
                "N'appelle plus run_sql : explique à l'utilisateur pourquoi "
                "la question n'a pas pu être traitée."
            )
            return ToolResult(
                success=False,
                result_for_llm=error,
                error=error,
                metadata={"error_type": "sql_attempts_exceeded", "sql": sql},
            )

        if not _is_read_only(sql):
            context.metadata["sql_failures"] = failures + 1
            error = "Requête refusée : seules les requêtes SELECT / WITH sont autorisées."
            return self._failure(sql, error, error + " Réécris la requête en lecture seule.")

        try:
            df = await self.sql_runner.run_sql(RunSqlToolArgs(sql=sql), context)
        except Exception as e:
            failures += 1
            context.metadata["sql_failures"] = failures
            error = str(e).strip()
            logger.info("Erreur SQL (échec %s/%s) : %s", failures, MAX_SQL_ATTEMPTS, error)
            llm_message = _repair_hint(error)
            if failures >= MAX_SQL_ATTEMPTS:
                llm_message = (
                    f"ERREUR SQL: {error}\nNombre maximal de tentatives atteint "
                    f"({MAX_SQL_ATTEMPTS}). N'appelle plus run_sql : explique à "
                    "l'utilisateur pourquoi la question n'a pas pu être traitée."
                )
            return self._failure(sql, error, llm_message)

        row_count = len(df)
        columns = [str(c) for c in df.columns]
        component_data = {"sql": sql, "df": df}

        if df.empty:
            result = "Requête exécutée avec succès. Aucune ligne renvoyée (ne pas appeler visualize_data)."
            component = DataFrameComponent(
                rows=[],
                columns=columns,
                title="Résultat de la requête",
                description="Aucune ligne renvoyée",
                data=component_data,
            )
            metadata: Dict[str, Any] = {"row_count": 0, "columns": columns, "sql": sql}
        else:
            filename = f"query_results_{uuid.uuid4().hex[:8]}.csv"
            csv_content = df.to_csv(index=False)
            await self.file_system.write_file(filename, csv_content, context, overwrite=True)

            preview = csv_content
            if len(preview) > MAX_LLM_PREVIEW_CHARS:
                preview = (
                    preview[:MAX_LLM_PREVIEW_CHARS]
                    + f"\n(Aperçu tronqué à {MAX_LLM_PREVIEW_CHARS} caractères.)"
                )
            result = (
                f"{row_count} ligne(s), {len(columns)} colonne(s).\n{preview}\n\n"
                f"Résultat enregistré dans le fichier : {filename}\n"
                f"Pour visualize_data, utiliser filename = {filename}"
            )
            component = DataFrameComponent(
                rows=df.head(MAX_COMPONENT_ROWS).to_dict("records"),
                columns=columns,
                row_count=row_count,
                column_count=len(columns),
                title="Résultat de la requête",
                description=f"{row_count} ligne(s), {len(columns)} colonne(s)",
                data=component_data,
            )
            metadata = {
                "row_count": row_count,
                "columns": columns,
                "sql": sql,
                "output_file": filename,
            }

        return ToolResult(
            success=True,
            result_for_llm=result,
            ui_component=UiComponent(
                rich_component=component,
                simple_component=SimpleTextComponent(text=result),
            ),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Prompt système et contexte (RAG)
# ---------------------------------------------------------------------------
class CeremaSystemPromptBuilder(SystemPromptBuilder):
    """Prompt système de l'assistant (en français, dialecte PostgreSQL)."""

    async def build_system_prompt(
        self, user: User, tools: List[ToolSchema]
    ) -> Optional[str]:
        tool_names = [t.name for t in tools]
        today = datetime.now().strftime("%Y-%m-%d")
        parts = [
            "Tu es DatafoncIA, l'assistant d'analyse de données foncières du Cerema. "
            f"Tu interroges une base PostgreSQL de données foncières. Date du jour : {today}.",
            "",
            "Règles :",
            "- Réponds toujours en français.",
            "- Pour toute question portant sur les données, écris une requête SQL PostgreSQL "
            f"et exécute-la avec l'outil {RUN_SQL_TOOL_NAME}. N'invente jamais de résultat.",
            "- Appuie-toi sur le contexte fourni plus bas (structure des tables, documentation "
            "des variables, exemples de requêtes validées). Utilise les noms exacts des tables "
            "et colonnes, en préfixant les tables par leur schéma.",
            "- En cas de doute sur la structure, tu peux interroger information_schema.columns.",
            "- Seules les requêtes en lecture (SELECT ou WITH ... SELECT) sont autorisées.",
            f"- Si {RUN_SQL_TOOL_NAME} renvoie une erreur, corrige la requête et réessaie "
            f"({MAX_SQL_ATTEMPTS} tentatives au maximum).",
        ]
        if "visualize_data" in tool_names:
            parts.append(
                "- Si le résultat contient plusieurs lignes et se prête à un graphique, appelle "
                "ensuite visualize_data avec le nom du fichier CSV indiqué par "
                f"{RUN_SQL_TOOL_NAME} et un titre en français. Ne trace pas de graphique pour "
                "une valeur unique ou un résultat vide."
            )
        parts.extend(
            [
                "- Le tableau de résultat et le graphique sont affichés à l'utilisateur en dehors "
                "de ta réponse : ne les recopie pas.",
                "- Termine par un court résumé en français qui répond à la question et commente "
                "les résultats.",
            ]
        )
        if tool_names:
            parts.append(f"\nOutils disponibles : {', '.join(tool_names)}")
        return "\n".join(parts)


class CeremaContextEnhancer(LlmContextEnhancer):
    """Ajoute au prompt système le contexte issu de l'entraînement.

    Équivalent du RAG de Vanna 0.x : documentation / structure des tables et
    exemples question → SQL les plus proches de la question posée.
    (L'enrichisseur par défaut de Vanna 2 applique un seuil de similarité de 0,7,
    trop strict pour cette base : on retient ici les k plus proches voisins.)
    """

    def __init__(
        self,
        agent_memory: ChromaAgentMemory,
        n_docs: int = N_CONTEXT_DOCS,
        n_examples: int = N_CONTEXT_EXAMPLES,
    ):
        self.agent_memory = agent_memory
        self.n_docs = n_docs
        self.n_examples = n_examples

    async def enhance_system_prompt(
        self, system_prompt: str, user_message: str, user: User
    ) -> str:
        context = ToolContext(
            user=user,
            conversation_id="context",
            request_id=str(uuid.uuid4()),
            agent_memory=self.agent_memory,
        )
        try:
            docs = await self.agent_memory.search_text_memories(
                query=user_message,
                context=context,
                limit=self.n_docs,
                similarity_threshold=0.0,
            )
            examples = await self.agent_memory.search_similar_usage(
                question=user_message,
                context=context,
                limit=self.n_examples,
                similarity_threshold=0.0,
                tool_name_filter=RUN_SQL_TOOL_NAME,
            )
        except Exception as e:
            logger.warning("Recherche de contexte impossible : %s", e)
            return system_prompt

        sections = []
        if docs:
            sections.append(
                "\n\n## Structure de la base et documentation\n\n"
                + "\n\n".join(f"- {' '.join(r.memory.content.split())}" for r in docs)
            )
        if examples:
            lines = []
            for r in examples:
                sql = r.memory.args.get("sql")
                if sql:
                    lines.append(f"Question : {r.memory.question}\nSQL :\n{sql}")
            if lines:
                sections.append(
                    "\n\n## Exemples de requêtes validées pour des questions proches\n\n"
                    + "\n\n".join(lines)
                )
        return system_prompt + "".join(sections)


class SessionUserResolver(UserResolver):
    """Un utilisateur par session applicative (identifiant passé dans les métadonnées)."""

    async def resolve_user(self, request_context: RequestContext) -> User:
        user_id = str(request_context.metadata.get("user_id") or "ddtia")
        return User(id=user_id, username=user_id, group_memberships=["user"])


# ---------------------------------------------------------------------------
# Résultat d'une question
# ---------------------------------------------------------------------------
@dataclass
class AgentAnswer:
    """Réponse de l'agent à une question."""

    question: str
    sql: Optional[str] = None  # dernière requête exécutée
    df: Optional[pd.DataFrame] = None  # résultat (None si aucune requête n'a abouti)
    error: Optional[str] = None  # dernière erreur si la question n'a pas abouti
    attempts: int = 0  # nombre de requêtes SQL exécutées
    sql_history: List[Tuple[str, Optional[str]]] = field(default_factory=list)  # (sql, erreur)
    figure: Optional[Any] = None  # plotly.graph_objects.Figure
    summary: Optional[str] = None  # réponse textuelle du LLM

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Façade : agent + mémoire + entraînement
# ---------------------------------------------------------------------------
class CeremaVanna:
    """Assistant text-to-SQL du Cerema bâti sur l'agent Vanna 2."""

    MAX_RETRIES = MAX_SQL_ATTEMPTS

    def __init__(self, chroma_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.agent_memory = ChromaAgentMemory(
            persist_directory=chroma_dir or CHROMA_DIR,
            collection_name=collection_name or CHROMA_COLLECTION,
        )
        self.sql_runner = PostgresReadOnlyRunner()
        self.file_system = LocalFileSystem(working_directory=WORK_DIR)
        self._agent: Optional[Agent] = None

    def log(self, message: str) -> None:
        logger.info(message)
        print(message)

    # -- Agent -------------------------------------------------------------
    @property
    def agent(self) -> Agent:
        """Agent Vanna, construit au premier appel (l'entraînement n'a pas besoin du LLM)."""
        if self._agent is None:
            tool_registry = ToolRegistry()
            tool_registry.register_local_tool(
                CeremaRunSqlTool(self.sql_runner, self.file_system), access_groups=[]
            )
            tool_registry.register_local_tool(
                VisualizeDataTool(file_system=self.file_system), access_groups=[]
            )

            # Les erreurs d'outil sont remontées à l'application (affichage / mode debug)
            ui_features = UiFeatures(
                feature_group_access={
                    **DEFAULT_UI_FEATURES,
                    UiFeature.UI_FEATURE_SHOW_TOOL_ERROR: [],
                }
            )

            self._agent = Agent(
                llm_service=build_llm_service(),
                tool_registry=tool_registry,
                user_resolver=SessionUserResolver(),
                agent_memory=self.agent_memory,
                config=AgentConfig(
                    stream_responses=False,
                    temperature=0.2,
                    # échecs SQL + requête réussie + graphique, avec un peu de marge
                    max_tool_iterations=MAX_SQL_ATTEMPTS + 5,
                    ui_features=ui_features,
                ),
                system_prompt_builder=CeremaSystemPromptBuilder(),
                llm_context_enhancer=CeremaContextEnhancer(self.agent_memory),
            )
        return self._agent

    async def ask_async(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: str = "ddtia",
    ) -> AgentAnswer:
        """Pose une question à l'agent et rassemble SQL, données, graphique et résumé."""
        import plotly.io as pio

        answer = AgentAnswer(question=question)
        texts: List[str] = []
        last_sql_error: Optional[str] = None
        fatal_error: Optional[str] = None
        errors: List[str] = []
        token = _request_errors.set(errors)
        try:
            request_context = RequestContext(metadata={"user_id": user_id})
            async for component in self.agent.send_message(
                request_context=request_context,
                message=question,
                conversation_id=conversation_id,
            ):
                rich = getattr(component, "rich_component", None)
                if isinstance(rich, ChartComponent):
                    try:
                        answer.figure = pio.from_json(json.dumps(rich.data, default=str))
                    except Exception as e:
                        logger.warning("Graphique illisible : %s", e)
                elif isinstance(rich, DataFrameComponent) and "sql" in rich.data:
                    answer.attempts += 1
                    answer.sql = rich.data["sql"]
                    answer.df = rich.data.get("df")
                    answer.sql_history.append((answer.sql, None))
                    last_sql_error = None
                elif isinstance(rich, NotificationComponent) and "sql" in rich.data:
                    answer.attempts += 1
                    last_sql_error = rich.data.get("error") or rich.message
                    answer.sql_history.append((rich.data["sql"], last_sql_error))
                    if answer.df is None:
                        answer.sql = rich.data["sql"]
                elif isinstance(rich, RichTextComponent):
                    if rich.content:
                        texts.append(rich.content)
                elif isinstance(rich, StatusCardComponent) and rich.status == "error":
                    fatal_error = errors[-1] if errors else (rich.description or rich.title)
        finally:
            _request_errors.reset(token)

        if texts:
            answer.summary = texts[-1]
        if fatal_error:
            answer.error = fatal_error
        elif answer.df is None and last_sql_error:
            answer.error = last_sql_error
        return answer

    def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: str = "ddtia",
    ) -> AgentAnswer:
        """Version synchrone de `ask_async` (Streamlit, notebook, scripts)."""
        return run_sync(self.ask_async(question, conversation_id, user_id))

    # -- Base de données -----------------------------------------------------
    def run_sql(self, sql: str) -> pd.DataFrame:
        """Exécute une requête en lecture seule et renvoie un DataFrame."""
        return self.sql_runner.run_sql_sync(sql)

    # -- Mémoire / entraînement ------------------------------------------------
    def _training_context(self) -> ToolContext:
        return ToolContext(
            user=User(id="training", username="training", group_memberships=["admin"]),
            conversation_id="training",
            request_id=str(uuid.uuid4()),
            agent_memory=self.agent_memory,
        )

    async def _save_documentations(self, documentations: List[str]) -> None:
        context = self._training_context()
        for documentation in documentations:
            await self.agent_memory.save_text_memory(content=documentation, context=context)

    async def _save_question_sql(self, pairs: List[Tuple[str, str]]) -> None:
        context = self._training_context()
        for question, sql in pairs:
            await self.agent_memory.save_tool_usage(
                question=question,
                tool_name=RUN_SQL_TOOL_NAME,
                args={"sql": sql},
                context=context,
                success=True,
            )

    def train(
        self,
        question: Optional[str] = None,
        sql: Optional[str] = None,
        documentation: Optional[str] = None,
        ddl: Optional[str] = None,
    ) -> None:
        """Ajoute un élément d'entraînement (même signature que `vn.train` de Vanna 0.x)."""
        if question and sql:
            run_sync(self._save_question_sql([(question, sql)]))
        elif sql:
            raise ValueError("Un SQL d'entraînement doit être accompagné de sa question")
        if documentation:
            run_sync(self._save_documentations([documentation]))
        if ddl:
            run_sync(self._save_documentations([ddl]))

    def clean(self) -> int:
        """Vide la mémoire d'entraînement. Renvoie le nombre d'éléments supprimés."""
        deleted = run_sync(self.agent_memory.clear_memories(context=self._training_context()))
        self.log(message=f"Mémoire vidée : {deleted} éléments supprimés")
        return deleted

    def get_training_data(self) -> pd.DataFrame:
        """Contenu de la mémoire d'entraînement sous forme de DataFrame."""
        results = self.agent_memory._get_collection().get()
        rows = []
        for memory_id, metadata in zip(results.get("ids") or [], results.get("metadatas") or []):
            metadata = metadata or {}
            if metadata.get("is_text_memory"):
                rows.append(
                    {
                        "id": memory_id,
                        "training_data_type": "documentation",
                        "question": None,
                        "content": metadata.get("content", ""),
                        "timestamp": metadata.get("timestamp"),
                    }
                )
            else:
                args = json.loads(metadata.get("args_json", "{}"))
                rows.append(
                    {
                        "id": memory_id,
                        "training_data_type": "sql",
                        "question": metadata.get("question"),
                        "content": args.get("sql", ""),
                        "timestamp": metadata.get("timestamp"),
                    }
                )
        return pd.DataFrame(
            rows, columns=["id", "training_data_type", "question", "content", "timestamp"]
        )

    def train_complete_dataset(self, schema, doc_tables=None, documentation=None, examples=None):
        """
        Méthode complète d'entraînement combinant toutes les sources :
        structure et commentaires PostgreSQL, documentation Datafoncier, puis la
        documentation métier et les exemples question/SQL fournis par l'utilisateur
        (voir training/train_template.py).
        """
        self.log(message=f"Début de l'entraînement complet du dataset pour {schema}")

        success_count = 0
        total_steps = 5

        # 1. Structure PostgreSQL
        self.log(message="Phase 1: Entraînement avec la structure PostgreSQL")
        try:
            self.train_with_pg_structure(schema)
            success_count += 1
        except Exception as e:
            self.log(message=f"Erreur phase 1 : {e}")

        # 2. Commentaires PostgreSQL
        self.log(message="Phase 2: Entraînement avec les commentaires PostgreSQL")
        try:
            self.train_with_pgtable_comments(schema)
            success_count += 1
        except Exception as e:
            self.log(message=f"Erreur phase 2 : {e}")

        # 3. Documentation API
        self.log(message="Phase 3: Entraînement avec la documentation Datafoncier")
        if self.train_with_documentation_datafoncier(tables=doc_tables, schema=schema):
            success_count += 1

        # 4. Documentation métier fournie par l'utilisateur
        self.log(message="Phase 4: Entraînement avec la documentation métier")
        try:
            self.train_with_documentation(documentation or [])
            success_count += 1
        except Exception as e:
            self.log(message=f"Erreur phase 4 : {e}")

        # 5. Exemples question/SQL fournis par l'utilisateur
        self.log(message="Phase 5: Entraînement avec les exemples question/SQL")
        try:
            if not self.train_with_examples(examples or []):
                self.log(message="Phase 5 : au moins un exemple a été rejeté")
            else:
                success_count += 1
        except Exception as e:
            self.log(message=f"Erreur phase 5 : {e}")

        self.log(message=f"Entraînement complet terminé : {success_count}/{total_steps} phases réussies")
        return success_count == total_steps

    def train_with_documentation(self, documentation):
        """Entraînement avec des textes de documentation métier (liste de chaînes)"""
        documentations = [" ".join(text.split()) for text in documentation if text and text.strip()]
        run_sync(self._save_documentations(documentations))
        self.log(message=f"Documentation métier : {len(documentations)} textes enregistrés")
        return True

    def validate_sql(self, sql):
        """Vérifie qu'une requête est valide sans l'exécuter (EXPLAIN).

        Returns:
            None si la requête est valide, sinon le message d'erreur
        """
        sql = _clean_sql(sql)
        if not _is_read_only(sql):
            return "seules les requêtes SELECT / WITH sont autorisées"
        try:
            self.run_sql(f"EXPLAIN {sql}")
            return None
        except Exception as e:
            return str(e).strip()

    def train_with_examples(self, examples, validate=True):
        """
        Entraînement avec des exemples question/SQL.

        Args:
            examples: liste de dictionnaires {"question": ..., "sql": ...}
            validate: vérifie chaque requête sur la base (EXPLAIN) et rejette les
                requêtes invalides, pour ne pas apprendre d'exemple faux

        Returns:
            True si tous les exemples ont été enregistrés
        """
        pairs = []
        rejected = 0
        for i, example in enumerate(examples, start=1):
            question = (example.get("question") or "").strip()
            sql = _clean_sql(example.get("sql") or "")
            if not question or not sql:
                self.log(message=f"Exemple {i} rejeté : question ou SQL manquant")
                rejected += 1
                continue
            if validate:
                error = self.validate_sql(sql)
                if error:
                    self.log(message=f"Exemple {i} rejeté ({question[:50]}...) : {error}")
                    rejected += 1
                    continue
            self.log(message=f"Entraînement avec question : {question[:50]}...")
            pairs.append((question, sql))

        run_sync(self._save_question_sql(pairs))
        self.log(message=f"Exemples question/SQL : {len(pairs)} enregistrés, {rejected} rejetés")
        return rejected == 0

    def train_with_pg_structure(self, schema):
        """Entrainement à partir de la structure du schéma PostgreSQL (une fiche par table)"""
        df_columns = self.run_sql(f"""
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
            ORDER BY table_name, ordinal_position
            """)

        documentations = []
        for (table_schema, table_name), columns in df_columns.groupby(
            ["table_schema", "table_name"], sort=False
        ):
            column_list = ", ".join(
                f"{row.column_name} ({row.data_type})" for row in columns.itertuples()
            )
            documentations.append(
                f"Structure de la table {table_schema}.{table_name} "
                f"(base PostgreSQL) - colonnes : {column_list}"
            )
            self.log(message=f"Entrainement : structure de la table {table_schema}.{table_name}")

        run_sync(self._save_documentations(documentations))

    def train_with_pgtable_comments(self, schema):
        """Entrainement à partir des commentaires de colonnes des tables du schéma PostgreSQL"""
        comments = self.run_sql(f"""
            SELECT
            col_description(
                (table_schema||'.'||table_name)::regclass::oid, ordinal_position
            ) as column_comment,
            *
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
        """)

        documentations = []
        for i, row in comments.iterrows():
            if row["column_comment"]:
                self.log(message=f"Entrainement : commentaire variable {row['column_name']} ({row['table_schema']})")
                documentations.append(
                    f"La variable {row['column_name']} de la table "
                    f"{row['table_schema']}.{row['table_name']} "
                    f"correspond à {row['column_comment']}"
                )

        run_sync(self._save_documentations(documentations))

    def train_with_documentation_datafoncier(self, tables=None, schema=None):
        """
        Entraîne la vector database avec la documentation des variables
        depuis l'API de documentation
        """
        documentations = []
        for table in tables or []:
            params = {"table": table}
            variables_data = self._call_api(DOC_DATAFONCIER_TRAINING_URL, params=params)

            if variables_data is None:
                self.log(message=f"Échec de récupération documentation pour la table : {table}")
                continue

            self.log(message=f"Documentation récupérée pour {table} : {len(variables_data)} variables")

            for variable in variables_data:
                variable_name = variable.get("nom")
                description = variable.get("description")
                data_type = variable.get("type")
                longueur = variable.get("longueur")
                table_info = variable.get("table")
                table_name = table_info.get("nom", table) if isinstance(table_info, dict) else table

                if variable_name and description:
                    documentation = (
                        f"La variable {variable_name} de la table "
                        f"{schema + '.' if schema else ''}{table_name} "
                        f"correspond à : {description}"
                    )

                    if data_type:
                        documentation += f"\nType de données : {data_type}"
                        if longueur == 999:  # type array
                            documentation += "[]"

                    self.log(message=f"Entraînement documentation : {variable_name}")
                    documentations.append(documentation)

            self.log(message=f"Documentation terminée pour la table {table}")

        run_sync(self._save_documentations(documentations))

        self.log(message=f"Documentation Datafoncier terminée : {len(documentations)} variables traitées")
        return True

    def _call_api(self, url, method="GET", headers=None, params=None, data=None, timeout=30):
        """
        Méthode générique pour interroger une API

        Returns:
            dict: Réponse JSON de l'API ou None en cas d'erreur
        """
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "CeremaVanna/1.0"
        }

        if headers:
            default_headers.update(headers)

        try:
            self.log(message=f"Appel API {method} : {url}")

            response = requests.request(
                method=method,
                url=url,
                headers=default_headers,
                params=params,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()

            try:
                json_response = response.json()
                self.log(message=f"API {method} {url} : succès ({response.status_code})")
                return json_response
            except ValueError:
                self.log(message=f"API {method} {url} : réponse non-JSON reçue")
                return {"text": response.text}
        except requests.exceptions.Timeout:
            self.log(message=f"Timeout lors de l'appel API {url} après {timeout}s")
            return None
        except requests.exceptions.ConnectionError:
            self.log(message=f"Erreur de connexion lors de l'appel API {url}")
            return None
        except requests.exceptions.HTTPError as e:
            self.log(message=f"Erreur HTTP {e.response.status_code} lors de l'appel API {url}")
            return None
        except Exception as e:
            self.log(message=f"Erreur inattendue lors de l'appel API {url} : {e}")
            return None
