import uuid

import streamlit as st

from core import AgentAnswer, CeremaVanna


@st.cache_resource
def setup_vanna() -> CeremaVanna:
    """Instance partagée de l'assistant (agent Vanna 2, mémoire Chroma, accès PostgreSQL)."""
    return CeremaVanna()


def new_conversation_id() -> str:
    """Identifiant d'une nouvelle conversation (l'agent garde le fil par conversation)."""
    return str(uuid.uuid4())


def ask_question(question: str, conversation_id: str, user_id: str) -> AgentAnswer:
    """
    Pose une question à l'agent : génération du SQL, exécution avec auto-correction,
    graphique et résumé.

    Returns:
        AgentAnswer (sql, df, error, attempts, figure, summary)
    """
    vn = setup_vanna()
    return vn.ask(question, conversation_id=conversation_id, user_id=user_id)
