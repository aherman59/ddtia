from pathlib import Path

import streamlit as st

from core import MAX_SQL_ATTEMPTS, AgentAnswer
from vanna_calls import ask_question, new_conversation_id

# Configuration
AVATAR_URL = Path(__file__).parent / "assets" / "LogoCerema_vertical.jpg"
st.set_page_config(layout="wide")

NO_QUESTION = "Sélectionner une question..."

# Questions suggérées
SAMPLE_QUESTIONS = [
    "Combien y a-t-il de mutations de maisons et quel est le prix moyen sur Arras, par année ?",
    "Nombre de mutations annuellement dans le département du Pas-de-Calais (62)",
    "Nombre de locaux ayant muté sur le département du Pas-de-Calais (62)",
    "Répartition des locaux selon leur type sur Arras",
    "Nombre d'appartements T3 ayant muté sur Arras",
    "Montant total des transactions par année dans le Pas-de-Calais (62)",
    "Prix médian au m² des appartements sur Arras en 2024",
    "Prix médian des maisons vendues sur Arras",
    "Surface moyenne du terrain des maisons dans le Pas-de-Calais (62)",
    "Ventes de logements de particulier à particulier sur Arras en 2024",
    "Nombre de mutations par trimestre sur le Pas-de-Calais (62)",
    "Evolution du prix au m² des appartements sur Arras depuis 2020",
    "Nombre de terrains à bâtir vendus dans le Pas-de-Calais (62)",
    "Top 10 des communes du Pas-de-Calais par volume de transactions",
    "Mutations concernant exclusivement la commune d'Arras (code INSEE: 62041)"
]


def init_session_state():
    """Initialise l'historique et les identifiants de session / conversation"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = new_conversation_id()
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = new_conversation_id()


def clear_conversation():
    """Efface l'historique et démarre une nouvelle conversation avec l'agent"""
    st.session_state["messages"] = []
    st.session_state["conversation_id"] = new_conversation_id()
    st.session_state.pop("pending_question", None)


def setup_sidebar():
    """Configuration de la barre latérale avec les paramètres d'affichage"""

    if st.sidebar.button("🔄 Nouvelle conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    st.sidebar.divider()

    # Suggestions dans la sidebar
    st.sidebar.subheader("💡 Questions suggérées")
    selected_question = st.sidebar.selectbox(
        "Choisir une question",
        options=[NO_QUESTION] + SAMPLE_QUESTIONS,
        key="suggested_question_select"
    )

    # Bouton de validation
    if st.sidebar.button(
        "▶️ Exécuter cette question",
        use_container_width=True,
        disabled=(selected_question == NO_QUESTION),
        type="primary" if selected_question != NO_QUESTION else "secondary"
    ):
        st.session_state["pending_question"] = selected_question
        st.rerun()

    st.sidebar.divider()

    st.sidebar.title("⚙️ Paramètres d'affichage")

    st.sidebar.checkbox("🔍 Montrer le SQL", value=True, key="show_sql")
    st.sidebar.checkbox("📊 Montrer la table de résultat", value=True, key="show_table")
    st.sidebar.checkbox("📈 Montrer le graphique", value=True, key="show_chart")
    st.sidebar.checkbox("📝 Montrer le Résumé", value=True, key="show_summary")

    # Mode debug
    st.sidebar.checkbox("🐞 Mode debug (détails techniques)", value=False, key="debug_mode")


def display_sql_info(answer: AgentAnswer):
    """Affiche la requête SQL avec le nombre de tentatives"""
    if not answer.sql or not st.session_state.get("show_sql", True):
        return

    assistant_message_sql = st.chat_message("assistant", avatar=AVATAR_URL)

    if answer.df is None:
        assistant_message_sql.subheader("🔍 Dernière requête SQL tentée")
        assistant_message_sql.warning(f"⚠️ Cette requête a échoué après {answer.attempts} tentative(s)")
    else:
        assistant_message_sql.subheader("🔍 Requête SQL générée")
        if answer.attempts > 1:
            assistant_message_sql.info(
                f"✅ Requête corrigée automatiquement (tentative {answer.attempts}/{MAX_SQL_ATTEMPTS})"
            )

    assistant_message_sql.code(answer.sql, language="sql", line_numbers=True)

    # Détail des tentatives en mode debug
    if st.session_state.get("debug_mode") and len(answer.sql_history) > 1:
        with assistant_message_sql.expander("🐞 Détail des tentatives"):
            for i, (sql, error) in enumerate(answer.sql_history, start=1):
                st.markdown(f"**Tentative {i}** — {'❌ ' + error if error else '✅ succès'}")
                st.code(sql, language="sql")


def display_data_table(df):
    """Affiche le tableau de données"""
    if st.session_state.get("show_table", True):
        assistant_message_table = st.chat_message("assistant", avatar=AVATAR_URL)
        assistant_message_table.subheader("📊 Données résultantes")

        if len(df) > 10:
            assistant_message_table.info(f"📋 Affichage des 10 premières lignes sur {len(df)} résultats")
            assistant_message_table.dataframe(df.head(10))
        else:
            assistant_message_table.dataframe(df)


def display_chart(answer: AgentAnswer, key: str):
    """Affiche le graphique produit par l'agent, s'il y en a un"""
    if answer.figure is None or not st.session_state.get("show_chart", True):
        return

    assistant_message_chart = st.chat_message("assistant", avatar=AVATAR_URL)
    assistant_message_chart.subheader("📈 Visualisation")
    assistant_message_chart.plotly_chart(answer.figure, key=f"chart_{key}")


def display_summary(answer: AgentAnswer):
    """Affiche le résumé des résultats"""
    if not answer.summary or not st.session_state.get("show_summary", True):
        return

    assistant_message_summary = st.chat_message("assistant", avatar=AVATAR_URL)
    assistant_message_summary.subheader("📝 Résumé")
    assistant_message_summary.write(answer.summary)


def display_error_with_retry(answer: AgentAnswer, key: str):
    """Affiche l'erreur avec option de nouvelle tentative"""
    assistant_message_error = st.chat_message("assistant", avatar=AVATAR_URL)
    if answer.attempts:
        assistant_message_error.error(f"❌ Erreur après {answer.attempts} tentative(s) de correction automatique")
    else:
        assistant_message_error.error("❌ La question n'a pas pu être traitée")

    if st.session_state.get("debug_mode"):
        assistant_message_error.code(answer.error or "Erreur inconnue", language="text")
    else:
        assistant_message_error.info("👉 Essayez de reformuler votre question ou activez le mode debug")

    # Explication éventuelle fournie par l'assistant
    if answer.summary:
        assistant_message_error.write(answer.summary)

    if assistant_message_error.button("🔄 Réessayer", key=f"retry_{key}"):
        st.session_state["pending_question"] = answer.question
        st.rerun()


def display_answer(answer: AgentAnswer, key: str):
    """Affiche la réponse de l'agent (résultat, SQL, graphique, résumé)"""
    # Réponse purement textuelle (aucune requête SQL exécutée)
    if answer.success and answer.df is None:
        assistant_message = st.chat_message("assistant", avatar=AVATAR_URL)
        assistant_message.write(answer.summary or "⚠️ Aucune réponse n'a été produite")
        return

    # Création des onglets
    resultat_tab, sql_tab = st.tabs(["📊 Résultat", "🔍 SQL"])

    with sql_tab:
        display_sql_info(answer)

    with resultat_tab:
        # Gestion des erreurs
        if not answer.success or answer.df is None:
            display_error_with_retry(answer, key)
            return

        if answer.df.empty:
            assistant_message_empty = st.chat_message("assistant", avatar=AVATAR_URL)
            assistant_message_empty.warning("⚠️ Aucun résultat trouvé pour cette requête")
            if answer.attempts > 1:
                assistant_message_empty.info(f"La requête a été corrigée {answer.attempts - 1} fois automatiquement")
            display_summary(answer)
            return

        # Indicateur de succès avec corrections
        if answer.attempts > 1:
            success_msg = st.chat_message("assistant", avatar=AVATAR_URL)
            success_msg.success(
                f"✅ Requête exécutée avec succès après {answer.attempts - 1} correction(s) automatique(s)"
            )

        display_data_table(answer.df)
        display_chart(answer, key)
        display_summary(answer)


def process_question(my_question):
    """Traite la question utilisateur et affiche les résultats avec auto-réparation"""
    st.session_state["messages"].append({"role": "user", "content": my_question})
    st.chat_message("user").write(my_question)

    with st.spinner("⏳ Génération et exécution SQL (avec auto-correction si nécessaire)..."):
        answer = ask_question(
            question=my_question,
            conversation_id=st.session_state["conversation_id"],
            user_id=st.session_state["user_id"],
        )

    st.session_state["messages"].append({"role": "assistant", "answer": answer})
    display_answer(answer, key=str(len(st.session_state["messages"]) - 1))


def main():
    """Fonction principale de l'application"""
    st.title("🤖 DatafoncIA - l'Assistant d'analyse de données foncières")
    st.caption(f"✨ Avec auto-correction intelligente des erreurs SQL (max {MAX_SQL_ATTEMPTS} tentatives)")

    init_session_state()

    # Configuration de la barre latérale
    setup_sidebar()

    # Affichage de l'historique des conversations
    for i, msg in enumerate(st.session_state["messages"]):
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            display_answer(msg["answer"], key=str(i))

    # Zone d'input toujours disponible
    prompt = st.chat_input("💬 Pose-moi une question pour analyser tes données")

    # Gestion de la question à traiter
    question_to_process = None

    # Priorité à la question en attente (depuis menu déroulant ou bouton Réessayer)
    if st.session_state.get("pending_question"):
        question_to_process = st.session_state.pop("pending_question")
    # Sinon, question depuis le chat_input
    elif prompt:
        question_to_process = prompt

    # Traitement de la question si elle existe
    if question_to_process:
        process_question(question_to_process)


if __name__ == "__main__":
    main()
