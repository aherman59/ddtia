import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo

    # Le code de l'assistant (agent Vanna 2) est dans app/core.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
    from core import CeremaVanna

    cerema = CeremaVanna()

    def show(answer):
        """Affiche une réponse de l'agent : SQL, données, graphique, résumé"""
        items = [mo.md(f"**Question** : {answer.question}")]
        for sql, error in answer.sql_history:
            items.append(mo.md(f"```sql\n{sql}\n```"))
            if error:
                items.append(mo.md(f"❌ `{error}`"))
        if answer.df is not None:
            items.append(mo.ui.table(answer.df))
        if answer.figure is not None:
            items.append(mo.ui.plotly(answer.figure))
        if answer.error:
            items.append(mo.md(f"**Erreur** : {answer.error}"))
        if answer.summary:
            items.append(mo.md(answer.summary))
        return mo.vstack(items)

    return cerema, show


@app.cell
def _(cerema):
    # Contenu de la mémoire (l'entraînement se fait avec un script créé à partir
    # de training/train_template.py, voir le README)
    cerema.get_training_data()
    return


@app.cell
def _(cerema, show):
    show(cerema.ask("Quels sont les 20 premières villes où les prix de maisons sont les plus élevés (au moins 11 mutations)?"))
    return


@app.cell
def _(cerema, show):
    show(cerema.ask("Peux tu me donner le nombre de mutations par année qui ont eu lieu sur la parcelle 62263000AH0264 ?"))
    return


@app.cell
def _(cerema, show):
    show(cerema.ask("Donne moi le nombre de locaux ayant muté pour chacune des 3 communes :Arras, Béthune et Calais via la table mutation."))
    return


if __name__ == "__main__":
    app.run()
