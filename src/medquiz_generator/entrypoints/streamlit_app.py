"""Streamlit review app: exam simulator (client preview) + human review with persistence."""

from __future__ import annotations

import html
import re
from pathlib import Path

import streamlit as st

from medquiz_generator.application import artifacts
from medquiz_generator.application.context import RunContext
from medquiz_generator.composition.container import Container
from medquiz_generator.domain.errors import MedQuizError
from medquiz_generator.domain.models import (
    EvalReport,
    FigureRef,
    QuestionStatus,
    QuizItem,
    RunReport,
    StructuredDocument,
    Taxonomy,
)
from medquiz_generator.domain.text import Paragraph, paragraphs_around
from medquiz_generator.entrypoints.logsetup import configure_logging
from medquiz_generator.entrypoints.presentation import render_eval, render_report, render_taxonomy

st.set_page_config(page_title="MedQuiz Generator", page_icon="🩺", layout="wide")

STATUS_LABELS = {
    QuestionStatus.VERIFIED: "✅ verificato",
    QuestionStatus.FLAGGED: "⚠️ da rivedere",
    QuestionStatus.DISCARDED: "❌ scartato",
    QuestionStatus.GENERATED: "⏳ generato",
}


@st.cache_resource
def get_container() -> Container:
    container = Container()
    configure_logging(container.settings.log_level, json_logs=container.settings.log_json)
    return container


def load_items(container: Container, run_id: str) -> list[QuizItem]:
    return container.review.items(run_id)


def load_document(container: Container, run_id: str) -> StructuredDocument | None:
    if container.store.exists(run_id, artifacts.DOCUMENT):
        return container.store.load(run_id, artifacts.DOCUMENT, StructuredDocument)
    return None


def option_label(item: QuizItem, option_id: str) -> str:
    text = next((o.text for o in item.options if o.id == option_id), "")
    return f"{option_id}. {text}"


_CAPTION = re.compile(r"^(figura|fig\.|tabella|tab\.)\s*\d", re.IGNORECASE)


def highlighted_html(paragraph: Paragraph) -> str:
    """One paragraph as HTML, with the cited span wrapped in <mark>."""
    if paragraph.quote_span is None:
        return html.escape(paragraph.text)
    start, end = paragraph.quote_span
    return (
        html.escape(paragraph.text[:start])
        + "<mark>"
        + html.escape(paragraph.text[start:end])
        + "</mark>"
        + html.escape(paragraph.text[end:])
    )


def figure_for_caption(
    document: StructuredDocument, section_id: str, caption: str
) -> FigureRef | None:
    """The extracted figure whose caption opens this paragraph, if any."""
    for figure in document.figures:
        if (
            figure.section_id == section_id
            and figure.caption
            and caption.startswith(figure.caption)
        ):
            return figure
    return None


def source_block(item: QuizItem, document: StructuredDocument | None) -> None:
    """The cited passage, paragraph by paragraph: captions set apart (with the figure
    itself when it was extracted), tables rendered as tables, the quote highlighted."""
    st.markdown(
        f"**Fonte:** {item.source.section_path}  \n"
        f"**Pagine sezione:** {item.source.page_start}-{item.source.page_end}"
        + (f" · **pagina citata:** {item.source.page}" if item.source.page else "")
    )
    section = document.section_by_id(item.source.section_id) if document else None
    if document is None or section is None:
        st.markdown(f"> {item.source.quote}")
        return
    paragraphs, before, after = paragraphs_around(section.text, item.source.quote)
    run_dir = get_container().store.run_dir(item.model_metadata.run_id)
    with st.container(border=True):
        if before:
            st.caption("…")
        for paragraph in paragraphs:
            if paragraph.text.lstrip().startswith("|"):
                st.markdown(paragraph.text)
                continue
            if _CAPTION.match(paragraph.text) and paragraph.quote_span is None:
                figure = figure_for_caption(document, section.id, paragraph.text)
                icon = "🖼️" if paragraph.text.lower().startswith("fig") else "📋"
                st.markdown(
                    f"<div style='color:#6b7280;font-style:italic;font-size:0.9em'>"
                    f"{icon} {html.escape(paragraph.text)}</div>",
                    unsafe_allow_html=True,
                )
                if figure is not None and (run_dir / figure.path).exists():
                    st.image(str(run_dir / figure.path), width=320)
                continue
            style = "background:#f5fbfa;border-left:4px solid #0F766E;padding:0.4rem 0.8rem"
            if paragraph.quote_span is None:
                style = "padding:0.2rem 0.8rem"
            st.markdown(
                f"<div style='{style};white-space:pre-wrap'>{highlighted_html(paragraph)}</div>",
                unsafe_allow_html=True,
            )
        if after:
            st.caption("…")


# ------------------------------------------------------------------- sidebar


def sidebar(container: Container) -> str | None:
    st.sidebar.title("🩺 MedQuiz Generator")
    runs = container.store.list_runs()
    latest = container.store.latest_run_id()
    run_id: str | None = None
    if runs:
        run_id = st.sidebar.selectbox(
            "Run",
            options=list(reversed(runs)),
            index=list(reversed(runs)).index(latest or runs[-1]),
        )
    else:
        st.sidebar.info("Nessun run: avvia la pipeline qui sotto.")

    with st.sidebar.expander("Nuovo run (pipeline completa)", expanded=not runs):
        uploaded = st.file_uploader("PDF nativo", type=["pdf"])
        pdf_path_text = st.text_input("…oppure percorso PDF", value="")
        total = st.number_input(
            "Numero totale di quesiti (obbligatorio)", min_value=1, value=None, step=1
        )
        include_flagged = st.checkbox(
            "Includi i quesiti flaggati nell'export",
            value=container.settings.export_include_flagged,
        )
        if st.button("Avvia pipeline", type="primary"):
            if total is None:
                st.error("Indica il numero totale di quesiti.")
            else:
                pdf_path: Path | None = None
                if uploaded is not None:
                    uploads = Path(".uploads")
                    uploads.mkdir(exist_ok=True)
                    pdf_path = uploads / uploaded.name
                    pdf_path.write_bytes(uploaded.getvalue())
                elif pdf_path_text.strip():
                    pdf_path = Path(pdf_path_text.strip())
                if pdf_path is None or not pdf_path.exists():
                    st.error("Carica un PDF o indica un percorso valido.")
                else:
                    new_id = container.new_run_id()
                    ctx = container.context(new_id)
                    with st.spinner(f"Pipeline in esecuzione (run {new_id})…"):
                        try:
                            report = container.pipeline(ctx).run(
                                ctx, pdf_path, int(total), include_flagged=include_flagged
                            )
                        except MedQuizError as exc:
                            st.error(str(exc))
                        else:
                            st.success(
                                f"Run {new_id}: {report.verified} verificati, "
                                f"{report.flagged} flaggati, {report.discarded} scartati "
                                f"({report.usage_total.cost_usd:.4f} USD)"
                            )
                            st.rerun()
    return run_id


# ----------------------------------------------------------------- simulator


def simulator_tab(items: list[QuizItem], document: StructuredDocument | None) -> None:
    deliverable = [i for i in items if i.status.status == QuestionStatus.VERIFIED]
    flagged = [i for i in items if i.status.status == QuestionStatus.FLAGGED]
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col3:
        show_flagged = st.toggle("Includi flaggati", value=False)
    pool = deliverable + (flagged if show_flagged else [])
    materie = sorted({i.materia for i in pool if i.materia})
    with col1:
        materia = st.selectbox("Materia", ["Tutte", *materie])
    if materia != "Tutte":
        pool = [i for i in pool if i.materia == materia]
    argomenti = sorted({i.argomento for i in pool if i.argomento})
    with col2:
        argomento = st.selectbox("Argomento", ["Tutti", *argomenti])
    if argomento != "Tutti":
        pool = [i for i in pool if i.argomento == argomento]
    with col4:
        mode = st.radio("Modalità", ["Uno alla volta", "Elenco"], horizontal=True)
    if not pool:
        st.info("Nessun quesito disponibile con questi filtri.")
        return

    answers: dict[str, str] = st.session_state.setdefault("answers", {})
    if mode == "Uno alla volta":
        index = st.session_state.get("sim_index", 0) % len(pool)
        item = pool[index]
        st.caption(f"Quesito {index + 1} di {len(pool)} · {item.materia} › {item.argomento}")
        render_question(item, answers, document, key=f"one-{item.id}")
        prev_col, next_col, score_col = st.columns([1, 1, 4])
        if prev_col.button("◀ Precedente"):
            st.session_state["sim_index"] = (index - 1) % len(pool)
            st.rerun()
        if next_col.button("Successivo ▶"):
            st.session_state["sim_index"] = (index + 1) % len(pool)
            st.rerun()
        answered = [i for i in pool if i.id in answers]
        correct = sum(1 for i in answered if answers[i.id] == i.correct_option_id)
        score_col.metric("Punteggio", f"{correct}/{len(answered)}")
    else:
        for number, item in enumerate(pool, start=1):
            with st.expander(f"{number}. {item.question[:110]}…", expanded=False):
                render_question(item, answers, document, key=f"list-{item.id}")


def figure_block(item: QuizItem) -> None:
    if item.image is None:
        return
    container = get_container()
    path = container.store.run_dir(item.model_metadata.run_id) / item.image.path
    if path.exists():
        st.image(str(path), caption=item.image.caption or item.image.figure_id)
    else:
        st.warning(f"Immagine non trovata: {item.image.path}")


def render_question(
    item: QuizItem, answers: dict[str, str], document: StructuredDocument | None, *, key: str
) -> None:
    figure_block(item)
    st.markdown(f"**{item.question}**")
    choice = st.radio(
        "Risposta",
        options=[o.id for o in item.options],
        format_func=lambda oid: option_label(item, oid),
        index=None,
        key=f"radio-{key}",
        label_visibility="collapsed",
    )
    if st.button("Conferma", key=f"confirm-{key}", disabled=choice is None):
        answers[item.id] = choice or ""
    if item.id in answers:
        given = answers[item.id]
        if given == item.correct_option_id:
            st.success(f"Corretto! Risposta {item.correct_option_id}.")
        else:
            st.error(
                f"Errato: hai risposto {given}, la risposta corretta è {item.correct_option_id}."
            )
        st.markdown(f"**Spiegazione.** {item.explanation}")
        with st.expander("Riferimento alla fonte"):
            source_block(item, document)
        st.caption(f"Difficoltà stimata: {item.difficulty.value}")


# -------------------------------------------------------------------- review


def review_tab(
    container: Container, run_id: str, items: list[QuizItem], document: StructuredDocument | None
) -> None:
    review = container.review
    state = review.state(run_id)
    statuses = st.multiselect(
        "Stato",
        options=[s.value for s in QuestionStatus],
        default=[QuestionStatus.VERIFIED.value, QuestionStatus.FLAGGED.value],
    )
    materie = sorted({i.materia for i in items if i.materia})
    materia = st.selectbox("Materia", ["Tutte", *materie], key="rev-materia")
    filtered = [i for i in items if i.status.status.value in statuses]
    if materia != "Tutte":
        filtered = [i for i in filtered if i.materia == materia]
    rows = [
        {
            "id": i.id,
            "stato": STATUS_LABELS[i.status.status],
            "materia": i.materia,
            "argomento": i.argomento,
            "difficoltà": i.difficulty.value,
            "domanda": i.question[:90],
            "motivo": i.status.reason[:60],
            "decisione": (d.decision if (d := state.decisions.get(i.id)) else ""),
        }
        for i in filtered
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    if not filtered:
        return
    selected_id = st.selectbox("Quesito", [i.id for i in filtered])
    item = next(i for i in filtered if i.id == selected_id)
    left, right = st.columns(2)
    with left:
        figure_block(item)
        st.markdown(f"**{item.question}**")
        for option in item.options:
            marker = "✅" if option.id == item.correct_option_id else "▫️"
            st.markdown(f"{marker} **{option.id}.** {option.text}")
        st.markdown(f"**Spiegazione.** {item.explanation}")
        st.caption(
            f"{STATUS_LABELS[item.status.status]} · {item.status.reason}"
            + (f" · judge: {item.status.judge_reason}" if item.status.judge_reason else "")
        )
    with right:
        source_block(item, document)

    st.divider()
    note = st.text_input("Nota del revisore", key=f"note-{item.id}")
    decision = state.decisions.get(item.id)
    col_a, col_b, col_c, col_d = st.columns(4)
    if col_a.button("✅ Approva", key=f"approve-{item.id}"):
        review.decide(run_id, item.id, "approved", note)
        st.rerun()
    if col_b.button("❌ Boccia", key=f"reject-{item.id}"):
        review.decide(run_id, item.id, "rejected", note)
        st.rerun()
    if col_c.button("↺ Annulla decisione", key=f"clear-{item.id}", disabled=decision is None):
        review.clear(run_id, item.id)
        st.rerun()
    can_promote = (
        decision is not None
        and decision.decision == "approved"
        and item.status.status in (QuestionStatus.VERIFIED, QuestionStatus.FLAGGED)
    )
    if col_d.button(
        "⭐ Promuovi al golden set", key=f"promote-{item.id}", disabled=not can_promote
    ):
        path = review.promote(run_id, item.id)
        st.success(f"Promosso: {path}")
    if decision is not None:
        st.info(f"Decisione: **{decision.decision}** — {decision.note or 'nessuna nota'}")

    st.divider()
    approved = review.approved_items(run_id)
    st.markdown(f"**Approvati:** {len(approved)} / {len(items)}")
    if st.button("Esporta solo gli approvati (JSON + XLSX)", disabled=not approved):
        ctx: RunContext = container.context(run_id)
        written = container.export_stage().run(ctx, approved_only=True)
        for fmt, path in written.items():
            st.download_button(
                f"Scarica {fmt}", data=path.read_bytes(), file_name=path.name, key=f"dl-{fmt}"
            )


# -------------------------------------------------------------------- report


def report_tab(container: Container, run_id: str) -> None:
    ctx = container.context(run_id)
    report: RunReport = container.report_stage().run(ctx)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Verificati", report.verified)
    m2.metric("Flaggati", report.flagged)
    m3.metric("Scartati", report.discarded)
    m4.metric("Costo totale (USD)", f"{report.usage_total.cost_usd:.4f}")
    m5.metric(
        "Costo / quesito consegnato",
        f"{report.cost_per_delivered_question_usd:.4f}"
        if report.cost_per_delivered_question_usd is not None
        else "n/d",
    )
    st.code(render_report(report), language="text")
    if container.store.exists(run_id, artifacts.EVAL_REPORT):
        eval_report = container.store.load(run_id, artifacts.EVAL_REPORT, EvalReport)
        st.subheader("Eval (golden set)")
        st.code(render_eval(eval_report), language="text")
    exported = ctx.meta().exported_files
    if exported:
        st.subheader("Export")
        for fmt, path_str in exported.items():
            path = Path(path_str)
            if path.exists():
                st.download_button(
                    f"Scarica {fmt}", data=path.read_bytes(), file_name=path.name, key=f"rep-{fmt}"
                )


def taxonomy_tab(container: Container, run_id: str) -> None:
    if not container.store.exists(run_id, artifacts.TAXONOMY):
        st.info("Tassonomia non ancora estratta per questo run.")
        return
    taxonomy = container.store.load(run_id, artifacts.TAXONOMY, Taxonomy)
    st.caption(
        f"Estratta dal testo ({taxonomy.source}) — {len(taxonomy.subjects)} materie, "
        f"{taxonomy.section_count} sezioni mappate"
    )
    st.code(render_taxonomy(taxonomy), language="text")


# ---------------------------------------------------------------------- main


def main() -> None:
    container = get_container()
    run_id = sidebar(container)
    if run_id is None:
        st.title("MedQuiz Generator")
        st.write("Carica un PDF e indica il numero di quesiti per iniziare.")
        return
    items = load_items(container, run_id)
    document = load_document(container, run_id)
    st.title(f"Run {run_id}")
    tab_sim, tab_rev, tab_rep, tab_tax = st.tabs(
        ["🎓 Simulatore", "🔍 Revisione", "📊 Report", "🗂️ Tassonomia"]
    )
    with tab_sim:
        simulator_tab(items, document)
    with tab_rev:
        review_tab(container, run_id, items, document)
    with tab_rep:
        report_tab(container, run_id)
    with tab_tax:
        taxonomy_tab(container, run_id)


main()
