from __future__ import annotations

import html
from pathlib import Path

from harbor_rag.eval.metrics import MetricRow, aggregate


def write_reports(
    directory: Path,
    *,
    baseline: list[MetricRow],
    bad_retriever: list[MetricRow],
    no_guardrail: list[MetricRow],
    thresholds: dict[str, float],
    ragas_version: str,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "eval_report.md"
    html_path = directory / "eval_report.html"
    body = _markdown(
        baseline=baseline,
        bad_retriever=bad_retriever,
        no_guardrail=no_guardrail,
        thresholds=thresholds,
        ragas_version=ragas_version,
    )
    markdown_path.write_text(body, encoding="utf-8")
    html_path.write_text(_html(body), encoding="utf-8")
    return markdown_path, html_path


def _markdown(
    *,
    baseline: list[MetricRow],
    bad_retriever: list[MetricRow],
    no_guardrail: list[MetricRow],
    thresholds: dict[str, float],
    ragas_version: str,
) -> str:
    base_scores = aggregate(baseline)
    bad_scores = aggregate(bad_retriever)
    unguarded_scores = aggregate(no_guardrail)
    lines = [
        "# Harbor & Pine offline evaluation",
        "",
        "Synthetic corpus. This report is from the deterministic offline pipeline.",
        "LLM-judged RAGAS metrics were not run. No judge scores are included.",
        "RAGAS package pinned for live metrics and for NonLLMContextRecall",
        f"and StringPresence: {ragas_version}.",
        "",
        "## Before and after",
        "",
        "The bad retriever is the broken variant.",
        "The baseline retriever is the one the gate is meant to pass.",
        "Both columns are from the same offline run.",
        "They are not a historical customer comparison.",
        "",
        "| Metric | Bad retriever | Baseline | Threshold |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in thresholds:
        bad = f"{bad_scores[name]:.3f}"
        base = f"{base_scores[name]:.3f}"
        limit = f"{thresholds[name]:.3f}"
        lines.append(f"| {name} | {bad} | {base} | {limit} |")
    lines.extend(
        [
            "",
            "## Guardrail variant",
            "",
            "The no-guardrail variant answers steering and legal questions",
            "with a non-refusal.",
            "",
            "| Metric | No guardrail | Baseline |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in ("refusal_correctness", "string_presence", "grounded_amounts"):
        lines.append(f"| {name} | {unguarded_scores[name]:.3f} | {base_scores[name]:.3f} |")
    lines.extend(["", "## Baseline question failures", ""])
    lines.extend(_failure_lines(baseline))
    lines.extend(["", "## Bad retriever question failures", ""])
    lines.extend(_failure_lines(bad_retriever))
    lines.append("")
    return "\n".join(lines)


def _failure_lines(rows: list[MetricRow]) -> list[str]:
    failed = [row for row in rows if row.failed]
    if not failed:
        return ["None."]
    lines = [
        "| Id | Expected | Predicted | Hit | String presence | Recall | Question |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in failed:
        hit = "" if row.hit is None else str(row.hit)
        recall = "" if row.non_llm_context_recall is None else f"{row.non_llm_context_recall:.3f}"
        question = row.user_input.replace("|", "\\|")
        presence = f"{row.string_presence:.3f}"
        cells = [
            row.item_id,
            row.expected_behavior.value,
            row.predicted_behavior,
            hit,
            presence,
            recall,
            question,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _html(markdown: str) -> str:
    blocks: list[str] = []
    table_lines: list[str] = []

    def flush_table() -> None:
        if not table_lines:
            return
        parsed = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in table_lines
            if not set(line.replace("|", "").replace("-", "").replace(":", "").strip()) <= {""}
        ]
        parsed = [
            cells
            for cells in parsed
            if cells and not all(set(cell) <= set("-: ") for cell in cells)
        ]
        if not parsed:
            table_lines.clear()
            return
        head, body = parsed[0], parsed[1:]
        header = "".join(f"<th>{html.escape(cell)}</th>" for cell in head)
        body_html = "".join(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>"
            for cells in body
        )
        blocks.append(f"<table><thead><tr>{header}</tr></thead><tbody>{body_html}</tbody></table>")
        table_lines.clear()

    for line in markdown.splitlines():
        if line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            blocks.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            blocks.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.strip():
            blocks.append(f"<p>{html.escape(line)}</p>")
    flush_table()
    content = "\n".join(blocks)
    style = (
        "body { font-family: Georgia, serif; margin: 2rem auto; max-width: 960px;"
        " color: #1c1917; background: #fafaf9; }"
        "table { border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; background: white; }"
        "th, td { border: 1px solid #d6d3d1; padding: 0.4rem 0.6rem; text-align: left; }"
        "th { background: #f5f5f4; }"
        "h1 { font-size: 1.6rem; }"
        "h2 { font-size: 1.2rem; margin-top: 2rem; }"
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        "<title>Harbor and Pine offline evaluation</title><style>"
        + style
        + "</style></head><body>"
        + content
        + "</body></html>\n"
    )
