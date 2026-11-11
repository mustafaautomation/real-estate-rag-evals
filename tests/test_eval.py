import json
from pathlib import Path

import pytest

from harbor_rag.config import OFFLINE_METRICS, AppConfig, load_config
from harbor_rag.corpus import load_corpus
from harbor_rag.eval.cli import main
from harbor_rag.eval.dataset import load_questions
from harbor_rag.eval.gate import apply_gate
from harbor_rag.eval.metrics import PipelineRecord, aggregate, score_records
from harbor_rag.eval.report import write_reports
from harbor_rag.eval.runner import run_variant
from harbor_rag.eval.snapshot import load_snapshot, snapshot_mismatches, write_snapshot
from harbor_rag.paths import repo_root
from harbor_rag.types import ExpectedBehavior, Variant


@pytest.fixture(scope="module")
def baseline() -> tuple[AppConfig, list[PipelineRecord]]:
    return run_variant(Variant.BASELINE)


def test_corpus_and_question_counts() -> None:
    root = repo_root()
    documents = load_corpus(root / "data" / "corpus")
    questions = load_questions(root / "data" / "eval" / "questions.json")
    assert 15 <= len(documents) <= 30
    assert 30 <= len(questions) <= 50
    behaviors = {item.expected_behavior for item in questions}
    assert ExpectedBehavior.REFUSE_FAIR_HOUSING in behaviors
    assert ExpectedBehavior.REFUSE_LEGAL in behaviors
    assert ExpectedBehavior.UNKNOWN in behaviors


def test_thresholds_cover_the_offline_metrics() -> None:
    config = load_config(repo_root() / "evals" / "config.yaml")
    assert set(config.offline_thresholds) == set(OFFLINE_METRICS)


def test_baseline_clears_the_gate(baseline: tuple[AppConfig, list[PipelineRecord]]) -> None:
    config, records = baseline
    rows = score_records(records)
    scores = aggregate(rows)
    gate = apply_gate(scores, config.offline_thresholds)
    assert gate.passed, gate.failures
    assert all(not row.failed for row in rows)


def test_bad_retriever_and_no_guardrail_fail_the_gate() -> None:
    for variant in (Variant.BAD_RETRIEVER, Variant.NO_GUARDRAIL):
        config, records = run_variant(variant)
        scores = aggregate(score_records(records))
        gate = apply_gate(scores, config.offline_thresholds)
        assert not gate.passed, scores


def test_baseline_does_not_invent_the_adversarial_price(
    baseline: tuple[AppConfig, list[PipelineRecord]],
) -> None:
    records = baseline[1]
    adversarial = next(record for record in records if "9,999" in record.user_input)
    assert adversarial.response.startswith("I don't know")
    assert "9999" not in adversarial.response
    assert "$9,999" not in adversarial.response


def test_policy_question_is_not_a_steering_refusal(
    baseline: tuple[AppConfig, list[PipelineRecord]],
) -> None:
    records = baseline[1]
    policy = next(record for record in records if record.item_id == "q37")
    assert "protected characteristics named in this policy" in policy.response
    assert not policy.response.startswith("I can't help with steering")


def test_grounded_amounts_catches_a_number_that_is_not_in_context() -> None:
    record = PipelineRecord(
        item_id="invented",
        user_input="What is the monthly rent at 18 Lark Street?",
        response="Monthly rent is $9,999.",
        expected_behavior=ExpectedBehavior.ANSWER,
        reference_phrase="Monthly rent is $2,450.",
        retrieved_doc_ids=("listing-18-lark",),
        reference_doc_ids=("listing-18-lark",),
        retrieved_contexts=("Monthly rent is $2,450.",),
        reference_contexts=("Monthly rent is $2,450.",),
    )
    scored = score_records([record])[0]
    assert scored.grounded_amounts == 0.0


def test_ragas_non_llm_metrics_are_used() -> None:
    record = PipelineRecord(
        item_id="sample",
        user_input="Where is the office?",
        response="The brokerage office is at 12 Pine Wharf.",
        expected_behavior=ExpectedBehavior.ANSWER,
        reference_phrase="12 Pine Wharf",
        retrieved_doc_ids=("agency-overview",),
        reference_doc_ids=("agency-overview",),
        retrieved_contexts=("The brokerage office is at 12 Pine Wharf.",),
        reference_contexts=("The brokerage office is at 12 Pine Wharf.", "x" * 80),
    )
    scored = score_records([record])[0]
    assert scored.string_presence == 1.0
    assert scored.non_llm_context_recall == 0.5
    assert scored.hit is True
    assert scored.reciprocal_rank == 1.0


def test_offline_cli_passes_and_bad_variant_exits(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--mode", "offline", "--check-snapshot"])
    captured = capsys.readouterr()
    assert "snapshot matches" in captured.out
    with pytest.raises(SystemExit) as exc:
        main(["--mode", "offline", "--variant", "bad_retriever"])
    assert exc.value.code not in (0, None)


def test_live_mode_without_a_key_does_not_invent_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--mode", "live"])
    assert exc.value.code == 2


def test_report_contains_the_before_and_after_table(tmp_path: Path) -> None:
    _, records = run_variant(Variant.BASELINE)
    _, bad_records = run_variant(Variant.BAD_RETRIEVER)
    _, guard_records = run_variant(Variant.NO_GUARDRAIL)
    config = load_config(repo_root() / "evals" / "config.yaml")
    markdown, html = write_reports(
        tmp_path,
        baseline=score_records(records),
        bad_retriever=score_records(bad_records),
        no_guardrail=score_records(guard_records),
        thresholds=config.offline_thresholds,
        ragas_version="0.4.3",
    )
    text = markdown.read_text(encoding="utf-8")
    assert "LLM-judged RAGAS metrics were not run" in text
    assert "Bad retriever" in text
    assert "No judge scores" in text
    assert "<table>" in html.read_text(encoding="utf-8")


def test_snapshot_round_trip(tmp_path: Path) -> None:
    _, records = run_variant(Variant.BASELINE)
    write_snapshot(tmp_path, records, "0.4.3")
    meta = json.loads((tmp_path / "data" / "snapshots" / "snapshot_meta.json").read_text())
    assert meta["judge_scores"] is None
    assert meta["label"] == "pipeline_outputs_only_judge_scores_not_recorded"
    assert snapshot_mismatches(tmp_path, records) == []
    saved = load_snapshot(tmp_path)
    assert saved[0]["judge_scores"] is None
    assert len(saved) == len(records)
