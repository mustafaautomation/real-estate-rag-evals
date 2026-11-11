from __future__ import annotations

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from harbor_rag.eval.gate import apply_gate
from harbor_rag.eval.live import LiveEvalError, score_live
from harbor_rag.eval.metrics import aggregate
from harbor_rag.eval.report import write_reports
from harbor_rag.eval.runner import run_variant, score_variant
from harbor_rag.eval.snapshot import snapshot_mismatches, write_snapshot
from harbor_rag.paths import repo_root
from harbor_rag.types import Variant


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="harbor-eval")
    parser.add_argument("--mode", choices=("offline", "live"), required=True)
    parser.add_argument(
        "--variant",
        choices=tuple(item.value for item in Variant),
        default=Variant.BASELINE.value,
    )
    parser.add_argument("--check-snapshot", action="store_true")
    parser.add_argument("--write-snapshot", action="store_true")
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    root = args.root or repo_root()
    variant = Variant(args.variant)
    try:
        if args.mode == "live":
            _live(root, variant, args.write_report)
            return
        _offline(
            root,
            variant,
            check_snapshot=bool(args.check_snapshot),
            write_snapshot_file=bool(args.write_snapshot),
            report_dir=args.write_report,
        )
    except LiveEvalError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


def _offline(
    root: Path,
    variant: Variant,
    *,
    check_snapshot: bool,
    write_snapshot_file: bool,
    report_dir: Path | None,
) -> None:
    config, records, rows = score_variant(variant, root)
    scores = aggregate(rows)
    gate = apply_gate(scores, config.offline_thresholds)
    _print_scores(variant.value, scores, config.offline_thresholds)
    ragas_version = version("ragas")
    if write_snapshot_file:
        if variant is not Variant.BASELINE:
            raise SystemExit("refusing to write a snapshot for a non-baseline variant")
        write_snapshot(root, records, ragas_version)
        print(f"wrote snapshot under {root / 'data' / 'snapshots'}")
    if check_snapshot:
        if variant is not Variant.BASELINE:
            raise SystemExit("--check-snapshot is only valid for the baseline variant")
        mismatches = snapshot_mismatches(root, records)
        if mismatches:
            preview = "\n".join(mismatches[:8])
            raise SystemExit(f"snapshot mismatch\n{preview}")
        print("snapshot matches the baseline rerun")
    if report_dir is not None:
        _bad_config, _bad_records, bad_rows = score_variant(Variant.BAD_RETRIEVER, root)
        _guard_config, _guard_records, guard_rows = score_variant(Variant.NO_GUARDRAIL, root)
        markdown_path, html_path = write_reports(
            report_dir,
            baseline=rows,
            bad_retriever=bad_rows,
            no_guardrail=guard_rows,
            thresholds=config.offline_thresholds,
            ragas_version=ragas_version,
        )
        print(f"wrote {markdown_path}")
        print(f"wrote {html_path}")
    if not gate.passed:
        detail = "\n".join(gate.failures)
        raise SystemExit(f"eval gate failed\n{detail}")


def _live(root: Path, variant: Variant, report_dir: Path | None) -> None:
    config, records = run_variant(variant, root)
    scores = score_live(
        records,
        model=config.live_model,
        embedding_model=config.live_embedding_model,
    )
    gate = apply_gate(scores, config.live_thresholds)
    _print_scores(f"live:{variant.value}", scores, config.live_thresholds)
    if report_dir is not None:
        print("live mode does not rewrite the offline example report")
    if not gate.passed:
        detail = "\n".join(gate.failures)
        raise SystemExit(f"eval gate failed\n{detail}")


def _print_scores(label: str, scores: dict[str, float], thresholds: dict[str, float]) -> None:
    print(f"variant={label}")
    for name, score in scores.items():
        threshold = thresholds.get(name)
        mark = "" if threshold is None or score >= threshold else " FAIL"
        if threshold is None:
            print(f"  {name}: {score:.4f}")
        else:
            print(f"  {name}: {score:.4f}  threshold {threshold:.4f}{mark}")


if __name__ == "__main__":
    main()
