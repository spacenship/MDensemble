#!/usr/bin/env python3
"""Summarize deterministic validation and endpoint metrics from gap runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VAL_PATTERN = re.compile(
    r"step (?P<step>\d+) \| val_loss=(?P<val_loss>[-+0-9.eE]+) \| "
    r"fm=(?P<fm>[-+0-9.eE]+).*?zero_fm=(?P<zero_fm>[-+0-9.eE]+), "
    r"fm_improvement=(?P<fm_improvement>[-+0-9.eE]+)% \| lr=(?P<lr>[-+0-9.eE]+)"
)
ENDPOINT_PATTERN = re.compile(
    r"step (?P<step>\d+) \| endpoint_rmsd: source=(?P<source_rmsd>[-+0-9.eE]+), "
    r"generated=(?P<generated_rmsd>[-+0-9.eE]+), improvement=(?P<endpoint_improvement>[-+0-9.eE]+)%, "
    r"win_rate=(?P<win_rate>[-+0-9.eE]+)% \| endpoint_physics: "
    r"bond=(?P<endpoint_bond>[-+0-9.eE]+), angle=(?P<endpoint_angle>[-+0-9.eE]+), "
    r"clash=(?P<endpoint_clash>[-+0-9.eE]+)"
)


def _numeric_groups(match: re.Match[str]) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for name, value in match.groupdict().items():
        values[name] = int(value) if name == "step" else float(value)
    return values


def parse_log(path: Path) -> list[dict[str, float | int]]:
    by_step: dict[int, dict[str, float | int]] = {}
    for line in path.read_text().splitlines():
        val_match = VAL_PATTERN.search(line)
        if val_match:
            values = _numeric_groups(val_match)
            by_step[int(values["step"])] = values
            continue
        endpoint_match = ENDPOINT_PATTERN.search(line)
        if endpoint_match:
            values = _numeric_groups(endpoint_match)
            step = int(values.pop("step"))
            by_step.setdefault(step, {"step": step}).update(values)
    return [by_step[step] for step in sorted(by_step) if "val_loss" in by_step[step]]


def summarize_run(gap: int, records: list[dict[str, float | int]]) -> dict[str, Any]:
    if not records:
        raise ValueError(f"No validation records found for gap={gap}")
    complete = [record for record in records if "endpoint_improvement" in record]
    if not complete:
        raise ValueError(f"No endpoint records found for gap={gap}")
    return {
        "gap": gap,
        "num_validations": len(records),
        "best_validation": min(complete, key=lambda record: float(record["val_loss"])),
        "best_endpoint": max(complete, key=lambda record: float(record["endpoint_improvement"])),
        "final": complete[-1],
    }


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| gap | best val step | best val | FM improve | endpoint@best-val | best endpoint step | best endpoint | win rate | endpoint bond |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        best_val = summary["best_validation"]
        best_endpoint = summary["best_endpoint"]
        lines.append(
            f"| {summary['gap']} | {best_val['step']} | {best_val['val_loss']:.6f} | "
            f"{best_val['fm_improvement']:.2f}% | {best_val['endpoint_improvement']:.2f}% | "
            f"{best_endpoint['step']} | {best_endpoint['endpoint_improvement']:.2f}% | "
            f"{best_endpoint['win_rate']:.2f}% | {best_endpoint['endpoint_bond']:.6f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("ablation_logs"))
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    summaries = [
        summarize_run(gap, parse_log(args.log_dir / f"gap{gap}.log")) for gap in (1, 2, 5)
    ]
    markdown = markdown_table(summaries)
    print(markdown)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summaries, indent=2) + "\n")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown + "\n")


if __name__ == "__main__":
    main()
