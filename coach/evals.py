"""Eval suites: which metrics decide whether training worked, and the bar for each.

A suite is designed from the scout's notes alongside the reward, then scored
against eval results for the base and trained models.
"""

import json
import math
from pathlib import Path


def validate_eval_suite(evals, game):
    if not isinstance(evals, list) or not 2 <= len(evals) <= 8:
        raise ValueError("An eval suite needs 2-8 metrics")
    clean, seen = [], set()
    for item in evals:
        metric = item.get("metric") if isinstance(item, dict) else None
        if metric not in game.EVAL_METRICS:
            raise ValueError(f"Unknown eval metric {metric!r} for {game.NAME}; "
                             f"allowed: {', '.join(game.EVAL_METRICS)}")
        if metric in seen:
            raise ValueError(f"Eval metric {metric!r} appears twice")
        seen.add(metric)
        target = item.get("target")
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
            raise ValueError(f"Target for {metric} must be a finite number")
        clean.append({"metric": metric, "direction": game.EVAL_METRICS[metric]["direction"],
                      "target": float(target), "why": str(item.get("why", ""))[:500]})
    return clean


def load_metrics(path, game):
    """Metrics from a results file: eval.py summaries (Pac-Man) or coach evaluate output."""
    data = json.loads(Path(path).read_text())
    if "metrics" in data:
        return data["metrics"]
    if hasattr(game, "metrics_from_summary"):
        return game.metrics_from_summary(data)
    raise ValueError(f"{path} has no metrics this game can read")


def _passes(direction, value, target):
    return value >= target if direction == "max" else value <= target


def compare(suite, base, tuned):
    rows = []
    for item in suite:
        metric, direction = item["metric"], item["direction"]
        if metric not in base or metric not in tuned:
            raise KeyError(f"Results are missing eval metric {metric!r}")
        delta = tuned[metric] - base[metric]
        rows.append({"metric": metric, "direction": direction, "target": item["target"],
                     "base": base[metric], "tuned": tuned[metric], "delta": delta,
                     "improved": delta > 0 if direction == "max" else delta < 0,
                     "base_passed": _passes(direction, base[metric], item["target"]),
                     "tuned_passed": _passes(direction, tuned[metric], item["target"])})
    return rows


def markdown_table(rows):
    lines = ["| metric | goal | base | trained | change | trained meets goal |",
             "|---|---|---:|---:|---:|---|"]
    for row in rows:
        goal = ("≥ " if row["direction"] == "max" else "≤ ") + f"{row['target']:g}"
        lines.append(f"| {row['metric']} | {goal} | {row['base']:.3g} | {row['tuned']:.3g} | "
                     f"{row['delta']:+.3g} | {'yes' if row['tuned_passed'] else 'no'} |")
    return "\n".join(lines)
