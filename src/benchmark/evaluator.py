"""Evaluate benchmark results from a JSONL file.

Reads cases.jsonl produced by runner.py and computes:
- AUC-ROC
- Accuracy at threshold=50
- Average predicted probability by group
- Confusion matrix
"""

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def load_results(jsonl_path: Union[str, Path]) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate(jsonl_path: Union[str, Path]) -> dict:
    """Compute evaluation metrics from a benchmark JSONL file.

    Args:
        jsonl_path: Path to cases.jsonl produced by runner.py

    Returns:
        Dict with evaluation metrics.
    """
    records = load_results(jsonl_path)

    if not records:
        return {"error": "No records found in JSONL file"}

    labels: list[int] = []
    probs: list[float] = []
    errors: list[dict] = []
    n_default = 0
    n_control = 0

    default_probs: list[float] = []
    control_probs: list[float] = []

    for rec in records:
        label = 1 if rec.get("label") else 0
        group = rec.get("group", "unknown")
        pred = rec.get("prediction", {})

        if group == "default":
            n_default += 1
        elif group == "control":
            n_control += 1

        prob = pred.get("probability_pct")
        if prob is None or "error" in pred:
            errors.append({"company": rec.get("company_name"), "error": pred.get("error")})
            continue

        try:
            prob = float(prob)
        except (TypeError, ValueError):
            errors.append({"company": rec.get("company_name"), "error": f"Bad prob: {prob}"})
            continue

        labels.append(label)
        probs.append(prob)

        if group == "default":
            default_probs.append(prob)
        elif group == "control":
            control_probs.append(prob)

    n_parsed = len(labels)
    n_error = len(errors)

    if n_parsed < 2:
        return {
            "n_default": n_default, "n_control": n_control,
            "n_parsed": n_parsed, "n_error": n_error,
            "error": "Not enough parseable results to compute metrics",
            "errors": errors,
        }

    # AUC-ROC
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
        auc = float(roc_auc_score(labels, probs))
    except ImportError:
        auc = _manual_auc(labels, probs)
    except Exception as e:
        auc = None
        logger.warning(f"AUC computation failed: {e}")

    # Accuracy at threshold 50
    predicted_labels = [1 if p >= 50 else 0 for p in probs]
    correct = sum(p == l for p, l in zip(predicted_labels, labels))
    accuracy = correct / n_parsed

    # Confusion matrix
    tp = sum(1 for p, l in zip(predicted_labels, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predicted_labels, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predicted_labels, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predicted_labels, labels) if p == 0 and l == 0)

    avg_prob_default = sum(default_probs) / len(default_probs) if default_probs else None
    avg_prob_control = sum(control_probs) / len(control_probs) if control_probs else None

    return {
        "n_default": n_default,
        "n_control": n_control,
        "n_parsed": n_parsed,
        "n_error": n_error,
        "auc_roc": round(auc, 4) if auc is not None else None,
        "accuracy_at_50": round(accuracy, 4),
        "avg_prob_default_group": round(avg_prob_default, 1) if avg_prob_default is not None else None,
        "avg_prob_control_group": round(avg_prob_control, 1) if avg_prob_control is not None else None,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "errors": errors if errors else [],
    }


def _manual_auc(labels: list[int], scores: list[float]) -> float:
    """Compute AUC via Mann-Whitney U (no sklearn required)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    u = sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg)
    return u / (len(pos) * len(neg))


def print_report(metrics: dict) -> None:
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION REPORT")
    print("=" * 60)
    print(f"Default cases  : {metrics.get('n_default', '?')}")
    print(f"Control cases  : {metrics.get('n_control', '?')}")
    print(f"Parsed OK      : {metrics.get('n_parsed', '?')}")
    print(f"Parse errors   : {metrics.get('n_error', '?')}")
    print()

    auc = metrics.get("auc_roc")
    acc = metrics.get("accuracy_at_50")
    print(f"AUC-ROC        : {auc:.4f}" if auc is not None else "AUC-ROC        : N/A")
    print(f"Accuracy@50    : {acc:.4f}" if acc is not None else "Accuracy@50    : N/A")
    print()

    pd = metrics.get("avg_prob_default_group")
    pc = metrics.get("avg_prob_control_group")
    print(f"Avg prob (default group) : {pd:.1f}%" if pd is not None else "Avg prob (default group) : N/A")
    print(f"Avg prob (control group) : {pc:.1f}%" if pc is not None else "Avg prob (control group) : N/A")
    print()

    cm = metrics.get("confusion_matrix", {})
    if cm:
        print(f"Confusion matrix (threshold=50%):")
        print(f"  Predicted →  Default   Control")
        print(f"  Actual Default : {cm.get('TP', 0):6d}    {cm.get('FN', 0):6d}")
        print(f"  Actual Control : {cm.get('FP', 0):6d}    {cm.get('TN', 0):6d}")

    errors = metrics.get("errors", [])
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {e.get('company', '?')}: {e.get('error', '?')}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    print("=" * 60)
