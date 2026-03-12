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

    # Optimal threshold via Youden's J (maximises TPR - FPR)
    best_thresh, best_j = _youden_threshold(labels, probs)

    # Metrics at multiple thresholds
    thresholds = sorted({25, 30, 35, 40, best_thresh, 50})
    threshold_metrics = {}
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in probs]
        tp_t = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp_t = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn_t = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        tn_t = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
        acc_t = sum(p == l for p, l in zip(preds, labels)) / n_parsed
        threshold_metrics[t] = {
            "accuracy": round(acc_t, 4),
            "confusion_matrix": {"TP": tp_t, "FP": fp_t, "FN": fn_t, "TN": tn_t},
        }

    avg_prob_default = sum(default_probs) / len(default_probs) if default_probs else None
    avg_prob_control = sum(control_probs) / len(control_probs) if control_probs else None

    return {
        "n_default": n_default,
        "n_control": n_control,
        "n_parsed": n_parsed,
        "n_error": n_error,
        "auc_roc": round(auc, 4) if auc is not None else None,
        "optimal_threshold": best_thresh,
        "optimal_youden_j": round(best_j, 4),
        "threshold_metrics": threshold_metrics,
        # Keep legacy key for backward compat
        "accuracy_at_50": threshold_metrics[50]["accuracy"],
        "confusion_matrix": threshold_metrics[50]["confusion_matrix"],
        "avg_prob_default_group": round(avg_prob_default, 1) if avg_prob_default is not None else None,
        "avg_prob_control_group": round(avg_prob_control, 1) if avg_prob_control is not None else None,
        "errors": errors if errors else [],
    }


def _youden_threshold(labels: list[int], probs: list[float]) -> tuple[int, float]:
    """Find optimal threshold via Youden's J = TPR - FPR.

    Returns (threshold, j_score) where threshold is in [0, 100].
    """
    best_t, best_j = 50, -1.0
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 50, 0.0
    for t in range(1, 100):
        preds = [1 if p >= t else 0 for p in probs]
        tpr = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1) / n_pos
        fpr = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0) / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return best_t, best_j


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
    opt_t = metrics.get("optimal_threshold")
    opt_j = metrics.get("optimal_youden_j")
    print(f"AUC-ROC              : {auc:.4f}" if auc is not None else "AUC-ROC              : N/A")
    if opt_t is not None:
        print(f"Optimal threshold    : {opt_t}%  (Youden's J = {opt_j:.4f})")
    print()

    pd = metrics.get("avg_prob_default_group")
    pc = metrics.get("avg_prob_control_group")
    print(f"Avg prob (default group) : {pd:.1f}%" if pd is not None else "Avg prob (default group) : N/A")
    print(f"Avg prob (control group) : {pc:.1f}%" if pc is not None else "Avg prob (control group) : N/A")
    print()

    thresh_metrics = metrics.get("threshold_metrics", {})
    if thresh_metrics:
        print(f"{'Threshold':>10}  {'Accuracy':>10}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'TN':>6}")
        print("  " + "-" * 52)
        for t, tm in sorted(thresh_metrics.items()):
            marker = " ← optimal" if t == opt_t else ""
            cm = tm["confusion_matrix"]
            print(f"  {t:>8}%  {tm['accuracy']:>10.4f}  {cm['TP']:>6}  {cm['FP']:>6}  {cm['FN']:>6}  {cm['TN']:>6}{marker}")
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
