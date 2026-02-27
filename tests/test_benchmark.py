"""Unit tests for src/benchmark — no DB or LLM required."""

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.benchmark.config import BenchmarkConfig
from src.benchmark.case_builder import CaseRecord
from src.benchmark.evaluator import evaluate, _manual_auc, print_report
from src.benchmark.prompt_builder import build_prompt


# ---------------------------------------------------------------------------
# BenchmarkConfig tests
# ---------------------------------------------------------------------------

class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.prediction_horizon_months == 6
        assert cfg.lookback_months == 12
        assert cfg.n_default_cases == 100
        assert cfg.n_control_cases == 100
        assert cfg.include_risk_indicators is True
        assert cfg.include_fundamentals is True

    def test_custom_values(self):
        cfg = BenchmarkConfig(prediction_horizon_months=12, lookback_months=24, n_default_cases=50)
        assert cfg.prediction_horizon_months == 12
        assert cfg.lookback_months == 24
        assert cfg.n_default_cases == 50

    def test_no_fundamentals_flag(self):
        cfg = BenchmarkConfig(include_fundamentals=False)
        assert cfg.include_fundamentals is False


# ---------------------------------------------------------------------------
# CaseRecord tests
# ---------------------------------------------------------------------------

class TestCaseRecord:
    def _make_case(self, group="default", label=True) -> CaseRecord:
        return CaseRecord(
            u3_company_number=12345,
            id_bb_company=99999,
            ticker="TEST US",
            company_name="Test Corp",
            industry_sector="Technology",
            cutoff_date=date(2020, 1, 1),
            event_date=date(2020, 7, 1) if group == "default" else None,
            label=label,
            group=group,
        )

    def test_default_case_fields(self):
        case = self._make_case("default", True)
        assert case.label is True
        assert case.group == "default"
        assert case.event_date == date(2020, 7, 1)
        assert case.cutoff_date < case.event_date

    def test_control_case_fields(self):
        case = self._make_case("control", False)
        assert case.label is False
        assert case.group == "control"
        assert case.event_date is None


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict]) -> str:
    """Write records to a temp JSONL file, return path."""
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


class TestEvaluator:
    def test_perfect_separation(self):
        records = [
            {"u3_company_number": 1, "company_name": "A", "cutoff_date": "2020-01-01",
             "label": True, "group": "default",
             "prediction": {"probability_pct": 80}},
            {"u3_company_number": 2, "company_name": "B", "cutoff_date": "2020-01-01",
             "label": False, "group": "control",
             "prediction": {"probability_pct": 10}},
        ]
        path = _write_jsonl(records)
        metrics = evaluate(path)
        os.unlink(path)

        assert metrics["n_default"] == 1
        assert metrics["n_control"] == 1
        assert metrics["n_parsed"] == 2
        assert metrics["n_error"] == 0
        assert metrics["auc_roc"] == 1.0
        assert metrics["accuracy_at_50"] == 1.0
        assert metrics["avg_prob_default_group"] == 80.0
        assert metrics["avg_prob_control_group"] == 10.0

    def test_random_prediction(self):
        """50% predictions should give roughly 0.5 AUC."""
        records = [
            {"u3_company_number": i, "company_name": f"Co{i}", "cutoff_date": "2020-01-01",
             "label": i % 2 == 0, "group": "default" if i % 2 == 0 else "control",
             "prediction": {"probability_pct": 50}}
            for i in range(10)
        ]
        path = _write_jsonl(records)
        metrics = evaluate(path)
        os.unlink(path)

        # With all probs=50, all predictions are 1 at threshold 50
        assert metrics["n_parsed"] == 10
        assert metrics["auc_roc"] == 0.5

    def test_handles_missing_probability(self):
        records = [
            {"u3_company_number": 1, "company_name": "Good", "cutoff_date": "2020-01-01",
             "label": True, "group": "default",
             "prediction": {"probability_pct": 70}},
            {"u3_company_number": 2, "company_name": "Bad", "cutoff_date": "2020-01-01",
             "label": False, "group": "control",
             "prediction": {"error": "LLM failed", "probability_pct": None}},
        ]
        path = _write_jsonl(records)
        metrics = evaluate(path)
        os.unlink(path)

        assert metrics["n_parsed"] == 1
        assert metrics["n_error"] == 1

    def test_empty_file(self):
        path = _write_jsonl([])
        metrics = evaluate(path)
        os.unlink(path)
        assert "error" in metrics

    def test_confusion_matrix_values(self):
        records = [
            # TP: label=1, predicted HIGH (prob>=50)
            {"u3_company_number": 1, "label": True, "group": "default",
             "company_name": "A", "cutoff_date": "2020-01-01",
             "prediction": {"probability_pct": 70}},
            # TN: label=0, predicted LOW (prob<50)
            {"u3_company_number": 2, "label": False, "group": "control",
             "company_name": "B", "cutoff_date": "2020-01-01",
             "prediction": {"probability_pct": 20}},
            # FP: label=0, predicted HIGH
            {"u3_company_number": 3, "label": False, "group": "control",
             "company_name": "C", "cutoff_date": "2020-01-01",
             "prediction": {"probability_pct": 60}},
            # FN: label=1, predicted LOW
            {"u3_company_number": 4, "label": True, "group": "default",
             "company_name": "D", "cutoff_date": "2020-01-01",
             "prediction": {"probability_pct": 30}},
        ]
        path = _write_jsonl(records)
        metrics = evaluate(path)
        os.unlink(path)

        cm = metrics["confusion_matrix"]
        assert cm["TP"] == 1
        assert cm["TN"] == 1
        assert cm["FP"] == 1
        assert cm["FN"] == 1

    def test_print_report_doesnt_crash(self, capsys):
        records = [
            {"u3_company_number": 1, "label": True, "group": "default",
             "company_name": "A", "cutoff_date": "2020-01-01",
             "prediction": {"probability_pct": 80}},
            {"u3_company_number": 2, "label": False, "group": "control",
             "company_name": "B", "cutoff_date": "2020-01-01",
             "prediction": {"probability_pct": 10}},
        ]
        path = _write_jsonl(records)
        metrics = evaluate(path)
        os.unlink(path)
        print_report(metrics)
        captured = capsys.readouterr()
        assert "AUC" in captured.out
        assert "Accuracy" in captured.out


class TestManualAUC:
    def test_perfect(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        assert _manual_auc(labels, scores) == 1.0

    def test_random(self):
        labels = [1, 0, 1, 0]
        scores = [0.5, 0.5, 0.5, 0.5]
        assert _manual_auc(labels, scores) == 0.5

    def test_all_same_class(self):
        labels = [1, 1, 1]
        scores = [0.9, 0.7, 0.5]
        result = _manual_auc(labels, scores)
        assert result != result  # NaN


# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def _make_case(self) -> CaseRecord:
        return CaseRecord(
            u3_company_number=1,
            id_bb_company=None,
            ticker="TEST US",
            company_name="Test Corp",
            industry_sector="Energy",
            cutoff_date=date(2020, 6, 1),
            event_date=date(2020, 12, 1),
            label=True,
            group="default",
        )

    def _make_data(self) -> dict:
        return {
            "company": {
                "u3_company_number": 1,
                "id_bb_company": None,
                "ticker": "TEST US",
                "company_name": "Test Corp",
                "country_name": "United States",
                "market_status": "DLST",
                "prime_exchange": "NYSE",
                "id_isin": None,
                "industry_sector": "Energy",
                "industry_group": "Oil & Gas",
                "industry_subgroup": "Integrated",
            },
            "fundamentals": {"income_statement": [], "balance_sheet": [], "cash_flow": []},
            "risk_indicators": [],
            "market_data": [],
            "credit_events": [],
            "macro": None,
            "transcripts": [],
        }

    def test_prompt_contains_cutoff_date(self):
        cfg = BenchmarkConfig(prediction_horizon_months=6)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "2020-06-01" in prompt

    def test_prompt_contains_horizon(self):
        cfg = BenchmarkConfig(prediction_horizon_months=6)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "6 months" in prompt

    def test_prompt_contains_end_date(self):
        cfg = BenchmarkConfig(prediction_horizon_months=6)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        # cutoff 2020-06-01 + 6 months = 2020-12-01
        assert "2020-12-01" in prompt

    def test_prompt_warns_no_future_data(self):
        cfg = BenchmarkConfig()
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "NOT" in prompt or "IMPORTANT" in prompt

    def test_prompt_is_nonempty_string(self):
        cfg = BenchmarkConfig()
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert isinstance(prompt, str)
        assert len(prompt) > 200


# ---------------------------------------------------------------------------
# data_fetcher as_of integration smoke test (no DB)
# ---------------------------------------------------------------------------

class TestDataFetcherAsOf:
    """Ensure _fetch_* functions accept as_of without crashing at import/logic level."""

    def test_fetch_risk_accepts_as_of(self):
        """Check that the function signature includes as_of."""
        from src.pipeline.data_fetcher import _fetch_risk_indicators
        import inspect
        sig = inspect.signature(_fetch_risk_indicators)
        assert "as_of" in sig.parameters

    def test_fetch_market_accepts_as_of(self):
        from src.pipeline.data_fetcher import _fetch_market_data
        import inspect
        sig = inspect.signature(_fetch_market_data)
        assert "as_of" in sig.parameters

    def test_fetch_macro_accepts_as_of(self):
        from src.pipeline.data_fetcher import _fetch_macro_context
        import inspect
        sig = inspect.signature(_fetch_macro_context)
        assert "as_of" in sig.parameters

    def test_fetch_credit_events_accepts_as_of(self):
        from src.pipeline.data_fetcher import _fetch_credit_events
        import inspect
        sig = inspect.signature(_fetch_credit_events)
        assert "as_of" in sig.parameters

    def test_fetch_all_accepts_as_of(self):
        from src.pipeline.data_fetcher import fetch_all
        import inspect
        sig = inspect.signature(fetch_all)
        assert "as_of" in sig.parameters

    def test_fetch_fundamentals_accepts_as_of(self):
        from src.pipeline.data_fetcher import _fetch_fundamentals
        import inspect
        sig = inspect.signature(_fetch_fundamentals)
        assert "as_of" in sig.parameters


# ---------------------------------------------------------------------------
# Predictor system prompt tests (no API call)
# ---------------------------------------------------------------------------

class TestPredictorPrompt:
    def test_system_prompt_template_has_placeholders(self):
        from src.pipeline.predictor import _SYSTEM_PROMPT_TEMPLATE
        assert "{horizon_clause}" in _SYSTEM_PROMPT_TEMPLATE
        assert "{cutoff_date}" in _SYSTEM_PROMPT_TEMPLATE

    def test_default_system_prompt_renders(self):
        from src.pipeline.predictor import SYSTEM_PROMPT
        assert "12" in SYSTEM_PROMPT or "months" in SYSTEM_PROMPT

    def test_predict_signature_has_cutoff(self):
        from src.pipeline.predictor import predict
        import inspect
        sig = inspect.signature(predict)
        assert "cutoff_date" in sig.parameters
        assert "horizon_months" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
