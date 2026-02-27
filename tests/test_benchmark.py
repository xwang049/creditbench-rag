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
from src.benchmark.prompt_builder import (
    build_prompt,
    _build_year_map,
    _blur_years_in_text,
    _anonymize_company_dict,
)


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
        # Risk indicators and market data are OFF by default (limited coverage)
        assert cfg.include_risk_indicators is False
        assert cfg.include_market_data is False
        # Macro and fundamentals are ON by default
        assert cfg.include_macro is True
        assert cfg.include_fundamentals is True
        # Contamination-prevention ON by default
        assert cfg.anonymize_company is True
        assert cfg.blur_year is True

    def test_custom_values(self):
        cfg = BenchmarkConfig(prediction_horizon_months=12, lookback_months=24, n_default_cases=50)
        assert cfg.prediction_horizon_months == 12
        assert cfg.lookback_months == 24
        assert cfg.n_default_cases == 50

    def test_no_fundamentals_flag(self):
        cfg = BenchmarkConfig(include_fundamentals=False)
        assert cfg.include_fundamentals is False

    def test_opt_in_risk_and_market(self):
        cfg = BenchmarkConfig(include_risk_indicators=True, include_market_data=True)
        assert cfg.include_risk_indicators is True
        assert cfg.include_market_data is True

    def test_disable_anonymization(self):
        cfg = BenchmarkConfig(anonymize_company=False, blur_year=False)
        assert cfg.anonymize_company is False
        assert cfg.blur_year is False


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
# Year-blurring unit tests
# ---------------------------------------------------------------------------

class TestBuildYearMap:
    def test_cutoff_year_is_Y0(self):
        ym = _build_year_map(2020)
        assert ym[2020] == "Y0"

    def test_prior_years(self):
        ym = _build_year_map(2020)
        assert ym[2019] == "Y-1"
        assert ym[2018] == "Y-2"
        assert ym[2010] == "Y-10"

    def test_future_years(self):
        ym = _build_year_map(2020)
        assert ym[2021] == "Y+1"
        assert ym[2022] == "Y+2"

    def test_coverage_span(self):
        ym = _build_year_map(2020)
        # Should cover at least 20 years back and 2 years forward
        assert 2000 in ym
        assert 1990 in ym
        assert 2022 in ym


class TestBlurYearsInText:
    """Tests that only date/fiscal-period years are replaced, not numeric levels."""

    def _ym(self) -> dict[int, str]:
        return _build_year_map(2020)

    # --- ISO dates -----------------------------------------------------------
    def test_iso_date_full(self):
        result = _blur_years_in_text("As of 2020-06-01", self._ym())
        assert "Y0-06-01" in result
        assert "2020" not in result

    def test_iso_date_year_month_only(self):
        result = _blur_years_in_text("2019-12", self._ym())
        assert "Y-1-12" in result
        assert "2019" not in result

    def test_multiple_iso_dates(self):
        result = _blur_years_in_text(
            "cutoff 2020-06-01 end 2020-12-01 prior 2018-03-15", self._ym()
        )
        assert "Y0-06-01" in result
        assert "Y0-12-01" in result
        assert "Y-2-03-15" in result
        assert "2020" not in result
        assert "2018" not in result

    # --- Fiscal periods ------------------------------------------------------
    def test_fiscal_year_A_suffix(self):
        result = _blur_years_in_text("Period: 2020A 2019A", self._ym())
        assert "Y0A" in result
        assert "Y-1A" in result

    def test_fiscal_quarter_Q_suffix(self):
        result = _blur_years_in_text("2019Q3", self._ym())
        assert "Y-1Q3" in result
        assert "2019" not in result

    # --- Numeric levels (must NOT be blurred) --------------------------------
    def test_sp500_level_preserved(self):
        """S&P500 values like 2000, 2400 must not be replaced."""
        result = _blur_years_in_text("S&P500=2000, VIX=24.3", self._ym())
        assert "2000" in result        # numeric level — untouched
        assert "24.3" in result

    def test_vix_and_yield_levels_preserved(self):
        result = _blur_years_in_text("VIX=80.0, 10Y=3.84%, CPI=230.5", self._ym())
        assert "80.0" in result
        assert "3.84" in result
        assert "230.5" in result

    def test_revenue_large_number_preserved(self):
        # "2000.5M" — followed by "." so NOT matching YYYY-MM pattern;
        # also not followed by A/Q. Must remain unchanged.
        result = _blur_years_in_text("Revenue: 2000.5M, Assets: 19000B", self._ym())
        assert "2000.5M" in result

    def test_risk_indicator_year_month_column(self):
        # Risk indicator table rows look like "2020-06   DTD=5.2"
        result = _blur_years_in_text("2020-06   5.2   0.35", self._ym())
        assert "Y0-06" in result
        assert "5.2" in result

    def test_macro_row_with_date_and_levels(self):
        line = "As of 2020-06-01: S&P500=3100, VIX=28.1, Unemp=13.3%, CPI=257.0"
        result = _blur_years_in_text(line, self._ym())
        assert "Y0-06-01" in result
        assert "3100" in result
        assert "28.1" in result
        assert "13.3" in result
        assert "2020" not in result

    def test_bond_yield_row(self):
        line = "Treasury yields (2020-06-01): 2Y=0.17%, 10Y=0.65%, 30Y=1.41%"
        result = _blur_years_in_text(line, self._ym())
        assert "Y0-06-01" in result
        assert "0.17" in result
        assert "0.65" in result
        assert "2020" not in result


# ---------------------------------------------------------------------------
# Company anonymisation unit tests
# ---------------------------------------------------------------------------

class TestAnonymizeCompanyDict:
    def _make_company(self) -> dict:
        return {
            "u3_company_number": 99999,
            "id_bb_company": 12345,
            "ticker": "TEST US",
            "company_name": "Test Corp",
            "country_name": "United States",
            "market_status": "—",
            "prime_exchange": "—",
            "id_isin": None,
            "industry_sector": "Energy",
            "industry_group": None,
            "industry_subgroup": None,
        }

    def test_name_replaced(self):
        anon = _anonymize_company_dict(self._make_company())
        assert "Test Corp" not in anon["company_name"]
        assert "99999" in anon["company_name"]

    def test_ticker_replaced(self):
        anon = _anonymize_company_dict(self._make_company())
        assert "TEST US" not in anon["ticker"]
        assert "99999" in anon["ticker"]

    def test_sector_preserved(self):
        anon = _anonymize_company_dict(self._make_company())
        assert anon["industry_sector"] == "Energy"

    def test_country_preserved(self):
        anon = _anonymize_company_dict(self._make_company())
        assert anon["country_name"] == "United States"

    def test_original_dict_not_mutated(self):
        original = self._make_company()
        _ = _anonymize_company_dict(original)
        assert original["company_name"] == "Test Corp"
        assert original["ticker"] == "TEST US"


# ---------------------------------------------------------------------------
# PromptBuilder integration tests
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

    # --- Tests with both flags DISABLED (explicit raw mode) ------------------

    def test_prompt_contains_cutoff_date(self):
        """Cutoff date is present in raw form when blur_year=False."""
        cfg = BenchmarkConfig(prediction_horizon_months=6,
                              anonymize_company=False, blur_year=False)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "2020-06-01" in prompt

    def test_prompt_contains_horizon(self):
        cfg = BenchmarkConfig(prediction_horizon_months=6,
                              anonymize_company=False, blur_year=False)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "6 months" in prompt

    def test_prompt_contains_end_date(self):
        cfg = BenchmarkConfig(prediction_horizon_months=6,
                              anonymize_company=False, blur_year=False)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        # cutoff 2020-06-01 + 6 months = 2020-12-01
        assert "2020-12-01" in prompt

    def test_prompt_warns_no_future_data(self):
        cfg = BenchmarkConfig(anonymize_company=False, blur_year=False)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "NOT" in prompt or "IMPORTANT" in prompt

    def test_prompt_is_nonempty_string(self):
        cfg = BenchmarkConfig(anonymize_company=False, blur_year=False)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert isinstance(prompt, str)
        assert len(prompt) > 200

    # --- Anonymization tests -------------------------------------------------

    def test_anonymize_hides_company_name(self):
        cfg = BenchmarkConfig(anonymize_company=True, blur_year=False)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "Test Corp" not in prompt
        assert "Company-1" in prompt or "U3-1" in prompt

    def test_anonymize_hides_ticker(self):
        cfg = BenchmarkConfig(anonymize_company=True, blur_year=False)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "TEST US" not in prompt

    def test_sector_still_present_after_anonymize(self):
        cfg = BenchmarkConfig(anonymize_company=True, blur_year=False)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "Energy" in prompt

    # --- Year blurring tests -------------------------------------------------

    def test_blur_year_removes_calendar_year_from_cutoff(self):
        cfg = BenchmarkConfig(anonymize_company=False, blur_year=True)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "2020-06-01" not in prompt
        assert "Y0-06-01" in prompt

    def test_blur_year_replaces_end_date_year(self):
        cfg = BenchmarkConfig(prediction_horizon_months=6,
                              anonymize_company=False, blur_year=True)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "2020-12-01" not in prompt
        assert "Y0-12-01" in prompt

    def test_blur_year_preserves_horizon_integer(self):
        """The '6 months' integer must not be touched."""
        cfg = BenchmarkConfig(prediction_horizon_months=6,
                              anonymize_company=False, blur_year=True)
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "6 months" in prompt

    # --- Default config (both flags ON) tests --------------------------------

    def test_default_cfg_blurs_and_anonymizes(self):
        cfg = BenchmarkConfig()   # anonymize_company=True, blur_year=True
        case = self._make_case()
        data = self._make_data()
        prompt = build_prompt(data, case, cfg)
        assert "Test Corp" not in prompt
        assert "TEST US" not in prompt
        assert "2020-06-01" not in prompt
        assert "Y0-06-01" in prompt

    def test_prompt_with_fundamentals_blurs_fiscal_periods(self):
        """Fiscal periods like 2019A, 2020A in income statement are blurred."""
        cfg = BenchmarkConfig(anonymize_company=False, blur_year=True)
        case = self._make_case()
        data = self._make_data()
        data["fundamentals"]["income_statement"] = [
            {"FISCAL_YEAR_PERIOD": "2019A", "SALES_REV_TURN": 500e6,
             "EBIT": 50e6, "EBITDA": 80e6, "NET_INCOME": 30e6},
            {"FISCAL_YEAR_PERIOD": "2020A", "SALES_REV_TURN": 450e6,
             "EBIT": 30e6, "EBITDA": 60e6, "NET_INCOME": 10e6},
        ]
        prompt = build_prompt(data, case, cfg)
        assert "2019A" not in prompt
        assert "2020A" not in prompt
        assert "Y-1A" in prompt
        assert "Y0A" in prompt

    def test_prompt_with_macro_preserves_levels(self):
        """VIX, yield levels must survive year blurring."""
        cfg = BenchmarkConfig(anonymize_company=False, blur_year=True)
        case = self._make_case()
        data = self._make_data()
        data["macro"] = {
            "us": {
                "date": "2020-06-01",
                "sp500": 3100.0,
                "vix": 28.1,
                "unemployment": 13.3,
                "cpi": 257.0,
            },
            "bond_yields": {
                "date": "2020-06-01",
                "us_2y": 0.17,
                "us_5y": 0.37,
                "us_10y": 0.65,
                "us_30y": 1.41,
            },
        }
        prompt = build_prompt(data, case, cfg)
        # Macro levels preserved  (_fmt formats 3100 as "3,100" with comma)
        assert "28.1" in prompt
        assert "0.65" in prompt
        assert "3,100" in prompt
        # Date year blurred
        assert "2020-06-01" not in prompt
        assert "Y0-06-01" in prompt


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
