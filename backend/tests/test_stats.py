"""Unit tests for PsyStat Lab backend API endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Health Check ──────────────────────────────────────────────────────

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


# ── Stats Studio ──────────────────────────────────────────────────────

class TestTTest:
    def test_basic(self):
        res = client.post("/api/stats/t-test", json={
            "group1": [23, 25, 28, 30, 27, 24, 26],
            "group2": [31, 33, 35, 29, 34, 32, 30],
        })
        assert res.status_code == 200
        data = res.json()
        assert "t_statistic" in data
        assert "p_value" in data
        assert "cohens_d" in data
        assert isinstance(data["significant"], bool)

    def test_small_groups_fail(self):
        res = client.post("/api/stats/t-test", json={"group1": [1], "group2": [2]})
        assert res.status_code == 400

    def test_equal_groups(self):
        res = client.post("/api/stats/t-test", json={
            "group1": [10, 10, 10, 10],
            "group2": [10, 10, 10, 10],
        })
        assert res.status_code == 200
        data = res.json()
        # Equal groups produce NaN p-value (sanitized to None)
        assert data["p_value"] is None or data["p_value"] == 1.0


class TestPairedTTest:
    def test_basic(self):
        res = client.post("/api/stats/paired-t-test", json={
            "before": [23, 25, 28, 30, 27],
            "after": [25, 28, 30, 33, 29],
        })
        assert res.status_code == 200
        data = res.json()
        assert "mean_difference" in data
        assert data["n_pairs"] == 5

    def test_mismatched_length(self):
        res = client.post("/api/stats/paired-t-test", json={
            "before": [1, 2, 3],
            "after": [4, 5],
        })
        assert res.status_code == 400


class TestANOVA:
    def test_basic(self):
        res = client.post("/api/stats/anova", json={
            "groups": [[23, 25, 28, 30], [31, 33, 35, 29], [28, 30, 32, 34]],
        })
        assert res.status_code == 200
        data = res.json()
        assert "f_statistic" in data
        assert "eta_squared" in data
        assert "tukey_pairwise" in data

    def test_insufficient_groups(self):
        res = client.post("/api/stats/anova", json={"groups": [[1, 2, 3]]})
        assert res.status_code == 400


class TestChiSquare:
    def test_basic(self):
        res = client.post("/api/stats/chi-square", json={
            "observed": [[40, 10], [20, 30]],
        })
        assert res.status_code == 200
        data = res.json()
        assert "chi2_statistic" in data
        assert "cramers_v" in data


class TestRegression:
    def test_basic(self):
        res = client.post("/api/stats/regression", json={
            "x": [[1], [2], [3], [4], [5]],
            "y": [2.1, 4.0, 5.8, 8.2, 9.8],
        })
        assert res.status_code == 200
        data = res.json()
        assert "r_squared" in data
        assert data["r_squared"] > 0.9

    def test_too_few_observations(self):
        res = client.post("/api/stats/regression", json={"x": [[1]], "y": [2]})
        assert res.status_code == 400


class TestDescriptive:
    def test_basic(self):
        res = client.post("/api/stats/descriptive", json={"data": [1, 2, 3, 4, 5]})
        assert res.status_code == 200
        data = res.json()
        assert data["mean"] == 3.0
        assert data["n"] == 5


# ── APA Formatter ─────────────────────────────────────────────────────

class TestAPA:
    def test_ttest_format(self):
        res = client.post("/api/apa/format", json={
            "test_type": "ttest",
            "results": {
                "t_statistic": 2.31, "p_value": 0.029, "cohens_d": 0.94,
                "df": 24, "group1_mean": 25.57, "group2_mean": 32.0,
                "group1_std": 2.51, "group2_std": 2.16,
                "group1_n": 7, "group2_n": 7,
            }
        })
        assert res.status_code == 200
        data = res.json()
        assert "apa_text" in data
        assert "t(24)" in data["apa_text"]
        assert data["interpretation"] == "significant"

    def test_anova_format(self):
        res = client.post("/api/apa/format", json={
            "test_type": "anova",
            "results": {
                "f_statistic": 5.23, "p_value": 0.012, "eta_squared": 0.35,
                "df_between": 2, "df_within": 12, "group_means": [25, 30, 35],
            }
        })
        assert res.status_code == 200
        assert "F(2, 12)" in res.json()["apa_text"]


# ── Assumption Checking ───────────────────────────────────────────────

class TestAssumptions:
    def test_ttest_assumptions(self):
        res = client.post("/api/assumptions/check", json={
            "test_type": "ttest",
            "groups": [[23, 25, 28, 30, 27, 24, 26], [31, 33, 35, 29, 34, 32, 30]],
        })
        assert res.status_code == 200
        data = res.json()
        assert "checks" in data
        assert "passed_all" in data
        assert len(data["checks"]) > 0

    def test_regression_assumptions(self):
        res = client.post("/api/assumptions/check", json={
            "test_type": "regression",
            "x": [[1], [2], [3], [4], [5], [6], [7], [8]],
            "y": [2.1, 4.0, 5.8, 8.2, 9.8, 12.1, 14.0, 15.9],
        })
        assert res.status_code == 200
        data = res.json()
        assert any("Normality" in c["test"] for c in data["checks"])
        assert any("Breusch-Pagan" in c["test"] for c in data["checks"])


# ── MICE Imputation ───────────────────────────────────────────────────

class TestMICE:
    def test_basic_imputation(self):
        res = client.post("/api/mice/impute", json={
            "data": [
                {"a": 1, "b": 2}, {"a": 3, "b": None},
                {"a": 5, "b": 6}, {"a": None, "b": 8},
                {"a": 9, "b": 10}, {"a": 11, "b": 12},
            ],
            "n_imputations": 3,
            "max_iterations": 5,
        })
        assert res.status_code == 200
        data = res.json()
        assert "pooled_estimates" in data
        assert "imputed_data" in data
        assert data["n_imputations"] == 3

    def test_no_missing(self):
        res = client.post("/api/mice/impute", json={
            "data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "n_imputations": 3,
        })
        assert res.status_code == 200
        assert res.json()["n_imputations"] == 0


# ── Bayesian Advanced ─────────────────────────────────────────────────

class TestBayesianAdvanced:
    def test_mcmc_normal(self):
        res = client.post("/api/bayesian-advanced/mcmc", json={
            "data": [1.2, 2.3, 3.1, 2.8, 3.5, 2.1, 2.9, 3.3],
            "model": "normal",
            "n_samples": 500,
            "n_chains": 2,
            "burn_in": 100,
        })
        assert res.status_code == 200
        data = res.json()
        assert "parameters" in data
        assert "mu" in data["parameters"]
        assert "hdi_95" in data["parameters"]["mu"]

    def test_model_compare(self):
        res = client.post("/api/bayesian-advanced/model-compare", json={
            "data": [1.2, 2.3, 3.1, 2.8, 3.5, 2.1, 2.9, 3.3, 4.0, 3.7],
            "models": ["normal", "t"],
        })
        assert res.status_code == 200
        data = res.json()
        assert "best_model" in data
        assert len(data["models"]) == 2


# ── SEM ───────────────────────────────────────────────────────────────

class TestSEM:
    def test_path_analysis(self):
        import random
        random.seed(42)
        data = [{"x": i, "m": i * 0.5 + random.random(), "y": i * 0.3 + random.random()} for i in range(30)]
        res = client.post("/api/sem/path-analysis", json={
            "data": data,
            "paths": [{"from": "x", "to": "m"}, {"from": "m", "to": "y"}, {"from": "x", "to": "y"}],
            "mediators": ["m"],
        })
        assert res.status_code == 200
        data = res.json()
        assert "direct_effects" in data
        assert "indirect_effects" in data
        assert len(data["direct_effects"]) == 3


# ── Reports ───────────────────────────────────────────────────────────

class TestReports:
    def test_quick_report(self):
        res = client.post("/api/reports/quick", json={
            "module": "Stats Studio",
            "test_type": "t-test",
            "results": {"t_statistic": 2.31, "p_value": 0.029},
            "apa_text": "An independent-samples t-test revealed a significant difference, t(24) = 2.31, p = .029.",
        })
        assert res.status_code == 200
        data = res.json()
        assert "html" in data
        assert "t-test" in data["html"]
        assert "APA" in data["html"] or "apa" in data["html"].lower()


# ── Data Cleaning ─────────────────────────────────────────────────────

class TestDataCleaning:
    def test_outliers(self):
        res = client.post("/api/data/outliers", json={
            "data": [1, 2, 3, 4, 5, 100],
            "method": "iqr",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["n_outliers"] > 0

    def test_normality(self):
        res = client.post("/api/data/normality", json={
            "data": [1.1, 2.3, 3.2, 2.8, 3.5, 2.1, 2.9, 3.3, 4.0, 3.7],
        })
        assert res.status_code == 200
        data = res.json()
        assert "shapiro_wilk" in data
