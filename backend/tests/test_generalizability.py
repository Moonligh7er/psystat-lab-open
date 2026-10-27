"""
Generalizability theory (/api/generalizability/*) against lme4 (R 4.5.2) REML variance components.

For balanced fully crossed designs the ANOVA (expected mean squares) estimators equal REML
whenever no estimate is negative. Data (tests/data/gtheory_*.csv) are simulated with a fixed
seed: p x i (30 persons x 6 items) and p x i x o (30 persons x 5 items x 4 occasions, columns
ordered occasion by occasion).
"""

import csv
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.routers import generalizability as G

DATA = Path(__file__).parent / "data"
REF = "lme4 (R 4.5.2) REML"


def load(name):
    with open(DATA / name, newline="") as f:
        return [[float(v) for v in row] for row in csv.reader(f)]


@pytest.mark.validation
def test_p_x_i_components_match_lme4(record_validation):
    r = G.g_study(G.GStudyRequest(scores=load("gtheory_pxi.csv"), design="p_x_i"))
    for k, exp in (("person", 0.55811080), ("item", 0.22976391), ("person_x_item", 0.54320508)):
        assert r["variance_components"][k] == pytest.approx(exp, abs=1e-6)
        record_validation(method="G-study variance components (p x i)", endpoint="/api/generalizability/g-study",
                          statistic=k, observed=r["variance_components"][k], expected=exp, reference=REF,
                          reference_kind="r", dataset="simulated 30 x 6", tolerance_abs=1e-6)
    c = r["coefficients"]
    vc = r["variance_components"]
    assert c["generalizability_coefficient"] == pytest.approx(vc["person"] / (vc["person"] + vc["person_x_item"] / 6))
    assert c["dependability_coefficient"] == pytest.approx(
        vc["person"] / (vc["person"] + (vc["item"] + vc["person_x_item"]) / 6))


@pytest.mark.validation
def test_p_x_i_x_o_components_match_lme4(record_validation):
    r = G.g_study(G.GStudyRequest(scores=load("gtheory_pxixo.csv"), design="p_x_i_x_o",
                                  facet_structure={"n_occasions": 4}))
    ref = {"person": 0.59248419, "item": 0.67945991, "occasion": 0.20621016, "person_x_item": 0.11154741,
           "person_x_occasion": 0.08086584, "item_x_occasion": 0.02609151, "residual": 0.37341608}
    assert r["negative_components_set_to_zero"] == []
    for k, exp in ref.items():
        assert r["variance_components"][k] == pytest.approx(exp, abs=5e-6)  # lme4 optimiser tolerance
        record_validation(method="G-study variance components (p x i x o)", endpoint="/api/generalizability/g-study",
                          statistic=k, observed=r["variance_components"][k], expected=exp, reference=REF,
                          reference_kind="r", dataset="simulated 30 x 5 x 4", tolerance_abs=5e-6)
    d = G.d_study(G.DStudyRequest(variance_components=r["variance_components"], design="p_x_i_x_o",
                                  proposed_counts={"n_items": [5], "n_occasions": [4]}))
    assert d["d_study_table"][0]["generalizability_coefficient"] == pytest.approx(
        r["coefficients"]["generalizability_coefficient"])


def test_negative_component_is_reported_and_zeroed():
    r = G.g_study(G.GStudyRequest(scores=load("gtheory_pxixo_negative.csv"), design="p_x_i_x_o",
                                  facet_structure={"n_occasions": 3}))
    assert r["negative_components_set_to_zero"] == ["occasion"]
    assert r["variance_components"]["occasion"] == 0 and r["variance_components_raw"]["occasion"] < 0


def test_g_study_validation():
    with pytest.raises(HTTPException):  # columns not divisible into occasions
        G.g_study(G.GStudyRequest(scores=[[1, 2, 3]] * 5, design="p_x_i_x_o", facet_structure={"n_occasions": 2}))
    with pytest.raises(HTTPException):  # ragged
        G.g_study(G.GStudyRequest(scores=[[1, 2, 3], [1, 2]] * 3, design="p_x_i"))
