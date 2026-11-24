"""Standard error of Cohen's d in app/routers/effect_size.py.

_se_d used to return sqrt((n1+n2)/(n1 n2) + d^2/2) instead of
sqrt((n1+n2)/(n1 n2) + d^2/(2(n1+n2))), roughly doubling the SE for moderate d.
"""

import math

import pytest

from app.routers import effect_size as E
from app.routers import meta_analysis as M


def test_se_d_matches_borenstein_worked_example():
    # Borenstein, Hedges, Higgins & Rothstein (2009), Introduction to Meta-Analysis, ch. 4:
    # means 103 vs 100, SDs 5.5 and 4.5, n = 50 per group -> d = 0.5970, V_d = 0.0418, SE_d = 0.2045,
    # J = 0.9923, g = 0.5924, SE_g = 0.2029.
    r = E.from_statistics(E.FromStatisticsRequest(
        stat_type="means", mean1=103, mean2=100, sd1=5.5, sd2=4.5, n1=50, n2=50))
    assert r["cohens_d"]["value"] == pytest.approx(0.5970, abs=5e-4)
    assert r["cohens_d"]["se"] ** 2 == pytest.approx(0.0418, abs=5e-4)
    assert r["cohens_d"]["se"] == pytest.approx(0.2045, abs=5e-4)
    assert r["hedges_g"]["value"] == pytest.approx(0.5924, abs=5e-4)
    assert r["hedges_g"]["se"] == pytest.approx(0.2029, abs=5e-4)


@pytest.mark.parametrize("d,n1,n2", [(0.0, 10, 10), (0.5, 20, 25), (1.2, 40, 15), (-0.8, 100, 100)])
def test_se_d_closed_form(d, n1, n2):
    assert E._se_d(d, n1, n2) == pytest.approx(math.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2))))


def test_se_d_agrees_with_meta_analysis_module():
    # Same summary data through both routers must give the same SE for d
    # (meta_analysis rounds its output to 6 decimals).
    es = E.from_statistics(E.FromStatisticsRequest(
        stat_type="means", mean1=10.0, mean2=8.0, sd1=2.0, sd2=2.5, n1=20, n2=25))
    ma = M.convert_effect_sizes(M.EffectSizeRequest(means=[10.0, 8.0], sds=[2.0, 2.5], ns=[20, 25]))
    assert es["cohens_d"]["value"] == pytest.approx(ma["cohens_d"]["estimate"], abs=1e-6)
    assert es["cohens_d"]["se"] == pytest.approx(ma["cohens_d"]["se"], abs=1e-6)


def test_se_d_is_consistent_with_its_own_confidence_interval():
    # The non-central t CI for d is close to d +/- 1.96 SE for moderate samples;
    # the old SE was about twice the CI half-width.
    r = E.from_statistics(E.FromStatisticsRequest(
        stat_type="means", mean1=5.2, mean2=4.5, sd1=1.1, sd2=1.2, n1=30, n2=30))
    ci = r["cohens_d"]["ci"]
    half_width = (ci["upper"] - ci["lower"]) / 2
    assert 1.96 * r["cohens_d"]["se"] == pytest.approx(half_width, rel=0.05)
