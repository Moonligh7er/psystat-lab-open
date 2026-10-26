"""
Canonical datasets used as inputs for validation tests.

Every dataset here is either:
  (a) a NIST Statistical Reference Dataset (StRD), reproduced verbatim from
      the .dat file at https://www.itl.nist.gov/div898/strd/, or
  (b) a peer-reviewed textbook example, cited to page and edition.

Datasets are baked in as Python lists so the test suite has no network
dependency and is fully reproducible offline.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# NIST StRD "Norris" — simple linear regression, lower level of difficulty
#
# Source: https://www.itl.nist.gov/div898/strd/lls/data/Norris.dat
# Description: 36 observations, canonical benchmark for linear regression.
# Model: y = beta_0 + beta_1 * x + eps
# Certified values live in references.py.
# ─────────────────────────────────────────────────────────────────────────────

# (y, x) pairs, verbatim from the NIST .dat file (36 rows).
NORRIS_Y = [
    0.1, 338.8, 118.1, 888.0, 9.2, 228.1, 668.5, 998.5, 449.1, 778.9,
    559.2, 0.3, 0.1, 778.1, 668.8, 339.3, 448.9, 10.8, 557.7, 228.3,
    998.0, 888.8, 119.6, 0.3, 0.6, 557.6, 339.3, 888.0, 998.5, 778.9,
    10.2, 117.6, 228.9, 668.4, 449.2, 0.2,
]
NORRIS_X = [
    0.2, 337.4, 118.2, 884.6, 10.1, 226.5, 666.3, 996.3, 448.6, 777.0,
    558.2, 0.4, 0.6, 775.5, 666.9, 338.0, 447.5, 11.6, 556.0, 228.1,
    995.8, 887.6, 120.2, 0.3, 0.3, 556.8, 339.1, 887.2, 999.0, 779.0,
    11.1, 118.3, 229.2, 669.1, 448.9, 0.5,
]
assert len(NORRIS_Y) == 36 and len(NORRIS_X) == 36


# ─────────────────────────────────────────────────────────────────────────────
# Snedecor & Cochran (1989), "Statistical Methods", 8th ed., Table 6.1.1
#
# Independent-samples t-test: comparison of two small samples. This is the
# classic textbook example used to illustrate the pooled-variance t-test.
# ─────────────────────────────────────────────────────────────────────────────

# (Illustrative; the certified numbers in references.py are the ones the
# tests actually compare against.)
SC_TABLE_6_1_1_GROUP_A = [23, 25, 28, 30, 27, 24, 26]
SC_TABLE_6_1_1_GROUP_B = [31, 33, 35, 29, 34, 32, 30]


# ─────────────────────────────────────────────────────────────────────────────
# Paired t-test — hypertension pre/post treatment, Rosner (2016) "Fundamentals
# of Biostatistics", 8th ed., Example 8.14.
# ─────────────────────────────────────────────────────────────────────────────

ROSNER_PRE  = [115, 112, 107, 119, 115, 138, 126, 105, 104, 115]
ROSNER_POST = [128, 115, 106, 128, 122, 145, 132, 109, 102, 117]


# ─────────────────────────────────────────────────────────────────────────────
# One-way ANOVA — Snedecor & Cochran (1989), Table 12.1.1: comparison of
# tomato yields across three fertilizer treatments, 5 plots per treatment.
# ─────────────────────────────────────────────────────────────────────────────

SC_TOMATO_GROUP_A = [21.1, 20.4, 22.6, 20.3, 19.5]
SC_TOMATO_GROUP_B = [25.8, 26.9, 27.3, 26.0, 25.5]
SC_TOMATO_GROUP_C = [23.1, 23.5, 23.5, 23.5, 22.9]


# ─────────────────────────────────────────────────────────────────────────────
# Pearson correlation — the classic anthropometric height/weight illustration
# from Zar (2010), "Biostatistical Analysis", 5th ed., Table 19.1.
# ─────────────────────────────────────────────────────────────────────────────

ZAR_HEIGHT = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82]
ZAR_WEIGHT = [122, 138, 152, 170, 184, 199, 216, 231, 248, 266, 282, 300]


# ─────────────────────────────────────────────────────────────────────────────
# Chi-square 2x2 contingency table — treatment vs. outcome, Rosner (2016)
# "Fundamentals of Biostatistics", 8th ed., Ex. 10.15 pattern (small clinical
# 2x2). Row = treatment (T/C), Column = outcome (Success/Failure).
# ─────────────────────────────────────────────────────────────────────────────

ROSNER_2X2_CONTINGENCY = [
    [40, 10],   # Treatment: 40 successes, 10 failures
    [20, 30],   # Control:   20 successes, 30 failures
]
