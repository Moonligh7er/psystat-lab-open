"""Genomics — SNP risk analysis, Hardy-Weinberg, allele frequency, polygenic scoring."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy import stats as sp_stats

router = APIRouter()


class SNPAnalysisRequest(BaseModel):
    genotypes: list[str]  # e.g., ["AA", "AG", "GG", "AG", ...]
    allele_risk: str = ""  # the risk allele, e.g. "G"


class HardyWeinbergRequest(BaseModel):
    genotype_counts: dict[str, int]  # e.g., {"AA": 50, "AG": 40, "GG": 10}


class PolygenicScoreRequest(BaseModel):
    snps: list[dict]  # [{genotype: "AG", risk_allele: "G", effect_size: 0.3, weight: 1.0}, ...]


class GWASSimRequest(BaseModel):
    """Simulate a basic case-control allele frequency comparison."""
    case_allele_counts: list[int]    # [count_A, count_B] in cases
    control_allele_counts: list[int]  # [count_A, count_B] in controls


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/snp-analysis")
def snp_analysis(req: SNPAnalysisRequest):
    """Analyze genotype frequencies and allele counts for a single SNP."""
    genotypes = req.genotypes
    if len(genotypes) < 2:
        raise HTTPException(400, "Need at least 2 genotypes.")

    # Count genotypes
    genotype_counts: dict[str, int] = {}
    for g in genotypes:
        key = "".join(sorted(g.upper()))
        genotype_counts[key] = genotype_counts.get(key, 0) + 1

    n = len(genotypes)

    # Allele frequencies
    allele_counts: dict[str, int] = {}
    for g in genotypes:
        for allele in g.upper():
            allele_counts[allele] = allele_counts.get(allele, 0) + 1

    total_alleles = sum(allele_counts.values())
    allele_freqs = {a: c / total_alleles for a, c in allele_counts.items()}

    # Risk allele dosage
    risk_info = None
    if req.allele_risk:
        risk = req.allele_risk.upper()
        dosages = [g.upper().count(risk) for g in genotypes]
        risk_info = {
            "risk_allele": risk,
            "frequency": allele_freqs.get(risk, 0),
            "mean_dosage": float(np.mean(dosages)),
            "dosage_distribution": {
                "0_copies": dosages.count(0),
                "1_copy": dosages.count(1),
                "2_copies": dosages.count(2),
            },
        }

    return {
        "n_samples": n,
        "genotype_counts": genotype_counts,
        "genotype_frequencies": {k: v / n for k, v in genotype_counts.items()},
        "allele_counts": allele_counts,
        "allele_frequencies": allele_freqs,
        "risk_analysis": risk_info,
    }


@router.post("/hardy-weinberg")
def hardy_weinberg_test(req: HardyWeinbergRequest):
    """Hardy-Weinberg Equilibrium chi-square test."""
    counts = req.genotype_counts
    genotype_keys = list(counts.keys())
    if len(genotype_keys) != 3:
        raise HTTPException(400, "Provide exactly 3 genotype counts (e.g., AA, AG, GG).")

    # Sort to identify homozygotes and heterozygote
    sorted_keys = sorted(genotype_keys, key=lambda k: len(set(k)))
    # Heterozygote has 2 different alleles
    het_key = [k for k in sorted_keys if len(set(k)) == 2]
    hom_keys = [k for k in sorted_keys if len(set(k)) == 1]

    if len(het_key) != 1 or len(hom_keys) != 2:
        raise HTTPException(400, "Could not identify genotype classes. Use format like AA, AG, GG.")

    n_het = counts[het_key[0]]
    n_hom1 = counts[hom_keys[0]]
    n_hom2 = counts[hom_keys[1]]
    n = n_het + n_hom1 + n_hom2

    # Allele frequencies
    allele1 = hom_keys[0][0]
    allele2 = hom_keys[1][0]
    p = (2 * n_hom1 + n_het) / (2 * n)
    q = 1 - p

    # Expected under HWE
    exp_hom1 = p**2 * n
    exp_het = 2 * p * q * n
    exp_hom2 = q**2 * n

    observed = [n_hom1, n_het, n_hom2]
    expected = [exp_hom1, exp_het, exp_hom2]

    chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected) if e > 0)
    p_value = sp_stats.chi2.sf(chi2, df=1)

    return {
        "allele_frequencies": {allele1: float(p), allele2: float(q)},
        "observed": dict(zip(genotype_keys, [counts[k] for k in genotype_keys])),
        "expected": {hom_keys[0]: float(exp_hom1), het_key[0]: float(exp_het), hom_keys[1]: float(exp_hom2)},
        "chi2": float(chi2),
        "p_value": float(p_value),
        "df": 1,
        "in_equilibrium": bool(p_value > 0.05),
        "n": n,
    }


@router.post("/polygenic-score")
def polygenic_score(req: PolygenicScoreRequest):
    """Calculate weighted polygenic risk score from multiple SNPs."""
    if len(req.snps) == 0:
        raise HTTPException(400, "Provide at least one SNP.")

    snp_scores = []
    total_score = 0.0
    max_possible = 0.0

    for snp in req.snps:
        genotype = snp.get("genotype", "")
        risk_allele = snp.get("risk_allele", "")
        effect = snp.get("effect_size", 0.0)
        weight = snp.get("weight", 1.0)

        dosage = genotype.upper().count(risk_allele.upper())
        score = dosage * effect * weight
        total_score += score
        max_possible += 2 * abs(effect) * weight

        snp_scores.append({
            "genotype": genotype,
            "risk_allele": risk_allele,
            "dosage": dosage,
            "effect_size": effect,
            "weight": weight,
            "contribution": float(score),
        })

    percentile = (total_score / max_possible * 100) if max_possible > 0 else 50

    return {
        "total_score": float(total_score),
        "max_possible_score": float(max_possible),
        "normalized_percentile": float(percentile),
        "n_snps": len(req.snps),
        "snp_details": snp_scores,
        "risk_category": (
            "High" if percentile > 75 else
            "Moderate" if percentile > 50 else
            "Average" if percentile > 25 else
            "Low"
        ),
    }


@router.post("/gwas-test")
def gwas_association_test(req: GWASSimRequest):
    """Basic allelic association test (chi-square) for case-control GWAS."""
    case = req.case_allele_counts
    control = req.control_allele_counts
    if len(case) != 2 or len(control) != 2:
        raise HTTPException(400, "Provide exactly 2 allele counts for each group.")

    table = np.array([case, control])
    chi2, p_value, dof, expected = sp_stats.chi2_contingency(table)

    # Odds ratio
    or_val = (case[0] * control[1]) / (case[1] * control[0]) if case[1] * control[0] > 0 else float("inf")
    log_or = np.log(or_val) if or_val > 0 and or_val != float("inf") else 0
    # Haldane-Anscombe continuity correction for zero cells
    se_log_or = np.sqrt(sum(1/(c + 0.5) for c in [case[0], case[1], control[0], control[1]]))

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "odds_ratio": float(or_val),
        "log_odds_ratio": float(log_or),
        "se_log_or": float(se_log_or),
        "ci_95": [float(np.exp(log_or - 1.96 * se_log_or)), float(np.exp(log_or + 1.96 * se_log_or))],
        "significant": bool(p_value < 5e-8),  # Genome-wide significance threshold
        "suggestive": bool(p_value < 1e-5),
        "observed": {"cases": case, "controls": control},
        "expected": expected.tolist(),
    }
