"""Marginal maximum-likelihood IRT engine (Bock & Aitkin, 1981).

Unidimensional Rasch, 2PL, 3PL and graded response models. The latent trait is
N(0, 1) (Rasch: N(0, sigma^2) with sigma estimated, items fixed at slope 1, as
mirt's itemtype = "Rasch"). Identical response patterns are collapsed, the marginal
likelihood is integrated with Gauss-Hermite quadrature and maximised directly with
L-BFGS using analytic gradients (Fisher identity). Standard errors come from a
finite-difference Hessian of the marginal log-likelihood (observed information).
The 3PL guessing parameters get a Beta prior (posterior-mode estimates, as mirt's
"expbeta" prior); its SEs come from the Hessian of that penalised objective.
Person scores are EAP with posterior SDs. Reference values: tests/test_irt.py (mirt).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta as beta_dist

N_QUAD = 61


def _quadrature(n=N_QUAD):
    x, w = np.polynomial.hermite.hermgauss(n)
    return x * math.sqrt(2.0), w / math.sqrt(math.pi)


def _patterns(X):
    pats, idx, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return pats, counts.astype(float), idx.ravel()


class _Dichotomous:
    """Rasch / 2PL / 3PL. Parameter vector per model:
    rasch: [b_1..b_J, log sigma];  2pl: [a_1..a_J, b_1..b_J];  3pl: 2pl + [logit c_1..c_J]."""

    def __init__(self, X, model, c_prior=None):
        self.model = model
        self.pats, self.counts, self.idx = _patterns(X)
        self.J = X.shape[1]
        self.nodes, self.w = _quadrature()
        self.c_prior = c_prior  # (alpha, beta) Beta prior for the guessing parameters

    def unpack(self, th):
        J = self.J
        if self.model == "rasch":
            return np.ones(J), th[:J], np.zeros(J), math.exp(th[J])
        a, b = th[:J], th[J:2 * J]
        c = expit(th[2 * J:3 * J]) if self.model == "3pl" else np.zeros(J)
        return a, b, c, 1.0

    def _prob(self, th):
        a, b, c, sigma = self.unpack(th)
        theta = sigma * self.nodes
        z = a[None, :] * (theta[:, None] - b[None, :])           # Q x J
        s = expit(z)
        P = c[None, :] + (1 - c[None, :]) * s
        return np.clip(P, 1e-12, 1 - 1e-12), s, theta

    def _posterior(self, P):
        L = self.pats @ np.log(P).T + (1 - self.pats) @ np.log1p(-P).T  # patterns x Q
        L += np.log(self.w)[None, :]
        mx = L.max(axis=1, keepdims=True)
        E = np.exp(L - mx)
        m = E.sum(axis=1, keepdims=True)
        return E / m, (np.log(m) + mx).ravel()

    def negll_grad(self, th):
        P, s, theta = self._prob(th)
        post, logm = self._posterior(P)
        ll = float(self.counts @ logm)
        nw = post * self.counts[:, None]                      # weighted posterior, patterns x Q
        N = nw.sum(axis=0)                                    # expected persons at each node
        R = nw.T @ self.pats                                  # Q x J expected correct
        a, b, c, sigma = self.unpack(th)
        resid = (R - N[:, None] * P) / (P * (1 - P))          # dLL/dP
        dPdz = (1 - c[None, :]) * s * (1 - s)
        g_z = resid * dPdz                                    # dLL/dz
        J = self.J
        if self.model == "rasch":
            gb = -(g_z).sum(axis=0)
            gs = float((g_z * (sigma * self.nodes)[:, None]).sum())
            grad = np.concatenate([gb, [gs]])
        else:
            ga = (g_z * (theta[:, None] - b[None, :])).sum(axis=0)
            gb = -(g_z * a[None, :]).sum(axis=0)
            grad = np.concatenate([ga, gb])
            if self.model == "3pl":
                gc = (resid * (1 - s)).sum(axis=0) * c * (1 - c)
                al, be = self.c_prior
                ll += float(np.sum((al - 1) * np.log(c) + (be - 1) * np.log1p(-c)))
                gc += (al - 1) * (1 - c) - (be - 1) * c
                grad = np.concatenate([grad, gc])
        return -ll, -grad

    def start(self):
        p = np.clip(self.pats.T @ self.counts / self.counts.sum(), 0.02, 0.98)
        b0 = -np.log(p / (1 - p))
        if self.model == "rasch":
            return np.concatenate([b0, [0.0]])
        th = np.concatenate([np.ones(self.J), b0 / 1.7])
        if self.model == "3pl":
            al, be = self.c_prior
            th = np.concatenate([th, np.full(self.J, math.log(al / be))])
        return th


class _Graded:
    """Samejima graded response model. Per item: a and thresholds b_1 < ... < b_{K-1},
    parametrised as [a, b_1, log(b_2 - b_1), ...] so the order is always respected."""

    def __init__(self, X):
        self.pats, self.counts, self.idx = _patterns(X)
        self.J = X.shape[1]
        self.K = [int(X[:, j].max()) + 1 for j in range(self.J)]
        self.nodes, self.w = _quadrature()
        self.offsets = np.cumsum([0] + [k for k in self.K])  # K params per item (a + K-1 thresholds)

    def unpack_item(self, th, j):
        seg = th[self.offsets[j]:self.offsets[j + 1]]
        a = seg[0]
        b = np.cumsum(np.concatenate([[seg[1]], np.exp(seg[2:])]))
        return a, b

    def item_probs(self, th, j, theta=None):
        theta = self.nodes if theta is None else theta
        a, b = self.unpack_item(th, j)
        F = expit(a * (theta[:, None] - b[None, :]))                    # Q x (K-1)
        Fp = np.concatenate([np.ones((len(theta), 1)), F, np.zeros((len(theta), 1))], axis=1)
        return np.clip(Fp[:, :-1] - Fp[:, 1:], 1e-12, 1.0), F, a, b      # Q x K

    def negll_grad(self, th):
        Q = len(self.nodes)
        L = np.tile(np.log(self.w), (len(self.pats), 1))
        cache = []
        for j in range(self.J):
            P, F, a, b = self.item_probs(th, j)
            L += np.log(P[:, self.pats[:, j].astype(int)]).T
            cache.append((P, F, a, b))
        mx = L.max(axis=1, keepdims=True)
        E = np.exp(L - mx)
        m = E.sum(axis=1, keepdims=True)
        post = E / m
        ll = float(self.counts @ (np.log(m) + mx).ravel())
        nw = post * self.counts[:, None]
        grad = np.zeros_like(th)
        for j in range(self.J):
            P, F, a, b = cache[j]
            K = self.K[j]
            onehot = np.eye(K)[self.pats[:, j].astype(int)]          # patterns x K
            Rk = nw.T @ onehot                                       # Q x K expected counts
            dLdP = Rk / P                                            # Q x K
            # P_k = F_k - F_{k+1} (F_0 = 1, F_K = 0); F_k for k = 1..K-1 is column k-1 of F
            dLdF = dLdP[:, 1:] - dLdP[:, :-1]                        # Q x (K-1): d/dF_k
            dF = F * (1 - F)
            ga = float((dLdF * dF * (self.nodes[:, None] - b[None, :])).sum())
            gb = -(dLdF * dF * a).sum(axis=0)                        # d/db_k
            # chain rule to [b_1, log d_2, ...]: b_k = b_1 + sum_{i<=k} d_i
            seg = th[self.offsets[j]:self.offsets[j + 1]]
            g_raw = np.zeros(K)
            g_raw[0] = ga
            g_raw[1] = gb.sum()
            for i in range(2, K):
                g_raw[i] = gb[i - 1:].sum() * math.exp(seg[i])
            grad[self.offsets[j]:self.offsets[j + 1]] = g_raw
        return -ll, -grad

    def start(self):
        th = []
        tot = self.counts.sum()
        for j in range(self.J):
            K = self.K[j]
            freq = np.array([self.counts[self.pats[:, j] == k].sum() for k in range(K)]) / tot
            cum = np.clip(1 - np.cumsum(freq)[:-1], 0.02, 0.98)              # P(X >= k), k = 1..K-1
            b = -np.log(cum / (1 - cum)) / 1.7
            b = np.maximum.accumulate(b + np.arange(K - 1) * 1e-3)
            th += [1.0, b[0]] + list(np.log(np.maximum(np.diff(b), 0.05)))
        return np.array(th)


class _PartialCredit:
    """Masters' partial credit model (Rasch family): slope 1, theta ~ N(0, sigma^2) with sigma
    estimated (as mirt's itemtype = "Rasch" for polytomous items). P(X = k) is proportional to
    exp(sum_{l <= k} (theta - delta_l)). Parameters: all step difficulties, then log sigma."""

    def __init__(self, X):
        self.pats, self.counts, self.idx = _patterns(X)
        self.J = X.shape[1]
        self.K = [int(X[:, j].max()) + 1 for j in range(self.J)]
        self.nodes, self.w = _quadrature()
        self.offsets = np.cumsum([0] + [k - 1 for k in self.K])

    def item_probs(self, th, j, theta):
        d = th[self.offsets[j]:self.offsets[j + 1]]
        z = np.concatenate([np.zeros((len(theta), 1)),
                            np.cumsum(theta[:, None] - d[None, :], axis=1)], axis=1)   # Q x K
        z -= z.max(axis=1, keepdims=True)
        E = np.exp(z)
        return np.clip(E / E.sum(axis=1, keepdims=True), 1e-300, 1.0)

    def _posterior(self, th, theta):
        L = np.tile(np.log(self.w), (len(self.pats), 1))
        probs = []
        for j in range(self.J):
            P = self.item_probs(th, j, theta)
            L += np.log(P[:, self.pats[:, j].astype(int)]).T
            probs.append(P)
        mx = L.max(axis=1, keepdims=True)
        E = np.exp(L - mx)
        m = E.sum(axis=1, keepdims=True)
        return E / m, (np.log(m) + mx).ravel(), probs

    def negll_grad(self, th):
        sigma = math.exp(th[-1])
        theta = sigma * self.nodes
        post, logm, probs = self._posterior(th, theta)
        ll = float(self.counts @ logm)
        nw = post * self.counts[:, None]
        N = nw.sum(axis=0)
        grad = np.zeros_like(th)
        g_theta = np.zeros(len(theta))
        for j in range(self.J):
            P = probs[j]
            K = self.K[j]
            Rk = nw.T @ np.eye(K)[self.pats[:, j].astype(int)]              # Q x K expected counts
            # d log P_k / d delta_l = -[l <= k] + P(X >= l)
            obs_ge = np.cumsum(Rk[:, ::-1], axis=1)[:, ::-1][:, 1:]          # Q x (K-1): count(X >= l)
            exp_ge = np.cumsum(P[:, ::-1], axis=1)[:, ::-1][:, 1:] * N[:, None]
            grad[self.offsets[j]:self.offsets[j + 1]] = (exp_ge - obs_ge).sum(axis=0)
            k = np.arange(K)
            g_theta += Rk @ k - N * (P @ k)                                  # d/dtheta: k - E[k]
        grad[-1] = float(g_theta @ theta)                                    # dtheta / dlog sigma = theta
        return -ll, -grad

    def start(self):
        th = []
        tot = self.counts.sum()
        for j in range(self.J):
            K = self.K[j]
            freq = np.array([self.counts[self.pats[:, j] == k].sum() for k in range(K)]) / tot
            freq = np.clip(freq, 1e-3, None)
            th += list(np.log(freq[:-1] / freq[1:]))
        return np.array(th + [0.0])


def _hessian(f_grad, th, eps=1e-4):
    n = len(th)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps * max(1.0, abs(th[i]))
        H[:, i] = (f_grad(th + e)[1] - f_grad(th - e)[1]) / (2 * e[i])
    return (H + H.T) / 2


def _fit(model):
    res = minimize(model.negll_grad, model.start(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "gtol": 1e-7, "ftol": 1e-13})
    th = res.x
    try:
        cov = np.linalg.inv(_hessian(model.negll_grad, th))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.full((len(th), len(th)), np.nan)
        se = np.full(len(th), np.nan)
    model.cov = cov
    return th, se, -res.fun, res


def _eap(post_fn, nodes):
    post, _ = post_fn
    eap = post @ nodes
    psd = np.sqrt(np.clip(post @ nodes ** 2 - eap ** 2, 0, None))
    return eap, psd


def fit_dichotomous(X: np.ndarray, model: str, c_prior: tuple[float, float] = (5.0, 17.0)) -> dict:
    """Fit a Rasch, 2PL or 3PL model by MML. Returns parameters, SEs, fit and EAP scores."""
    m = _Dichotomous(X, model, c_prior if model == "3pl" else None)
    th, se, ll, res = _fit(m)
    a, b, c, sigma = m.unpack(th)
    log_prior = None
    if model == "3pl":
        # The optimiser maximised LL + log prior (posterior mode). Report the marginal LL itself,
        # as mirt does, with the normalised Beta log density alongside.
        al, be = c_prior
        ll -= float(np.sum((al - 1) * np.log(c) + (be - 1) * np.log1p(-c)))
        log_prior = float(np.sum(beta_dist.logpdf(c, al, be)))
    J = m.J
    P, _, theta = m._prob(th)
    post, _ = m._posterior(P)
    eap_p = post @ theta
    psd_p = np.sqrt(np.clip(post @ theta ** 2 - eap_p ** 2, 0, None))
    eap, psd = eap_p[m.idx], psd_p[m.idx]

    if model == "rasch":
        se_a, se_b, se_c = np.zeros(J), se[:J], np.full(J, np.nan)
        cov_ab = np.zeros(J)
        n_par = J + 1
        sigma_se = sigma * se[J]
    else:
        se_a, se_b = se[:J], se[J:2 * J]
        se_c = se[2 * J:3 * J] * c * (1 - c) if model == "3pl" else np.full(J, np.nan)
        cov_ab = np.array([m.cov[j, J + j] for j in range(J)])
        n_par = 3 * J if model == "3pl" else 2 * J
        sigma_se = None
    n = len(X)
    grid = np.linspace(-4, 4, 81)
    s = expit(a[None, :] * (sigma * 0 + grid[:, None] - b[None, :]))
    Pg = c[None, :] + (1 - c[None, :]) * s
    info_items = (a[None, :] ** 2) * ((Pg - c[None, :]) ** 2 / (1 - c[None, :]) ** 2) * (1 - Pg) / np.clip(Pg, 1e-12, None)
    rel = float(np.var(eap, ddof=1) / (np.var(eap, ddof=1) + np.mean(psd ** 2))) if np.var(eap) > 0 else 0.0
    return {
        "a": a, "b": b, "c": c, "se_a": se_a, "se_b": se_b, "se_c": se_c, "cov_ab": cov_ab,
        "latent_sd": sigma, "latent_sd_se": sigma_se,
        "log_likelihood": ll, "log_prior": log_prior, "n_parameters": n_par,
        "aic": -2 * ll + 2 * n_par, "bic": -2 * ll + math.log(n) * n_par,
        "converged": bool(res.success), "n_iterations": int(res.nit),
        "theta_eap": eap, "theta_psd": psd, "empirical_reliability": rel,
        "info_grid": grid, "item_information": info_items, "test_information": info_items.sum(axis=1),
    }


def fit_graded(X: np.ndarray) -> dict:
    """Fit Samejima's graded response model by MML."""
    m = _Graded(X)
    th, se, ll, res = _fit(m)
    items = []
    for j in range(m.J):
        a, b = m.unpack_item(th, j)
        items.append({"a": float(a), "b": [float(v) for v in b],
                      "se_a": float(se[m.offsets[j]])})
    # EAP
    L = np.tile(np.log(m.w), (len(m.pats), 1))
    for j in range(m.J):
        P, _, _, _ = m.item_probs(th, j)
        L += np.log(P[:, m.pats[:, j].astype(int)]).T
    mx = L.max(axis=1, keepdims=True)
    post = np.exp(L - mx)
    post /= post.sum(axis=1, keepdims=True)
    eap_p = post @ m.nodes
    psd_p = np.sqrt(np.clip(post @ m.nodes ** 2 - eap_p ** 2, 0, None))
    eap, psd = eap_p[m.idx], psd_p[m.idx]
    n_par = int(sum(m.K))
    n = len(X)
    grid = np.linspace(-4, 4, 81)
    info = np.zeros((len(grid), m.J))
    for j in range(m.J):
        P, F, a, b = m.item_probs(th, j, grid)
        Fp = np.concatenate([np.ones((len(grid), 1)), F, np.zeros((len(grid), 1))], axis=1)
        dF = a * Fp * (1 - Fp)
        dP = dF[:, :-1] - dF[:, 1:]
        info[:, j] = (dP ** 2 / P).sum(axis=1)
    rel = float(np.var(eap, ddof=1) / (np.var(eap, ddof=1) + np.mean(psd ** 2))) if np.var(eap) > 0 else 0.0
    return {"items": items, "log_likelihood": ll, "n_parameters": n_par,
            "aic": -2 * ll + 2 * n_par, "bic": -2 * ll + math.log(n) * n_par,
            "converged": bool(res.success), "n_iterations": int(res.nit),
            "theta_eap": eap, "theta_psd": psd, "empirical_reliability": rel,
            "info_grid": grid, "item_information": info, "test_information": info.sum(axis=1)}


def fit_partial_credit(X: np.ndarray) -> dict:
    """Fit the partial credit model by MML (latent SD estimated)."""
    m = _PartialCredit(X)
    th, se, ll, res = _fit(m)
    sigma = math.exp(th[-1])
    items = [{"steps": [float(v) for v in th[m.offsets[j]:m.offsets[j + 1]]],
              "se_steps": [float(v) for v in se[m.offsets[j]:m.offsets[j + 1]]]} for j in range(m.J)]
    theta = sigma * m.nodes
    post, _, _ = m._posterior(th, theta)
    eap_p = post @ theta
    psd_p = np.sqrt(np.clip(post @ theta ** 2 - eap_p ** 2, 0, None))
    eap, psd = eap_p[m.idx], psd_p[m.idx]
    grid = np.linspace(-4, 4, 81)
    info = np.zeros((len(grid), m.J))
    curves = []
    k_all = []
    for j in range(m.J):
        P = m.item_probs(th, j, grid)
        k = np.arange(m.K[j])
        info[:, j] = P @ k ** 2 - (P @ k) ** 2          # Rasch family: information = Var(X | theta)
        curves.append(P)
        k_all.append(k)
    # infit / outfit mean squares from EAP abilities
    infit, outfit = [], []
    Xi = X.astype(int)
    for j in range(m.J):
        P = m.item_probs(th, j, eap)
        k = np.arange(m.K[j])
        ex = P @ k
        var = np.clip(P @ k ** 2 - ex ** 2, 1e-12, None)
        r2 = (Xi[:, j] - ex) ** 2
        infit.append(float(r2.sum() / var.sum()))
        outfit.append(float((r2 / var).mean()))
    n_par = int(sum(k - 1 for k in m.K) + 1)
    n = len(X)
    rel = float(np.var(eap, ddof=1) / (np.var(eap, ddof=1) + np.mean(psd ** 2))) if np.var(eap) > 0 else 0.0
    return {"items": items, "latent_sd": sigma, "log_likelihood": ll, "n_parameters": n_par,
            "aic": -2 * ll + 2 * n_par, "bic": -2 * ll + math.log(n) * n_par,
            "converged": bool(res.success), "n_iterations": int(res.nit),
            "theta_eap": eap, "theta_psd": psd, "empirical_reliability": rel,
            "infit": infit, "outfit": outfit,
            "info_grid": grid, "item_information": info, "test_information": info.sum(axis=1),
            "category_curves": curves}
