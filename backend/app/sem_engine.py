"""Maximum-likelihood structural equation modelling (multi-group, mean structures).

Models are written as a lavaan-style parameter table and fitted in RAM form:
for each group, with variables v = (observed, latent),

    B = (I - A)^-1,   Sigma = F B S B' F',   mu = F B m

where A holds regressions and loadings (A[to, from]), S the (residual) variances and
covariances, m the intercepts/means, and F selects the observed variables.

Estimation follows lavaan's defaults for complete data (estimator = "ML",
likelihood = "normal"): the sample covariance matrix uses denominator N, the
discrepancy is F_ML = sum_g (N_g/N) [log|Sigma_g| + tr(S_g Sigma_g^-1) - log|S_g| - p
+ (ybar_g - mu_g)' Sigma_g^-1 (ybar_g - mu_g)], chi-square = N * F_ML, and standard
errors come from the expected information matrix. Parameters sharing a label are
constrained equal (lavaan's group.equal / labels). Reference values: tests/test_sem.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy import stats as sp_stats


@dataclass
class Param:
    lhs: str
    op: str            # "=~" loading (lhs latent, rhs indicator), "~" regression (lhs <- rhs),
                       # "~~" (co)variance, "~1" intercept / mean
    rhs: str = ""
    group: int = 0
    free: bool = True
    value: float = 0.0  # start value if free, fixed value otherwise
    label: str | None = None  # equal labels => equal parameters


@dataclass
class Model:
    observed: list[str]
    latent: list[str]
    params: list[Param]
    n_groups: int = 1
    meanstructure: bool = False
    defined: dict = field(default_factory=dict)  # name -> callable(est_dict) for derived effects


def _key(p: Param) -> str:
    return p.label if p.label else f"{p.lhs}{p.op}{p.rhs}@g{p.group}"


class Fitted:
    pass


class SEM:
    def __init__(self, model: Model, data: list[np.ndarray]):
        self.m = model
        self.obs, self.lat = model.observed, model.latent
        self.vars = self.obs + self.lat
        self.idx = {v: i for i, v in enumerate(self.vars)}
        self.p, self.q = len(self.obs), len(self.lat)
        self.k = self.p + self.q
        self.G = model.n_groups
        if len(data) != self.G:
            raise ValueError("one data matrix per group is required")
        self.n = np.array([len(d) for d in data], dtype=float)
        self.N = float(self.n.sum())
        self.S = [np.cov(d, rowvar=False, bias=True) for d in data]
        self.ybar = [d.mean(axis=0) for d in data]
        # free parameter vector: one entry per distinct key among free params
        self.free_keys: list[str] = []
        seen = {}
        for prm in model.params:
            if prm.free:
                kk = _key(prm)
                if kk not in seen:
                    seen[kk] = len(self.free_keys)
                    self.free_keys.append(kk)
        self.pos = seen
        self.theta0 = np.zeros(len(self.free_keys))
        for prm in model.params:
            if prm.free:
                self.theta0[seen[_key(prm)]] = prm.value

    # ── model-implied moments ──
    def matrices(self, theta, g):
        k = self.k
        A, S, M = np.zeros((k, k)), np.zeros((k, k)), np.zeros(k)
        for prm in self.m.params:
            if prm.group != g:
                continue
            val = theta[self.pos[_key(prm)]] if prm.free else prm.value
            if prm.op == "=~":
                A[self.idx[prm.rhs], self.idx[prm.lhs]] = val
            elif prm.op == "~":
                A[self.idx[prm.lhs], self.idx[prm.rhs]] = val
            elif prm.op == "~~":
                i, j = self.idx[prm.lhs], self.idx[prm.rhs]
                S[i, j] = S[j, i] = val
            elif prm.op == "~1":
                M[self.idx[prm.lhs]] = val
        return A, S, M

    def implied(self, theta, g):
        A, S, M = self.matrices(theta, g)
        B = np.linalg.inv(np.eye(self.k) - A)
        FB = B[: self.p]
        return FB @ S @ FB.T, FB @ M, (A, S, M, B)

    def objective(self, theta):
        total = 0.0
        for g in range(self.G):
            Sig, mu, _ = self.implied(theta, g)
            sign, logdet = np.linalg.slogdet(Sig)
            if sign <= 0:
                return 1e10
            Si = np.linalg.inv(Sig)
            Fg = logdet + np.trace(self.S[g] @ Si) - np.linalg.slogdet(self.S[g])[1] - self.p
            if self.m.meanstructure:
                d = self.ybar[g] - mu
                Fg += d @ Si @ d
            total += self.n[g] / self.N * Fg
        return total

    def gradient(self, theta):
        grad = np.zeros_like(theta)
        for g in range(self.G):
            Sig, mu, (A, S, M, B) = self.implied(theta, g)
            try:
                Si = np.linalg.inv(Sig)
            except np.linalg.LinAlgError:
                return np.zeros_like(theta)
            d = self.ybar[g] - mu if self.m.meanstructure else np.zeros(self.p)
            E = Si - Si @ (self.S[g] + np.outer(d, d)) @ Si          # dF/dSigma
            FB = B[: self.p]
            GA = 2 * FB.T @ E @ FB @ S @ B.T                           # dF/dA (through Sigma)
            GS = FB.T @ E @ FB                                         # dF/dS
            if self.m.meanstructure:
                Sid = Si @ d
                GA += -2 * np.outer(FB.T @ Sid, B @ M)                 # dF/dA (through mu)
                GM = -2 * FB.T @ Sid
            else:
                GM = np.zeros(self.k)
            w = self.n[g] / self.N
            for prm in self.m.params:
                if prm.group != g or not prm.free:
                    continue
                j = self.pos[_key(prm)]
                if prm.op == "=~":
                    grad[j] += w * GA[self.idx[prm.rhs], self.idx[prm.lhs]]
                elif prm.op == "~":
                    grad[j] += w * GA[self.idx[prm.lhs], self.idx[prm.rhs]]
                elif prm.op == "~~":
                    a, b = self.idx[prm.lhs], self.idx[prm.rhs]
                    grad[j] += w * (GS[a, b] if a == b else 2 * GS[a, b])
                elif prm.op == "~1":
                    grad[j] += w * GM[self.idx[prm.lhs]]
        return grad

    def _moment_jacobian(self, theta, g, h=1e-6):
        """d[vec(Sigma); mu]/dtheta by central differences (exact enough for SEs)."""
        cols_s, cols_m = [], []
        for j in range(len(theta)):
            e = np.zeros_like(theta)
            e[j] = h * max(1.0, abs(theta[j]))
            s1, m1, _ = self.implied(theta + e, g)
            s0, m0, _ = self.implied(theta - e, g)
            cols_s.append(((s1 - s0) / (2 * e[j])).ravel())
            cols_m.append((m1 - m0) / (2 * e[j]))
        return np.array(cols_s).T, np.array(cols_m).T

    def information(self, theta):
        """Expected information (unit: whole sample), as lavaan's information = 'expected'."""
        I = np.zeros((len(theta), len(theta)))
        for g in range(self.G):
            Sig, _, _ = self.implied(theta, g)
            Si = np.linalg.inv(Sig)
            Js, Jm = self._moment_jacobian(theta, g)
            I += self.n[g] * 0.5 * Js.T @ np.kron(Si, Si) @ Js
            if self.m.meanstructure:
                I += self.n[g] * Jm.T @ Si @ Jm
        return I

    def fit(self, max_scoring=50):
        res = optimize.minimize(self.objective, self.theta0, jac=self.gradient, method="L-BFGS-B",
                                options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10})
        theta = res.x
        # Fisher scoring polish: theta += I^-1 * score (score = -N/2 * grad)
        for _ in range(max_scoring):
            g = self.gradient(theta)
            if np.max(np.abs(g)) < 1e-10:
                break
            try:
                step = np.linalg.solve(self.information(theta), -self.N / 2 * g)
            except np.linalg.LinAlgError:
                break
            t = 1.0
            f0 = self.objective(theta)
            while t > 1e-4 and self.objective(theta + t * step) > f0 + 1e-14:
                t /= 2
            if t <= 1e-4:
                break
            theta = theta + t * step
        out = Fitted()
        out.theta = theta
        out.fmin = self.objective(theta)
        out.grad_max = float(np.max(np.abs(self.gradient(theta)))) if len(theta) else 0.0
        out.converged = bool(out.grad_max < 1e-5)
        try:
            out.vcov = np.linalg.inv(self.information(theta))
        except np.linalg.LinAlgError:
            out.vcov = np.full((len(theta), len(theta)), np.nan)
        out.se = np.sqrt(np.clip(np.diag(out.vcov), 0, None))
        return out

    # ── results ──
    def n_moments(self):
        per = self.p * (self.p + 1) / 2 + (self.p if self.m.meanstructure else 0)
        return per * self.G

    def baseline_chisq(self):
        """Independence model: free variances (and means), zero covariances."""
        total = 0.0
        for g in range(self.G):
            S = self.S[g]
            total += self.n[g] * (np.sum(np.log(np.diag(S))) - np.linalg.slogdet(S)[1])
        df = self.G * self.p * (self.p - 1) / 2
        return total, df

    def fit_measures(self, fitted, fixed_x_moments=0, fixed_x_vars=()):
        npar = len(fitted.theta)
        chisq = self.N * fitted.fmin
        df = self.n_moments() - npar - fixed_x_moments
        pval = float(sp_stats.chi2.sf(chisq, df)) if df > 0 else None
        cb, dfb = self.baseline_chisq()
        cfi = (1 - max(chisq - df, 0) / max(cb - dfb, chisq - df, 0)) if max(cb - dfb, chisq - df, 0) > 0 else 1.0
        tli = ((cb / dfb - chisq / df) / (cb / dfb - 1)) if df > 0 and dfb > 0 and cb / dfb != 1 else None
        G = self.G
        rmsea = math.sqrt(max(chisq - df, 0) / (df * self.N)) * math.sqrt(G) if df > 0 else None

        def lam(p):
            if df <= 0 or sp_stats.ncx2.cdf(chisq, df, 1e-9) < p:
                return 0.0
            hi = max(chisq, 1.0)
            while sp_stats.ncx2.cdf(chisq, df, hi) > p:
                hi *= 2
            return optimize.brentq(lambda L: sp_stats.ncx2.cdf(chisq, df, L) - p, 1e-9, hi, xtol=1e-12)
        rmsea_ci = ([math.sqrt(lam(0.95) / (df * self.N)) * math.sqrt(G),
                     math.sqrt(lam(0.05) / (df * self.N)) * math.sqrt(G)] if df > 0 else None)
        rmsea_p = (float(sp_stats.ncx2.sf(chisq, df, 0.05 ** 2 * df * self.N / G)) if df > 0 else None)
        # SRMR (Bentler), lavaan's default "srmr": residuals scaled by the sample SDs
        srmr_g = []
        for g in range(G):
            Sig, mu, _ = self.implied(fitted.theta, g)
            sd = np.sqrt(np.diag(self.S[g]))
            R = (self.S[g] - Sig) / np.outer(sd, sd)
            tri = R[np.tril_indices(self.p)]
            sq = list(tri ** 2)
            if self.m.meanstructure:
                sq += list(((self.ybar[g] - mu) / sd) ** 2)
            srmr_g.append(math.sqrt(np.mean(sq)))
        srmr = float(np.sum(self.n * np.array(srmr_g)) / self.N)
        # log-likelihood (saturated & model) and information criteria
        logl = 0.0
        logl_sat = 0.0
        for g in range(G):
            Sig, mu, _ = self.implied(fitted.theta, g)
            Si = np.linalg.inv(Sig)
            d = self.ybar[g] - mu if self.m.meanstructure else np.zeros(self.p)
            logl += -self.n[g] / 2 * (self.p * math.log(2 * math.pi) + np.linalg.slogdet(Sig)[1]
                                      + np.trace(self.S[g] @ Si) + d @ Si @ d)
            logl_sat += -self.n[g] / 2 * (self.p * math.log(2 * math.pi) + np.linalg.slogdet(self.S[g])[1] + self.p)
        # With fixed.x, lavaan reports the likelihood conditional on the exogenous observed
        # variables: remove their (exactly fitted) marginal part.
        if fixed_x_vars:
            ix = [self.obs.index(v) for v in fixed_x_vars]
            for g in range(G):
                Sxx = self.S[g][np.ix_(ix, ix)]
                lx = -self.n[g] / 2 * (len(ix) * math.log(2 * math.pi) + np.linalg.slogdet(Sxx)[1] + len(ix))
                logl -= lx
                logl_sat -= lx
        npar_ic = npar
        return {
            "chi_square": float(chisq), "df": float(df), "p_value": pval,
            "cfi": float(cfi), "tli": None if tli is None else float(tli),
            "rmsea": rmsea, "rmsea_ci90": rmsea_ci, "rmsea_p_close": rmsea_p,
            "srmr": srmr,
            "baseline_chi_square": float(cb), "baseline_df": float(dfb),
            "loglik": float(logl), "loglik_saturated": float(logl_sat),
            "n_parameters": int(npar),
            "aic": float(-2 * logl + 2 * npar_ic), "bic": float(-2 * logl + math.log(self.N) * npar_ic),
            "n_observations": int(self.N),
        }

    def estimates(self, fitted):
        """Every parameter-table row with estimate, SE, z, p and the standardized (std.all) value."""
        theta, se = fitted.theta, fitted.se
        rows = []
        implied_var = []
        for g in range(self.G):
            A, S, M = self.matrices(theta, g)
            B = np.linalg.inv(np.eye(self.k) - A)
            V = B @ S @ B.T
            implied_var.append(np.sqrt(np.clip(np.diag(V), 0, None)))
        for prm in self.m.params:
            g = prm.group
            sd = implied_var[g]
            if prm.free:
                j = self.pos[_key(prm)]
                est, s = float(theta[j]), float(se[j])
                z = est / s if s > 0 else None
                pv = float(2 * sp_stats.norm.sf(abs(z))) if z is not None else None
            else:
                est, s, z, pv = float(prm.value), None, None, None
            a, b = self.idx.get(prm.lhs), self.idx.get(prm.rhs)
            if prm.op == "=~":
                std = est * sd[a] / sd[b] if sd[b] > 0 else None
            elif prm.op == "~":
                std = est * sd[b] / sd[a] if sd[a] > 0 else None
            elif prm.op == "~~":
                std = est / (sd[a] * sd[b]) if sd[a] > 0 and sd[b] > 0 else None
            else:
                std = est / sd[a] if sd[a] > 0 else None
            rows.append({"lhs": prm.lhs, "op": prm.op, "rhs": prm.rhs, "group": g + 1, "label": prm.label,
                         "free": prm.free, "est": est, "se": s, "z": z, "p_value": pv,
                         "std_all": None if std is None else float(std)})
        return rows

    def r_squared(self, fitted, g=0):
        A, S, M = self.matrices(fitted.theta, g)
        B = np.linalg.inv(np.eye(self.k) - A)
        V = B @ S @ B.T
        out = {}
        for v in self.vars:
            i = self.idx[v]
            if np.any(A[i] != 0):  # endogenous
                out[v] = float(1 - S[i, i] / V[i, i]) if V[i, i] > 0 else None
        return out

    def defined(self, fitted, fn, h=1e-6):
        """A derived parameter f(theta) with its delta-method SE (as lavaan's ':=')."""
        theta = fitted.theta
        val = fn(self, theta)
        grad = np.zeros_like(theta)
        for j in range(len(theta)):
            e = np.zeros_like(theta)
            e[j] = h * max(1.0, abs(theta[j]))
            grad[j] = (fn(self, theta + e) - fn(self, theta - e)) / (2 * e[j])
        se = float(math.sqrt(max(grad @ fitted.vcov @ grad, 0)))
        z = val / se if se > 0 else None
        return {"est": float(val), "se": se, "z": z,
                "p_value": float(2 * sp_stats.norm.sf(abs(z))) if z is not None else None}

    def value(self, theta, lhs, op, rhs, group=0):
        for prm in self.m.params:
            if prm.lhs == lhs and prm.op == op and prm.rhs == rhs and prm.group == group:
                return theta[self.pos[_key(prm)]] if prm.free else prm.value
        return 0.0


class TwoLevelSEM(SEM):
    """Two-level ML SEM for clustered data (lavaan's sem(..., cluster = )), complete data.

    Group 0 of the parameter table is the within-cluster model (no mean structure) and
    group 1 the between-cluster model (with intercepts). For cluster j of size n_j,
    y_ij = mu + b_j + w_ij with Cov(w) = Sigma_W and Cov(b) = Sigma_B, which gives the
    exact log-likelihood

        -2 logL = (N - J) log|Sigma_W| + tr(Sigma_W^-1 SSW)
                  + sum_j [ log|Omega_j| + (v_j - mu)' Omega_j^-1 (v_j - mu) + p_w log n_j ] + const,
        Omega_j = Sigma_B + diag-block(Sigma_W / n_j, 0),

    where SSW is the pooled within-cluster scatter of the variables that vary within
    clusters and v_j stacks their cluster means with the cluster-level (between-only)
    variables. Standard errors use the observed information, chi-square compares with the
    unrestricted two-level model, as lavaan. Reference values: tests/test_multilevel_sem.py.
    """

    def __init__(self, model: Model, Y: np.ndarray, clusters, within_vars: list[str]):
        self.obs = model.observed
        self.p = len(self.obs)
        self.G = 2
        self._reset_params(model)
        Y = np.asarray(Y, dtype=float)
        self.within_vars = list(within_vars)
        self.wi = [self.obs.index(v) for v in within_vars]
        self.pw, self.pz = len(self.wi), self.p - len(self.wi)
        _, inv = np.unique(np.asarray(clusters), return_inverse=True)
        sizes = np.bincount(inv)
        means = np.array([Y[inv == j].mean(axis=0) for j in range(len(sizes))])
        dev = Y[:, self.wi] - means[inv][:, self.wi]
        self.SSW = dev.T @ dev
        self.N, self.J = float(len(Y)), float(len(sizes))
        self.n = np.array([self.N, self.J])
        self.sum_log_n = float(np.sum(np.log(sizes)))
        self.by_size = []
        for s in np.unique(sizes):
            v = means[sizes == s]
            self.by_size.append((float(s), float(len(v)), v.sum(axis=0), v.T @ v))
        self.cluster_means, self.cluster_sizes = means, sizes

    def _reset_params(self, model: Model):
        self.m = model
        self.lat, self.vars = model.latent, self.obs + model.latent
        self.idx = {v: i for i, v in enumerate(self.vars)}
        self.q, self.k = len(model.latent), self.p + len(model.latent)
        self.free_keys, seen = [], {}
        for prm in model.params:
            if prm.free and _key(prm) not in seen:
                seen[_key(prm)] = len(self.free_keys)
                self.free_keys.append(_key(prm))
        self.pos = seen
        self.theta0 = np.zeros(len(self.free_keys))
        for prm in model.params:
            if prm.free:
                self.theta0[seen[_key(prm)]] = prm.value

    def _f(self, SWw, SB, mu):
        """-2 logL up to a constant, and its derivatives w.r.t. Sigma_W (within block), Sigma_B, mu."""
        sign, ldW = np.linalg.slogdet(SWw)
        if sign <= 0:
            return None
        SWi = np.linalg.inv(SWw)
        f = (self.N - self.J) * ldW + np.trace(SWi @ self.SSW)
        EW = (self.N - self.J) * SWi - SWi @ self.SSW @ SWi
        EB = np.zeros((self.p, self.p))
        gmu = np.zeros(self.p)
        wi = np.ix_(self.wi, self.wi)
        for n, c, s1, s2 in self.by_size:
            Om = SB.copy()
            Om[wi] += SWw / n
            sign, ld = np.linalg.slogdet(Om)
            if sign <= 0:
                return None
            Oi = np.linalg.inv(Om)
            D = s2 - np.outer(s1, mu) - np.outer(mu, s1) + c * np.outer(mu, mu)
            f += c * ld + np.trace(Oi @ D)
            Gm = c * Oi - Oi @ D @ Oi
            EB += Gm
            EW += Gm[wi] / n
            gmu += -2 * Oi @ (s1 - c * mu)
        return f, EW, EB, gmu

    def _levels(self, theta):
        SW, _, _ = self.implied(theta, 0)
        SB, mu, _ = self.implied(theta, 1)
        return SW[np.ix_(self.wi, self.wi)], SB, mu

    def objective(self, theta):
        out = self._f(*self._levels(theta))
        return 1e10 if out is None else out[0] / self.N

    def _chain(self, theta, g, E, gmu, grad):
        A, S, M = self.matrices(theta, g)
        B = np.linalg.inv(np.eye(self.k) - A)
        FB = B[: self.p]
        GA = 2 * FB.T @ E @ FB @ S @ B.T + np.outer(FB.T @ gmu, B @ M)
        GS = FB.T @ E @ FB
        GM = FB.T @ gmu
        for prm in self.m.params:
            if prm.group != g or not prm.free:
                continue
            j = self.pos[_key(prm)]
            if prm.op == "=~":
                grad[j] += GA[self.idx[prm.rhs], self.idx[prm.lhs]]
            elif prm.op == "~":
                grad[j] += GA[self.idx[prm.lhs], self.idx[prm.rhs]]
            elif prm.op == "~~":
                a, b = self.idx[prm.lhs], self.idx[prm.rhs]
                grad[j] += GS[a, b] if a == b else 2 * GS[a, b]
            elif prm.op == "~1":
                grad[j] += GM[self.idx[prm.lhs]]

    def gradient(self, theta):
        out = self._f(*self._levels(theta))
        grad = np.zeros_like(theta)
        if out is None:
            return grad
        _, EWw, EB, gmu = out
        EW = np.zeros((self.p, self.p))
        EW[np.ix_(self.wi, self.wi)] = EWw
        self._chain(theta, 0, EW / self.N, np.zeros(self.p), grad)
        self._chain(theta, 1, EB / self.N, gmu / self.N, grad)
        return grad

    def hessian(self, theta, h=1e-6):
        H = np.zeros((len(theta), len(theta)))
        for j in range(len(theta)):
            e = np.zeros_like(theta)
            e[j] = h * max(1.0, abs(theta[j]))
            H[:, j] = (self.gradient(theta + e) - self.gradient(theta - e)) / (2 * e[j])
        return (H + H.T) / 2

    def loglik(self, theta):
        out = self._f(*self._levels(theta))
        const = (self.N * self.pw + self.J * self.pz) * math.log(2 * math.pi) + self.pw * self.sum_log_n
        return -0.5 * (out[0] + const)

    def fit(self, max_newton=50):
        res = optimize.minimize(self.objective, self.theta0, jac=self.gradient, method="L-BFGS-B",
                                options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-11})
        theta = res.x
        for _ in range(max_newton):
            g = self.gradient(theta)
            if np.max(np.abs(g)) < 1e-11:
                break
            try:
                step = -np.linalg.solve(self.hessian(theta), g)
            except np.linalg.LinAlgError:
                break
            t, f0 = 1.0, self.objective(theta)
            while t > 1e-4 and self.objective(theta + t * step) > f0 + 1e-15:
                t /= 2
            if t <= 1e-4:
                break
            theta = theta + t * step
        out = Fitted()
        out.theta = theta
        out.fmin = self.objective(theta)
        out.grad_max = float(np.max(np.abs(self.gradient(theta)))) if len(theta) else 0.0
        out.converged = bool(out.grad_max < 1e-6)
        try:
            out.vcov = np.linalg.inv(self.N / 2 * self.hessian(theta))
        except np.linalg.LinAlgError:
            out.vcov = np.full((len(theta), len(theta)), np.nan)
        out.se = np.sqrt(np.clip(np.diag(out.vcov), 0, None))
        out.loglik = self.loglik(theta)
        return out

    def moment_starts(self):
        """Method-of-moments within / between covariance matrices and grand mean (start values)."""
        SW = self.SSW / max(self.N - self.J, 1)
        sizes = self.cluster_sizes
        n_tilde = (self.N - np.sum(sizes ** 2) / self.N) / max(self.J - 1, 1)
        V = self.cluster_means
        mu = V.mean(axis=0)
        SB = np.cov(V, rowvar=False, ddof=1) if len(V) > 1 else np.zeros((self.p, self.p))
        SB = np.atleast_2d(SB)
        SB_adj = SB.copy()
        wi = np.ix_(self.wi, self.wi)
        SB_adj[wi] = SB[wi] - SW / n_tilde
        w, U = np.linalg.eigh(SB_adj)
        floor = 0.05 * np.maximum(np.diag(SB), 1e-6).mean()
        SB_pd = (U * np.maximum(w, floor)) @ U.T
        return SW, SB_pd, mu


def two_level_reference_models(sem: TwoLevelSEM, exo_within: list[str], exo_between: list[str]):
    """The unrestricted (saturated) and baseline models lavaan compares a two-level model with.

    Saturated: Sigma_W, Sigma_B and the between-level means all free. Baseline: variances and
    means free, covariances zero except among each level's exogenous observed variables.
    """
    SW, SB, mu = sem.moment_starts()
    obs, wv = sem.obs, sem.within_vars

    def build(full: bool):
        prm = []
        for i, a in enumerate(wv):
            for b in wv[i:]:
                if full or a == b or (a in exo_within and b in exo_within):
                    prm.append(Param(a, "~~", b, 0, value=float(SW[wv.index(a), wv.index(b)]) if (full or a == b) else 0.0))
        for i, a in enumerate(obs):
            for b in obs[i:]:
                if full or a == b or (a in exo_between and b in exo_between):
                    prm.append(Param(a, "~~", b, 1, value=float(SB[i, obs.index(b)]) if (full or a == b) else 0.0))
            prm.append(Param(a, "~1", "", 1, value=float(mu[i])))
        return prm

    out = []
    for full in (True, False):
        m = Model(observed=obs, latent=[], params=build(full), n_groups=2, meanstructure=True)
        ref = TwoLevelSEM.__new__(TwoLevelSEM)
        ref.__dict__.update(sem.__dict__)  # same data summaries, new parameter table
        ref._reset_params(m)
        out.append(ref)
    return out


def two_level_fit_measures(sem: TwoLevelSEM, fitted, exo_within: list[str], exo_between: list[str]) -> dict:
    """Fit measures as lavaan reports them for a two-level model."""
    sat, base = two_level_reference_models(sem, exo_within, exo_between)
    f_sat, f_base = sat.fit(), base.fit()
    ll, ll_sat, ll_base = fitted.loglik, f_sat.loglik, f_base.loglik
    npar = len(fitted.theta)
    moments = sem.pw * (sem.pw + 1) / 2 + sem.p * (sem.p + 1) / 2 + sem.p
    chisq = max(0.0, 2 * (ll_sat - ll))
    df = moments - npar
    cb = max(0.0, 2 * (ll_sat - ll_base))
    dfb = moments - len(f_base.theta)
    N = sem.N
    cfi = 1.0
    if max(cb - dfb, chisq - df, 0) > 0:
        cfi = 1 - max(chisq - df, 0) / max(cb - dfb, chisq - df, 0)
    tli = ((cb / dfb - chisq / df) / (cb / dfb - 1)) if df > 0 and dfb > 0 and cb / dfb != 1 else None
    rmsea = math.sqrt(max(chisq - df, 0) / (df * N)) if df > 0 else None
    # SRMR per level, as lavaan: correlation residuals against the unrestricted estimates,
    # averaged over the p(p+1)/2 unique elements (the diagonal residuals are 0), no means.
    SWs, SBs, mus = sat._levels(f_sat.theta)
    SWm, SBm, mum = sem._levels(fitted.theta)

    def srmr(S, Sig, *_):
        def cor(M):
            d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
            return M / np.outer(d, d)
        e = (cor(S) - cor(Sig))[np.tril_indices(len(S), -1)]
        return math.sqrt(float(np.sum(e ** 2)) / (len(S) * (len(S) + 1) / 2))

    return {
        "chi_square": float(chisq), "df": float(df),
        "p_value": float(sp_stats.chi2.sf(chisq, df)) if df > 0 else None,
        "cfi": float(cfi), "tli": None if tli is None else float(tli), "rmsea": rmsea,
        "srmr_within": srmr(SWs, SWm), "srmr_between": srmr(SBs, SBm, mus, mum),
        "baseline_chi_square": float(cb), "baseline_df": float(dfb),
        "loglik": float(ll), "loglik_unrestricted": float(ll_sat),
        "n_parameters": int(npar),
        "aic": float(-2 * ll + 2 * npar), "bic": float(-2 * ll + math.log(N) * npar),
        "n_observations": int(N), "n_clusters": int(sem.J),
        "unrestricted": {"sigma_within": SWs, "sigma_between": SBs, "mu": mus},
    }


# ── Model builders (lavaan defaults) ─────────────────────────────────────

def cfa_params(factors: dict[str, list[str]], groups: int = 1, *, meanstructure=False,
               equal: tuple = (), S_list=None, ybar_list=None, observed=None,
               residual_covariances: list | None = None, release: dict | None = None) -> list[Param]:
    """lavaan cfa(): marker-variable identification (first loading fixed to 1), latent
    variances and covariances free, residual variances free. With a mean structure:
    intercepts free and latent means fixed to 0 (unless intercepts are constrained equal,
    then latent means are free in every group but the first, as lavaan's group.equal).
    release: {"loadings": {indicators}, "intercepts": {indicators}, "residuals": {indicators}}
    exempts those parameters from the equality constraints (lavaan's group.partial)."""
    release = release or {}
    params = []
    lat = list(factors)
    for g in range(groups):
        S = S_list[g] if S_list is not None else None
        for f, inds in factors.items():
            for i, ind in enumerate(inds):
                lab = (f"L_{f}_{ind}" if "loadings" in equal and i > 0
                       and ind not in release.get("loadings", ()) else None)
                params.append(Param(f, "=~", ind, g, free=i > 0, value=1.0, label=lab))
        for ind in observed:
            v0 = 0.5 * S[observed.index(ind), observed.index(ind)] if S is not None else 0.5
            lab = f"R_{ind}" if "residuals" in equal and ind not in release.get("residuals", ()) else None
            params.append(Param(ind, "~~", ind, g, value=v0, label=lab))
        for a, b in residual_covariances or []:
            params.append(Param(a, "~~", b, g, value=0.0))
        for i, f in enumerate(lat):
            v0 = 0.05
            if S is not None:
                v0 = max(0.05, 0.5 * S[observed.index(factors[f][0]), observed.index(factors[f][0])])
            params.append(Param(f, "~~", f, g, value=v0))
            for f2 in lat[i + 1:]:
                params.append(Param(f, "~~", f2, g, value=0.0))
        if meanstructure:
            for ind in observed:
                lab = f"I_{ind}" if "intercepts" in equal and ind not in release.get("intercepts", ()) else None
                params.append(Param(ind, "~1", "", g, value=float(ybar_list[g][observed.index(ind)]), label=lab))
            for f in lat:
                free_mean = "intercepts" in equal and g > 0
                params.append(Param(f, "~1", "", g, free=free_mean, value=0.0))
    return params


def structural_params(regressions: list[tuple[str, str]], observed: list[str], latent_factors: dict | None,
                      S: np.ndarray, residual_covariances: list | None = None,
                      group: int = 0) -> tuple[list[Param], int]:
    """lavaan sem() defaults for one group, without a mean structure.

    regressions: (outcome, predictor) pairs among observed variables and/or latents.
    Measurement part (latent_factors): marker-variable identification as in cfa_params.
    Observed exogenous variables follow fixed.x = TRUE: their variances and covariances
    are fixed at the sample values (not free parameters). Residual variances of every
    endogenous variable are free, residual covariances among pure outcomes (endogenous
    observed variables that predict nothing: lavaan's auto.cov.y) and covariances among
    exogenous latents (auto.cov.lv.x) are free.
    Returns the parameters, the number of fixed exogenous moments (subtracted from df) and the
    exogenous observed variables (for the conditional log-likelihood).
    """
    latent_factors = latent_factors or {}
    lat = list(latent_factors)
    params: list[Param] = []
    indicators = [i for inds in latent_factors.values() for i in inds]
    for f, inds in latent_factors.items():
        for i, ind in enumerate(inds):
            params.append(Param(f, "=~", ind, group, free=i > 0, value=1.0))
    outcomes = {y for y, _ in regressions}
    predictors = {x for _, x in regressions}
    for y, x in regressions:
        params.append(Param(y, "~", x, group, value=0.0))
    oi = {v: i for i, v in enumerate(observed)}
    # observed variables: indicators and endogenous get free residual variances
    exo_obs = [v for v in observed if v not in outcomes and v not in indicators]
    for v in observed:
        if v in exo_obs:
            continue
        params.append(Param(v, "~~", v, group, value=0.5 * S[oi[v], oi[v]]))
    pure_y = [v for v in observed if v in outcomes and v not in predictors and v not in indicators]
    for i, a in enumerate(pure_y):
        for b in pure_y[i + 1:]:
            params.append(Param(a, "~~", b, group, value=0.0))
    for a, b in residual_covariances or []:
        params.append(Param(a, "~~", b, group, value=0.0))
    # fixed.x: exogenous observed moments fixed at sample values
    for i, a in enumerate(exo_obs):
        for b in exo_obs[i:]:
            params.append(Param(a, "~~", b, group, free=False, value=float(S[oi[a], oi[b]])))
    n_fixed_x = len(exo_obs) * (len(exo_obs) + 1) // 2
    # latents: variances free; covariances among exogenous latents free
    exo_lat = [f for f in lat if f not in outcomes]
    for f in lat:
        v0 = max(0.05, 0.5 * S[oi[latent_factors[f][0]], oi[latent_factors[f][0]]])
        params.append(Param(f, "~~", f, group, value=v0 if f in exo_lat else 0.5 * v0))
    for i, a in enumerate(exo_lat):
        for b in exo_lat[i + 1:]:
            params.append(Param(a, "~~", b, group, value=0.0))
    return params, n_fixed_x, exo_obs


def growth_params(columns: list[str], time_scores: list[float], ybar: np.ndarray, S: np.ndarray) -> list[Param]:
    """lavaan growth(): intercept loadings 1, slope loadings = time scores, observed
    intercepts fixed to 0, growth-factor means, variances and covariance free,
    residual variances free."""
    params = []
    for c, t in zip(columns, time_scores):
        params.append(Param("i", "=~", c, free=False, value=1.0))
        params.append(Param("s", "=~", c, free=False, value=float(t)))
    for j, c in enumerate(columns):
        params.append(Param(c, "~~", c, value=0.5 * S[j, j]))
        params.append(Param(c, "~1", "", free=False, value=0.0))
    params += [Param("i", "~~", "i", value=float(np.mean(np.diag(S))) * 0.5),
               Param("s", "~~", "s", value=0.1), Param("i", "~~", "s", value=0.0),
               Param("i", "~1", "", value=float(ybar[0])),
               Param("s", "~1", "", value=float((ybar[-1] - ybar[0]) / max(time_scores[-1] - time_scores[0], 1e-9)))]
    return params
