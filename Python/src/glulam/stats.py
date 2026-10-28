"""Descriptive statistics, distribution fitting with bootstrap goodness-of-fit, characteristic values."""

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

# ----------------------------------------------------------------------------- descriptives


def q05_order(x):
    """Empirical 5 % fractile as the ceil(0.05 n)-th order statistic (definition of the original analysis)."""
    xs = np.sort(np.asarray(x, dtype=float))
    return xs[max(0, int(np.ceil(0.05 * xs.size)) - 1)]


def describe(x):
    x = np.asarray(x, dtype=float)
    return {
        "N": x.size,
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "cov": x.std(ddof=1) / x.mean(),
        "skew": stats.skew(x, bias=False),
        "exkurt": stats.kurtosis(x, bias=False),
        "q05": q05_order(x),
        "min": x.min(),
        "max": x.max(),
    }


def describe_by(df, by, cols):
    rows = []
    for key, g in df.groupby(by, observed=True):
        for c in cols:
            rows.append({by if isinstance(by, str) else "group": key, "variable": c, **describe(g[c])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- characteristic values


def ks_factor(n, p=0.05, conf=0.75):
    """One-sided tolerance factor for the p-fractile at confidence `conf` (non-central t, unknown variance)."""
    return stats.nct.ppf(conf, df=n - 1, nc=stats.norm.ppf(1 - p) * np.sqrt(n)) / np.sqrt(n)


def fk_lognormal(x, p=0.05, conf=0.75):
    """Characteristic (lower p-fractile) value assuming a lognormal distribution, with sample-size factor."""
    y = np.log(np.asarray(x, dtype=float))
    return float(np.exp(y.mean() - ks_factor(y.size, p, conf) * y.std(ddof=1)))


def f05_lognormal_point(x):
    y = np.log(np.asarray(x, dtype=float))
    return float(np.exp(y.mean() - stats.norm.ppf(0.95) * y.std(ddof=1)))


# ----------------------------------------------------------------------------- distribution fitting (vectorised MLE)

DISTRIBUTIONS = ["Normal", "Lognormal", "Gamma", "Weibull"]


def _fit_normal(X):
    return X.mean(axis=-1), X.std(axis=-1)


def _fit_lognormal(X):
    L = np.log(X)
    return L.mean(axis=-1), L.std(axis=-1)


def _fit_gamma(X, iters=60):
    m = X.mean(axis=-1)
    s = np.log(m) - np.log(X).mean(axis=-1)
    k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(iters):
        k = k - (np.log(k) - special.digamma(k) - s) / (1 / k - special.polygamma(1, k))
        k = np.maximum(k, 1e-6)
    return k, m / k


def _fit_weibull(X, iters=100):
    g = np.exp(np.log(X).mean(axis=-1, keepdims=True))
    Z = X / g                                   # scale-free for numerical stability
    L = np.log(Z)
    c = 1.2 / np.log(X).std(axis=-1)
    for _ in range(iters):
        Zc = Z ** c[..., None]
        s0, s1, s2 = Zc.sum(-1), (Zc * L).sum(-1), (Zc * L * L).sum(-1)
        f = 1 / c + L.mean(-1) - s1 / s0
        fp = -1 / c ** 2 - (s2 * s0 - s1 ** 2) / s0 ** 2
        c = np.maximum(c - f / fp, 1e-3)
    lam = g[..., 0] * ((Z ** c[..., None]).mean(-1)) ** (1 / c)
    return c, lam


_FIT = {"Normal": _fit_normal, "Lognormal": _fit_lognormal, "Gamma": _fit_gamma, "Weibull": _fit_weibull}


def frozen(dist, p1, p2):
    if dist == "Normal":
        return stats.norm(loc=p1, scale=p2)
    if dist == "Lognormal":
        return stats.lognorm(s=p2, scale=np.exp(p1))
    if dist == "Gamma":
        return stats.gamma(a=p1, scale=p2)
    if dist == "Weibull":
        return stats.weibull_min(c=p1, scale=p2)
    raise ValueError(dist)


def fit(dist, x):
    p1, p2 = _FIT[dist](np.asarray(x, dtype=float)[None, :])
    return frozen(dist, float(p1[0]), float(p2[0]))


def _cdf(dist, p1, p2, Xs):
    """CDF of sorted samples Xs (B x n) under row-wise parameters."""
    p1, p2 = p1[:, None], p2[:, None]
    if dist == "Normal":
        return stats.norm.cdf(Xs, loc=p1, scale=p2)
    if dist == "Lognormal":
        return stats.norm.cdf(np.log(Xs), loc=p1, scale=p2)
    if dist == "Gamma":
        return special.gammainc(p1, Xs / p2)
    if dist == "Weibull":
        return 1 - np.exp(-(Xs / p2) ** p1)
    raise ValueError(dist)


def _edf_stats(U):
    """KS, Cramér-von Mises and Anderson-Darling statistics from sorted PIT values (B x n)."""
    U = np.clip(U, 1e-12, 1 - 1e-12)
    n = U.shape[-1]
    i = np.arange(1, n + 1)
    ks = np.maximum((i / n - U).max(-1), (U - (i - 1) / n).max(-1))
    cvm = ((U - (2 * i - 1) / (2 * n)) ** 2).sum(-1) + 1 / (12 * n)
    ad = -n - ((2 * i - 1) * (np.log(U) + np.log(1 - U[..., ::-1]))).sum(-1) / n
    return ks, cvm, ad


def gof(x, dist, n_boot=999, rng=None):
    """Fit `dist` by maximum likelihood; EDF statistics with parametric-bootstrap p-values (estimated parameters)."""
    rng = np.random.default_rng(rng)
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    p1, p2 = _FIT[dist](x[None, :])
    obs = _edf_stats(_cdf(dist, p1, p2, x[None, :]))
    fz = frozen(dist, float(p1[0]), float(p2[0]))
    ll = fz.logpdf(x).sum()
    sims = np.sort(fz.rvs(size=(n_boot, n), random_state=rng), axis=1)
    sims = np.maximum(sims, 1e-9)
    q1, q2 = _FIT[dist](sims)
    boot = _edf_stats(_cdf(dist, q1, q2, sims))
    pvals = [(np.sum(b >= o[0]) + 1) / (n_boot + 1) for b, o in zip(boot, obs)]
    k = 2
    return {
        "dist": dist, "p1": float(p1[0]), "p2": float(p2[0]),
        "KS": float(obs[0][0]), "KS_p": pvals[0],
        "CvM": float(obs[1][0]), "CvM_p": pvals[1],
        "AD": float(obs[2][0]), "AD_p": pvals[2],
        "AIC": 2 * k - 2 * ll, "BIC": k * np.log(n) - 2 * ll,
        "q05_fit": float(fz.ppf(0.05)), "q05_emp": q05_order(x),
    }


def gof_table(x, n_boot=999, rng=None):
    rng = np.random.default_rng(rng)
    return pd.DataFrame([gof(x, d, n_boot, rng) for d in DISTRIBUTIONS])


# ----------------------------------------------------------------------------- group comparisons


def welch_anova(groups):
    n = np.array([len(g) for g in groups], dtype=float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    k = len(groups)
    w = n / v
    mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    F = a / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    df2 = (k ** 2 - 1) / (3 * lam)
    return F, k - 1, df2, stats.f.sf(F, k - 1, df2)


def oneway_summary(groups):
    """Classical one-way ANOVA with effect sizes, Welch ANOVA, Kruskal-Wallis and Brown-Forsythe tests."""
    allx = np.concatenate(groups)
    N, k = allx.size, len(groups)
    gm = allx.mean()
    ssb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups)
    ssw = sum(((np.asarray(g) - np.mean(g)) ** 2).sum() for g in groups)
    msw = ssw / (N - k)
    F = (ssb / (k - 1)) / msw
    Fw, d1w, d2w, pw = welch_anova(groups)
    kw = stats.kruskal(*groups)
    bf = stats.levene(*groups, center="median")
    return {
        "F": F, "df1": k - 1, "df2": N - k, "p": stats.f.sf(F, k - 1, N - k),
        "eta2": ssb / (ssb + ssw), "omega2": (ssb - (k - 1) * msw) / (ssb + ssw + msw),
        "Welch_F": Fw, "Welch_df1": d1w, "Welch_df2": d2w, "Welch_p": pw,
        "KW_H": kw.statistic, "KW_p": kw.pvalue,
        "BF_W": bf.statistic, "BF_p": bf.pvalue,
    }


def boxcox_lambda(y, X):
    """Box-Cox parameter maximising the profile likelihood of a linear model y(lambda) ~ X."""
    y = np.asarray(y, dtype=float)
    sly = np.log(y).sum()
    n = y.size

    def nll(lam):
        z = np.log(y) if abs(lam) < 1e-8 else (y ** lam - 1) / lam
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        rss = ((z - X @ beta) ** 2).sum()
        return n / 2 * np.log(rss / n) - (lam - 1) * sly

    return optimize.minimize_scalar(nll, bounds=(-2, 2), method="bounded").x
