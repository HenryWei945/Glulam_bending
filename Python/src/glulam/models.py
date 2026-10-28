"""Linear models with categorical/interaction terms, BIC stepwise selection and cross-validation protocols.

A term is a tuple of variable names: ("E",) main effect, ("E", "h") interaction, ("E", "E") square.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from . import stats as gst

CATEGORICAL = {"cext", "cint", "fjext", "fjint", "glue", "family", "product"}
REFERENCE_LEVEL = {"cext": "C24"}   # reference coding used in the manuscript (Table 6)


def term_name(t):
    return t[0] if len(t) == 1 else (f"{t[0]}^2" if t[0] == t[1] else ":".join(t))


# ----------------------------------------------------------------------------- design matrix


def levels_of(df, variables):
    lv = {}
    for v in variables:
        if v in CATEGORICAL:
            cats = sorted(pd.Series(df[v]).astype(str).unique())
            ref = REFERENCE_LEVEL.get(v)
            if ref in cats:
                cats.remove(ref)
                cats = [ref] + cats
            lv[v] = cats
    return lv


def _var_columns(df, v, lv):
    if v in CATEGORICAL:
        s = df[v].astype(str).to_numpy()
        return [(f"{v}[{c}]", (s == c).astype(float)) for c in lv[v][1:]]
    return [(v, df[v].to_numpy(dtype=float))]


def design(df, terms, lv):
    names, cols = ["Intercept"], [np.ones(len(df))]
    for t in terms:
        parts = [_var_columns(df, v, lv) for v in t]
        if len(t) == 2 and t[0] == t[1]:
            parts = [parts[0], parts[0]]
        combos = [[]]
        for p in parts:
            combos = [c + [q] for c in combos for q in p]
        for combo in combos:
            names.append(":".join(n for n, _ in combo) if len(t) == 1 or t[0] != t[1] else f"{combo[0][0]}^2")
            cols.append(np.prod([c for _, c in combo], axis=0))
    return np.column_stack(cols), names


def unseen_levels(train, test, terms):
    """True for test rows whose categorical levels never occur in the training data."""
    bad = np.zeros(len(test), dtype=bool)
    for v in {v for t in terms for v in t if v in CATEGORICAL}:
        bad |= ~test[v].astype(str).isin(set(train[v].astype(str))).to_numpy()
    return bad


def ols(X, y):
    """Least squares on column-scaled X (robust rank detection); returns coefficients, rank and RSS."""
    scale = np.sqrt((X ** 2).mean(axis=0))
    scale[scale == 0] = 1.0
    beta_s, _, rank, _ = np.linalg.lstsq(X / scale, y, rcond=1e-10)
    beta = beta_s / scale
    rss = float(((y - X @ beta) ** 2).sum())
    return beta, int(rank), rss


def bic(rss, n, k):
    return n * np.log(rss / n) + k * np.log(n)


# ----------------------------------------------------------------------------- response transforms


class Transform:
    def __init__(self, kind="none"):
        self.kind, self.lam, self.s2 = kind, None, 0.0

    def fit(self, y, X=None):
        if self.kind == "boxcox":
            self.lam = gst.boxcox_lambda(y, X)
        return self

    def forward(self, y):
        if self.kind == "log":
            return np.log(y)
        if self.kind == "boxcox":
            return np.log(y) if abs(self.lam) < 1e-8 else (y ** self.lam - 1) / self.lam
        return y

    def inverse(self, z):
        if self.kind == "log":
            return np.exp(z + self.s2 / 2)          # conditional mean of a lognormal response
        if self.kind == "boxcox":
            if abs(self.lam) < 1e-8:
                return np.exp(z)
            return np.maximum(self.lam * z + 1, 1e-9) ** (1 / self.lam)   # back-transformed median
        return z


# ----------------------------------------------------------------------------- model specifications


class FixedOLS:
    """OLS with a fixed set of terms."""

    def __init__(self, terms, transform="none", name=None):
        self.terms, self.transform, self.name = [tuple(t) for t in terms], transform, name

    def fit(self, df, y="fm"):
        self.lv = levels_of(df, {v for t in self.terms for v in t})
        X, self.colnames = design(df, self.terms, self.lv)
        self.tr = Transform(self.transform)
        base = df[[v for v in ["E", "h", "hext", "hint", "b"]]].to_numpy(float)
        self.tr.fit(df[y].to_numpy(float), np.column_stack([np.ones(len(df)), base]))
        z = self.tr.forward(df[y].to_numpy(float))
        self.beta, self.rank, rss = ols(X, z)
        self.tr.s2 = rss / max(len(df) - self.rank, 1)
        self.n = len(df)
        return self

    def predict(self, df):
        X, _ = design(df, self.terms, self.lv)
        return self.tr.inverse(X @ self.beta)

    def selected(self):
        return self.terms


class StepwiseOLS:
    """Bidirectional BIC stepwise selection (MATLAB stepwiselm, Criterion='bic').

    The starting model is the full set of main effects in `base`. Terms may be removed
    down to an intercept (`Lower='constant'`) and added up to pairwise interactions,
    plus squares of numerical predictors when `upper='quadratic'`. With Criterion='bic',
    PEnter and PRemove are thresholds on the BIC change, not F-test p-values:
    add a term if BIC(new) - BIC(current) < 0.05; remove a term if keeping it worsens
    BIC by more than 0.10. A term that does not increase the column rank is dropped.
    Categorical predictors enter and leave as one group. Hierarchy is enforced.
    """

    # Thresholds used in the legacy stepwiselm calls (interpreted on the BIC scale).
    P_ENTER = 0.05
    P_REMOVE = 0.10

    def __init__(self, base, upper="interactions", transform="none", name=None, max_steps=200):
        self.base = [tuple(t) if isinstance(t, (tuple, list)) else (t,) for t in base]
        self.upper, self.transform, self.name, self.max_steps = upper, transform, name, max_steps

    def candidates(self):
        mains = [t[0] for t in self.base]
        cand = [(v,) for v in mains] + [tuple(sorted(p)) for p in combinations(mains, 2)]
        if self.upper == "quadratic":
            cand += [(v, v) for v in mains if v not in CATEGORICAL]
        return cand

    @staticmethod
    def _allowed_add(t, model):
        return all((v,) in model for v in set(t)) if len(t) > 1 else True

    @staticmethod
    def _allowed_remove(t, model):
        return not (len(t) == 1 and any(len(u) > 1 and t[0] in u for u in model))

    def fit(self, df, y="fm"):
        lv = levels_of(df, {t[0] for t in self.base})
        tr = Transform(self.transform)
        base_num = df[["E", "h", "hext", "hint", "b"]].to_numpy(float)
        tr.fit(df[y].to_numpy(float), np.column_stack([np.ones(len(df)), base_num]))
        z = tr.forward(df[y].to_numpy(float))
        n = len(df)
        cache = {}

        def fit_stats(model):
            key = frozenset(model)
            if key not in cache:
                terms = sorted(model, key=lambda t: (len(t), t))
                X, _ = design(df, terms, lv)
                _, rank, rss = ols(X, z)
                cache[key] = (bic(rss, n, rank), int(rank), int(X.shape[1]))
            return cache[key]

        def drop_redundant(model):
            # A column that does not raise the rank is removed, as in stepwiselm.
            changed = True
            while changed and model:
                changed = False
                _, rank, ncol = fit_stats(model)
                if rank >= ncol:
                    break
                for t in sorted(model, key=lambda u: (len(u), u)):
                    if not self._allowed_remove(t, model):
                        continue
                    _, rank_wo, _ = fit_stats(model - {t})
                    if rank_wo == rank:
                        model.discard(t)
                        changed = True
                        break
            return model

        model = drop_redundant(set(self.base))
        cand = self.candidates()
        for _ in range(self.max_steps):
            model = drop_redundant(model)
            current = fit_stats(model)[0]
            adds = []
            for t in cand:
                if t in model or not self._allowed_add(t, model):
                    continue
                trial = model | {t}
                bic_new, rank_new, ncol_new = fit_stats(trial)
                if rank_new < ncol_new:
                    continue
                delta = bic_new - current
                if delta < self.P_ENTER:
                    adds.append((delta, t))
            if adds:
                model.add(min(adds)[1])
                continue
            rems = []
            for t in model:
                if not self._allowed_remove(t, model):
                    continue
                worsen = current - fit_stats(model - {t})[0]
                if worsen > self.P_REMOVE:
                    rems.append((worsen, t))
            if rems:
                model.discard(max(rems)[1])
                continue
            break
        self.terms = sorted(model, key=lambda t: (len(t), t))
        self.final = FixedOLS(self.terms, self.transform).fit(df, y)
        return self

    def predict(self, df):
        return self.final.predict(df)

    def selected(self):
        return self.terms


# ----------------------------------------------------------------------------- metrics and CV protocols


def metrics(y, yhat):
    e = yhat - y
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "MBE": float(np.mean(e)),
        "SDerr": float(np.std(e, ddof=1)),
        "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
        "VAF": float(100 * (1 - np.var(y - yhat, ddof=1) / np.var(y, ddof=1))),
    }


def kfold_indices(n, k, rng):
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def cross_validate(df, make_model, k=5, repeats=10, seed=0, y="fm"):
    """Repeated K-fold CV; `make_model()` returns an unfitted spec, so any selection happens inside each fold.

    Returns per-fold metrics and the out-of-fold predictions of the first repetition.
    """
    rng = np.random.default_rng(seed)
    rows, oof = [], np.full(len(df), np.nan)
    for r in range(repeats):
        folds = kfold_indices(len(df), k, rng)
        for f, te_idx in enumerate(folds):
            te = df.iloc[te_idx]
            tr = df.drop(df.index[te_idx])
            m = make_model().fit(tr, y)
            yhat = m.predict(te)
            ok = ~unseen_levels(tr, te, m.selected())
            rows.append({"repeat": r, "fold": f, "n_test": int(ok.sum()),
                         **metrics(te[y].to_numpy(float)[ok], np.asarray(yhat)[ok])})
            if r == 0:
                oof[te_idx[ok]] = np.asarray(yhat)[ok]
    return pd.DataFrame(rows), oof


def leave_one_group_out(df, make_model, group="product", y="fm"):
    """Leave-one-product-out: train on all other products, predict the held-out one."""
    rows, oof = [], np.full(len(df), np.nan)
    for g in df[group].cat.categories if hasattr(df[group], "cat") else sorted(df[group].unique()):
        mask = (df[group] == g).to_numpy()
        if not mask.any():
            continue
        tr, te = df[~mask], df[mask]
        m = make_model().fit(tr, y)
        bad = unseen_levels(tr, te, m.selected())
        row = {"held_out": g, "n_test": int(mask.sum()), "predictable": not bad.any()}
        if not bad.any():
            yhat = np.asarray(m.predict(te))
            oof[np.where(mask)[0]] = yhat
            row.update(metrics(te[y].to_numpy(float), yhat))
        rows.append(row)
    res = pd.DataFrame(rows)
    ok = ~np.isnan(oof)
    pooled = metrics(df[y].to_numpy(float)[ok], oof[ok]) if ok.any() else {}
    return res, pooled, oof


def summarise_folds(folds):
    cols = ["RMSE", "MAE", "MBE", "SDerr", "R2", "VAF"]
    out = {}
    for c in cols:
        out[f"{c}_mean"] = folds[c].mean()
        out[f"{c}_sd"] = folds[c].std(ddof=1)
    return out
