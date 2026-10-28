"""Regression models M1-M7, numbered as in the manuscript, and two reference formulas.

M1 is stiffness only. M2 is the five section variables. M3 adds span and load positions.
M4 is stepwise from the M2 main effects. M5 adds the categorical layup and product.
M6 is stepwise with the engineered variables. M7 adds squares of the numerical predictors.

Protocol A  "selected on all data": stepwise formula chosen once on the full database, then repeated K-fold
            CV of that fixed formula (SDs reported over folds).
Protocol B  "nested": stepwise selection repeated inside every training fold (no information leakage).
Protocol C  "leave-one-product-out": selection and fit on six products, prediction of the seventh.
"""

import time

import numpy as np
import pandas as pd

from glulam import config, data, models as gm, plotting as gp, tables as gt

NUM5 = ["E", "h", "hext", "hint", "b"]
CAT5 = ["cext", "cint", "fjext", "fjint", "glue"]
# Manuscript M5. Family is the declared-strength label. M6 and M7 do not include it or product.
CAT_M5 = CAT5 + ["family", "product"]
# Engineered predictors in manuscript M6 and M7. No layup categoricals.
# logE = ln(E); hb = h/b; bh = b/h; Eh = E*h; Eh2 = E*h**2; Eb = E*b
# hint_frac = hint/h; h2b = h**2/b = 6*W/b**2; Eoh = E/h
# EZ7 = E*h**2/b = 6*E*W/b**2; EZ = E*h**3/(12*b) = E*I/b**2
# with I = b*h**3/12 and W = b*h**2/6 for a rectangular section.
ENG = ["logE", "hb", "bh", "Eh", "Eh2", "Eb", "hint_frac", "h2b", "Eoh", "EZ7", "EZ"]
# Manuscript M6 uses interactions. Manuscript M7 adds squares of the numerical predictors.
# ln_h600 = ln(h/600) is a starting main effect, as in S1. It may still be removed.
M6_BASE = NUM5 + ["ln_h600"] + ENG
M7_BASE = M6_BASE

SPECS = {
    "M1": dict(label="M1 -- $E_{\\mathrm{m,g}}$ only", kind="fixed", terms=[("E",)]),
    "M2": dict(label="M2 -- Section variables", kind="fixed", terms=[(v,) for v in NUM5]),
    "M3": dict(label="M3 -- All numerical variables", kind="fixed", terms=[(v,) for v in NUM5 + ["l", "a", "c"]]),
    "M4": dict(label="M4 -- Stepwise", kind="stepwise", base=NUM5, upper="interactions"),
    "M5": dict(label="M5 -- Stepwise + categorical", kind="stepwise", base=NUM5 + CAT_M5, upper="interactions"),
    "M6": dict(label="M6 -- Stepwise + engineered", kind="stepwise", base=M6_BASE, upper="interactions"),
    "M7": dict(label="M7 -- Stepwise quadratic engineered", kind="stepwise",
               base=M7_BASE, upper="quadratic"),
    "S1": dict(label="S1 -- $\\ln f_{\\mathrm{m}}\\sim\\ln E+\\ln h$ + outer class", kind="fixed",
               terms=[("ln_E",), ("ln_h600",), ("cext",)], transform="log"),
    "S2": dict(label="S2 -- S1 + product", kind="fixed",
               terms=[("ln_E",), ("ln_h600",), ("cext",), ("product",)], transform="log"),
}


def selected_m7_terms(df):
    """Terms of manuscript M7 after one stepwise fit on the analysis sample."""
    return gm.StepwiseOLS(M7_BASE, "quadratic").fit(df).selected()


def maker(spec):
    tr = spec.get("transform", "none")
    if spec["kind"] == "fixed":
        return lambda: gm.FixedOLS(spec["terms"], tr)
    return lambda: gm.StepwiseOLS(spec["base"], spec["upper"], tr)


def run(df):
    rows, oof_store, selected = [], {}, {}
    for key, spec in SPECS.items():
        t0 = time.time()
        mk = maker(spec)
        # Protocol A: selection on all data, then fixed-formula repeated CV
        full = mk().fit(df)
        selected[key] = [gm.term_name(t) for t in full.selected()]
        fixed = lambda full=full, tr=spec.get("transform", "none"): gm.FixedOLS(full.selected(), tr)
        fa, oof = gm.cross_validate(df, fixed, config.K_FOLDS, config.CV_REPEATS, config.SEED)
        oof_store[key] = oof
        row = {"model": key, "label": spec["label"], "n_terms": len(full.selected()),
               "rank_full": full.final.rank if spec["kind"] == "stepwise" else full.rank}
        row.update({f"A_{k}": v for k, v in gm.summarise_folds(fa).items()})
        # SD across repetitions of the pooled CV RMSE (not a measure of predictive uncertainty)
        pooled = pd.Series([np.sqrt(np.average(d["RMSE"] ** 2, weights=d["n_test"]))
                            for _, d in fa.groupby("repeat")])
        row["A_RMSE_sd_across_repeats"] = pooled.std(ddof=1)
        # Protocol B: nested selection (only differs for stepwise models)
        if spec["kind"] == "stepwise":
            fb, _ = gm.cross_validate(df, mk, config.K_FOLDS, config.NESTED_REPEATS, config.SEED + 1)
            row.update({f"B_{k}": v for k, v in gm.summarise_folds(fb).items()})
        else:
            row.update({f"B_{k}": row[f"A_{k}"] for k in gm.summarise_folds(fa)})
        # Protocol C: leave one product out
        lopo, pooled_c, _ = gm.leave_one_group_out(df, mk)
        row["C_products_predictable"] = int(lopo["predictable"].sum())
        row.update({f"C_{k}": v for k, v in pooled_c.items()})
        for _, r in lopo.iterrows():
            row[f"C_RMSE_{r['held_out']}"] = r.get("RMSE", np.nan)
        rows.append(row)
        print(f"{key}: {time.time() - t0:.1f}s  A R2={row['A_R2_mean']:.3f}  B R2={row['B_R2_mean']:.3f}  "
              f"C R2={row.get('C_R2', np.nan):.3f} ({row['C_products_predictable']}/7 products)", flush=True)
    return pd.DataFrame(rows), oof_store, selected


def coefficient_table(df):
    """Coefficients of the parsimonious model S1 (log scale) with standard errors."""
    import statsmodels.formula.api as smf
    m = smf.ols("ln_fm ~ ln_E + ln_h600 + C(cext, Treatment('C24'))", df).fit()
    out = pd.DataFrame({"term": m.params.index, "estimate": m.params.values, "se": m.bse.values,
                        "t": m.tvalues.values, "p": m.pvalues.values})
    out["term"] = out["term"].str.replace("C(cext, Treatment('C24'))[T.", "outer class ", regex=False).str.rstrip("]")
    vif = []
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = m.model.exog
    for i in range(X.shape[1]):
        vif.append(variance_inflation_factor(X, i) if i > 0 else np.nan)
    out["VIF"] = vif
    return out, m


def m7_vif(df):
    """Variance inflation factors of the manuscript M7 formula selected on all data."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    terms = selected_m7_terms(df)
    cats = {v for t in terms for v in t if v in gm.CATEGORICAL}
    lv = gm.levels_of(df, cats)
    X, names = gm.design(df, terms, lv)
    Xs = X.copy()
    Xs[:, 1:] = (Xs[:, 1:] - Xs[:, 1:].mean(0)) / Xs[:, 1:].std(0)
    return pd.DataFrame({"column": names[1:],
                         "VIF": [variance_inflation_factor(Xs, i) for i in range(1, Xs.shape[1])]})


def fig_obs_pred(df, oof):
    """Observed versus out-of-fold predictions for manuscript M1, M6 and M7."""
    keys = [
        ("M1", "M1: simple stiffness"),
        ("M6", "M6: stepwise, engineered variables"),
        ("M7", "M7: quadratic stepwise, engineered variables"),
    ]
    fig, axes = gp.figure("double", 0.36, 1, 3, sharex=True, sharey=True)
    lim = [10, 80]
    for ax, (k, title), lab in zip(axes, keys, "abc"):
        for fam in config.FAMILIES:
            m = (df["family"] == fam).to_numpy() & ~np.isnan(oof[k])
            st = gp.FAMILY_STYLE[fam]
            ax.plot(df["fm"].to_numpy()[m], oof[k][m], ls="none", marker=st["marker"], ms=1.8, mew=0,
                    color=st["color"], alpha=0.8, label=fam)
        ax.plot(lim, lim, color=gp.INK, lw=0.6)
        ok = ~np.isnan(oof[k])
        mt = gm.metrics(df["fm"].to_numpy()[ok], oof[k][ok])
        ax.text(0.97, 0.04, f"RMSE $={mt['RMSE']:.2f}$ MPa\n$R^2={mt['R2']:.2f}$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7)
        ax.set_title(title)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Measured " + gp.LABEL["fm"])
        gp.panel_label(ax, lab)
    axes[0].set_ylabel("Predicted (out-of-fold) " + gp.LABEL["fm"])
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), markerscale=2, handletextpad=0.2)
    return gp.save(fig, "fig_obs_pred")


def main():
    gp.use_latex_style()
    df = data.load()
    res, oof, selected = run(df)
    gt.save_csv(res, "T5_model_comparison")
    pd.DataFrame([{"model": k, "terms": " + ".join(v)} for k, v in selected.items()]).pipe(gt.save_csv, "T5_selected_terms")
    coef, _ = coefficient_table(df)
    gt.save_csv(coef, "T6_S1_coefficients")
    gt.save_csv(m7_vif(df), "T6_M7_vif")
    fig_obs_pred(df, {"M1": oof["M1"], "M6": oof["M6"], "M7": oof["M7"]})

    t = res[["label", "n_terms", "A_RMSE_mean", "A_RMSE_sd", "A_R2_mean", "A_R2_sd", "B_RMSE_mean", "B_R2_mean",
             "C_RMSE", "C_R2", "C_products_predictable"]].copy()
    t = gt.fmt(t, {"A_RMSE_mean": "{:.2f}", "A_RMSE_sd": "{:.2f}", "A_R2_mean": "{:.3f}", "A_R2_sd": "{:.3f}",
                   "B_RMSE_mean": "{:.2f}", "B_R2_mean": "{:.3f}", "C_RMSE": "{:.2f}", "C_R2": "{:.3f}"})
    t["A_RMSE"] = t["A_RMSE_mean"] + " $\\pm$ " + t["A_RMSE_sd"]
    t["A_R2"] = t["A_R2_mean"] + " $\\pm$ " + t["A_R2_sd"]
    t = t[["label", "n_terms", "A_RMSE", "A_R2", "B_RMSE_mean", "B_R2_mean", "C_RMSE", "C_R2", "C_products_predictable"]]
    t.columns = ["Model", "Terms", "RMSE (A)", "$R^2$ (A)", "RMSE (B)", "$R^2$ (B)", "RMSE (C)", "$R^2$ (C)",
                 "Products (C)"]
    gt.save_tex(t.astype(str), "T5_model_comparison",
                "Out-of-sample performance of the regression models. (A) formula selected on all data, "
                f"{config.CV_REPEATS}$\\times${config.K_FOLDS}-fold CV, mean $\\pm$ SD over folds; "
                "(B) selection repeated inside each training fold; (C) leave-one-product-out, pooled over "
                "the predictable products. RMSE in MPa.", "tab:models")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(res[["model", "n_terms", "A_RMSE_mean", "A_RMSE_sd", "A_RMSE_sd_across_repeats", "A_R2_mean",
                   "A_R2_sd", "B_RMSE_mean", "B_R2_mean", "C_RMSE", "C_R2", "C_products_predictable"]].round(3).to_string(index=False))
        print(coef.round(4).to_string(index=False))
        for k, v in selected.items():
            print(k, len(v), " + ".join(v)[:400])


if __name__ == "__main__":
    main()
