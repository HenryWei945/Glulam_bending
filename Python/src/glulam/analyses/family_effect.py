"""Information carried by the product-family label: ANOVA variants, variance components, nested-model tests."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from glulam import config, data, stats as gst, tables as gt

warnings.filterwarnings("ignore", category=UserWarning)


def oneway_tables(df):
    rows = []
    for factor in ["family", "product"]:
        for var in ["fm", "fm600", "E"]:
            groups = [g[var].to_numpy() for _, g in df.groupby(factor, observed=True)]
            rows.append({"factor": factor, "response": var, **gst.oneway_summary(groups)})
    return pd.DataFrame(rows)


def adjusted_anova(df):
    """Family term adjusted for width, depth and span (type II ANOVA)."""
    m = smf.ols("fm ~ C(family) + b + h + l", df).fit()
    a = sm.stats.anova_lm(m, typ=2).reset_index().rename(columns={"index": "term"})
    return a


def variance_components(df):
    """Random-intercept model for ln f_m: share of variance between products (ICC)."""
    rows = []
    for label, formula in [("null", "ln_fm ~ 1"), ("ln E + ln h", "ln_fm ~ ln_E + ln_h600")]:
        m = smf.mixedlm(formula, df, groups=df["product"].astype(str)).fit(reml=True)
        tau2 = float(m.cov_re.iloc[0, 0])
        s2 = float(m.scale)
        rows.append({"model": label, "tau2_between": tau2, "sigma2_within": s2, "ICC": tau2 / (tau2 + s2)})
    return pd.DataFrame(rows)


def nested_tests(df):
    """Likelihood-ratio tests: does the family label (or product) add information beyond measured variables?"""
    specs = [
        ("E, h, outer class", "ln_fm ~ ln_E + ln_h600 + C(cext)"),
        ("E, h, outer class, layup", "ln_fm ~ ln_E + ln_h600 + C(cext) + C(cint) + hext + hint + b"),
    ]
    rows = []
    for name, base in specs:
        m0 = smf.ols(base, df).fit()
        for add in ["C(family)", "C(product)"]:
            m1 = smf.ols(f"{base} + {add}", df).fit()
            lr, p, ddf = m1.compare_lr_test(m0)
            rows.append({"base": name, "added": add.strip("C()"), "LR": lr, "df": ddf, "p": p,
                         "R2_base": m0.rsquared, "R2_added": m1.rsquared, "dR2": m1.rsquared - m0.rsquared,
                         "dBIC": m1.bic - m0.bic})
    return pd.DataFrame(rows)


def lower_tail_family(df):
    """5 % quantile regression of ln f_m on the family label, with and without stiffness and depth."""
    rows = []
    for name, f in [("family only", "ln_fm ~ C(family)"), ("family + E + h", "ln_fm ~ C(family) + ln_E + ln_h600")]:
        q = smf.quantreg(f, df).fit(q=0.05, max_iter=5000)
        for term in q.params.index:
            if term.startswith("C(family)"):
                rows.append({"model": name, "term": term.replace("C(family)[T.", "").rstrip("]"),
                             "coef_vs_GL24": q.params[term], "ratio": np.exp(q.params[term]),
                             "p": q.pvalues[term]})
    return pd.DataFrame(rows)


def main():
    df = data.load()
    ow = oneway_tables(df)
    adj = adjusted_anova(df)
    vc = variance_components(df)
    nt = nested_tests(df)
    lt = lower_tail_family(df)
    for name, d in [("T4_oneway", ow), ("T4_adjusted_anova", adj), ("T4_variance_components", vc),
                    ("T4_nested_tests", nt), ("T4_lower_tail_family", lt)]:
        gt.save_csv(d, name)

    t = ow[ow["response"].isin(["fm", "E"])][["factor", "response", "F", "df1", "df2", "p", "eta2", "Welch_F",
                                              "Welch_p", "KW_p", "BF_p"]].copy()
    t["response"] = t["response"].map({"fm": "$f_{\\mathrm{m}}$", "E": "$E_{\\mathrm{m,g}}$"})
    for c in ["p", "Welch_p", "KW_p", "BF_p"]:
        t[c] = t[c].map(gt.pval)
    t = gt.fmt(t, {"F": "{:.2f}", "df1": "{:.0f}", "df2": "{:.0f}", "eta2": "{:.3f}", "Welch_F": "{:.2f}"})
    t.columns = ["Factor", "Response", "$F$", "df$_1$", "df$_2$", "$p$", "$\\eta^2$", "$F_{\\mathrm{Welch}}$",
                 "$p_{\\mathrm{Welch}}$", "$p_{\\mathrm{KW}}$", "$p_{\\mathrm{BF}}$"]
    gt.save_tex(t.astype(str), "T4_oneway",
                "One-way comparisons between families and between products: classical ANOVA with effect size "
                "$\\eta^2$, Welch ANOVA, Kruskal--Wallis (KW) and Brown--Forsythe test of equal variances (BF).",
                "tab:anova")
    with pd.option_context("display.width", 200):
        print(ow.round(4).to_string(index=False))
        print(adj.round(4).to_string(index=False))
        print(vc.round(4).to_string(index=False))
        print(nt.round(4).to_string(index=False))
        print(lt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
