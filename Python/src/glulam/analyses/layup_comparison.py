"""Matched comparison of P2 (GL24*, outer lamellae C24) and P7 (GL33*, outer lamellae C40).

Both products share depth, width, lamella thicknesses, inner lamella class, finger-joint and lamination adhesives;
only the outer-lamella grade (and the declared class) differ.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from glulam import config, data, plotting as gp, stats as gst, tables as gt

PAIR = ("P2", "P7")
COLORS = {"P2": gp.PALETTE[0], "P7": gp.PALETTE[3]}
MARK = {"P2": "o", "P7": "D"}
LS = {"P2": "-", "P7": ":"}


def subset(df, matched=True):
    d = df[df["product"].isin(PAIR)].copy()
    if matched:
        d = d[d["b"] == 48]
    d["product"] = d["product"].astype(str)
    return d


def boot_q05_diff(a, b, n=4999, rng=None):
    rng = np.random.default_rng(rng)
    da = [gst.q05_order(rng.choice(a, a.size)) - gst.q05_order(rng.choice(b, b.size)) for _ in range(n)]
    return np.percentile(da, [2.5, 97.5])


def compare(d, label):
    a = d[d["product"] == "P2"]
    b = d[d["product"] == "P7"]
    rows = []
    for var in ["fm", "E"]:
        x, y = b[var].to_numpy(), a[var].to_numpy()       # difference P7 - P2
        t = stats.ttest_ind(x, y, equal_var=False)
        se = np.sqrt(x.var(ddof=1) / x.size + y.var(ddof=1) / y.size)
        ci = (x.mean() - y.mean()) + np.array([-1, 1]) * stats.t.ppf(0.975, t.df) * se
        rows.append({"subset": label, "variable": var, "N_P2": y.size, "N_P7": x.size,
                     "mean_P2": y.mean(), "mean_P7": x.mean(), "diff": x.mean() - y.mean(),
                     "ci_lo": ci[0], "ci_hi": ci[1], "ratio": x.mean() / y.mean(), "p_welch": t.pvalue,
                     "p_mw": stats.mannwhitneyu(x, y).pvalue,
                     "q05_P2": gst.q05_order(y), "q05_P7": gst.q05_order(x),
                     "fk_P2": gst.fk_lognormal(y), "fk_P7": gst.fk_lognormal(x),
                     "cov_P2": y.std(ddof=1) / y.mean(), "cov_P7": x.std(ddof=1) / x.mean()})
    rows[0]["q05_diff_ci"] = tuple(np.round(boot_q05_diff(b["fm"].to_numpy(), a["fm"].to_numpy(),
                                                          rng=config.SEED), 2))
    return rows


def ancova(d):
    """Strength at equal stiffness: ln f_m ~ ln E + product (and with product-specific slopes)."""
    m1 = smf.ols("ln_fm ~ ln_E + C(product)", d).fit()
    m2 = smf.ols("ln_fm ~ ln_E * C(product)", d).fit()
    k = "C(product)[T.P7]"
    lo, hi = m1.conf_int().loc[k]
    return pd.DataFrame([{
        "shift_P7_vs_P2_at_equal_E": np.exp(m1.params[k]) - 1, "ci_lo": np.exp(lo) - 1, "ci_hi": np.exp(hi) - 1,
        "p": m1.pvalues[k], "slope_lnE_common": m1.params["ln_E"],
        "slope_P2": m2.params["ln_E"], "slope_P7": m2.params["ln_E"] + m2.params["ln_E:C(product)[T.P7]"],
        "p_slope_diff": m2.pvalues["ln_E:C(product)[T.P7]"],
        "r_P2": np.corrcoef(d.loc[d["product"] == "P2", "fm"], d.loc[d["product"] == "P2", "E"])[0, 1],
        "r_P7": np.corrcoef(d.loc[d["product"] == "P7", "fm"], d.loc[d["product"] == "P7", "E"])[0, 1],
    }])


def fig_pair(d):
    fig, axes = gp.figure("double", 0.34, 1, 3)
    lab = {"P2": "P2, GL24*, outer C24", "P7": "P7, GL33*, outer C40"}
    ax = axes[0]
    for p in PAIR:
        g = d[d["product"] == p]
        ax.plot(g["E"] / 1e3, g["fm"], ls="none", marker=MARK[p], ms=2.0, mew=0, color=COLORS[p], alpha=0.75,
                label=lab[p])
        b1, b0 = np.polyfit(np.log(g["E"]), np.log(g["fm"]), 1)
        e = np.linspace(g["E"].min(), g["E"].max(), 50)
        ax.plot(e / 1e3, np.exp(b0) * e ** b1, color=COLORS[p], ls=LS[p], lw=1.0)
    ax.set_xlabel(gp.LABEL["E"])
    ax.set_ylabel(gp.LABEL["fm"])
    ax.legend(loc="lower right", markerscale=2, handletextpad=0.2)
    for ax, var, scale, xl in [(axes[1], "fm", 1, gp.LABEL["fm"]), (axes[2], "E", 1e-3, gp.LABEL["E"])]:
        for p in PAIR:
            x = np.sort(d.loc[d["product"] == p, var].to_numpy() * scale)
            ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", color=COLORS[p], ls=LS[p], lw=1.0,
                    label=p)
        if var == "fm":
            for p, fam in [("P2", "GL24*"), ("P7", "GL33*")]:
                ax.axvline(config.FMK_DECLARED[fam], color=COLORS[p], ls=LS[p], lw=0.6)
            ax.text(24.5, 0.55, r"$f_{\mathrm{m,k}}=24$", rotation=90, fontsize=6, color=gp.INK_2)
            ax.text(33.5, 0.55, r"$f_{\mathrm{m,k}}=33$", rotation=90, fontsize=6, color=gp.INK_2)
        ax.axhline(0.05, color=gp.INK_2, lw=0.5, ls=(0, (1, 2)))
        ax.set_xlabel(xl)
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(0, 1)
    axes[2].legend(loc="lower right")
    for ax, l in zip(axes, "abc"):
        gp.panel_label(ax, l)
    return gp.save(fig, "fig_layup_P2_P7")


def main():
    gp.use_latex_style()
    df = data.load()
    check = (df[df["product"].isin(PAIR)].groupby("product", observed=True)
             [["h", "b", "hext", "hint", "cext", "cint", "fjext", "fjint", "glue", "l"]]
             .agg(lambda s: ", ".join(map(str, sorted(s.unique())))))
    print(check.to_string())
    rows = compare(subset(df, True), "matched b = 48 mm") + compare(subset(df, False), "all")
    res = pd.DataFrame(rows)
    anc = ancova(subset(df, True))
    gt.save_csv(res, "T7_layup_P2_P7")
    gt.save_csv(anc, "T7_layup_P2_P7_ancova")
    gt.save_csv(check.reset_index(), "T7_layup_P2_P7_geometry")
    fig_pair(subset(df, True))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(res.round(4).to_string(index=False))
        print(anc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
