"""Correlations (without test-setup quantities) and the strength-stiffness relationship within each product."""

import numpy as np
import pandas as pd
from scipy import stats

from glulam import config, data, plotting as gp, tables as gt


def correlation_matrix(df):
    cols = ["fm", "E", "h", "b", "hext", "hint"] + (["rho"] if data.has_density(df) else []) + \
           (["mc"] if "mc" in df.columns else [])
    d = df[cols].dropna()
    pear = d.corr(method="pearson")
    spear = d.corr(method="spearman")
    return pear, spear


def fisher_ci(r, n, level=0.95):
    z = np.arctanh(r)
    h = stats.norm.ppf(0.5 + level / 2) / np.sqrt(n - 3)
    return np.tanh(z - h), np.tanh(z + h)


def within_product_relation(df):
    rows = []
    for p, g in df.groupby("product", observed=True):
        r, pval = stats.pearsonr(g["fm"], g["E"])
        lo, hi = fisher_ci(r, len(g))
        slope, icpt = np.polyfit(np.log(g["E"]), np.log(g["fm"]), 1)
        resid = np.log(g["fm"]) - (icpt + slope * np.log(g["E"]))
        row = {"product": p, "family": g["family"].iloc[0], "N": len(g), "r_fm_E": r, "r_ci_lo": lo,
               "r_ci_hi": hi, "rho_spearman": stats.spearmanr(g["fm"], g["E"]).statistic,
               "slope_lnfm_lnE": slope, "sigma_ln_resid": resid.std(ddof=2),
               "sigma_ln_fm": np.log(g["fm"]).std(ddof=1)}
        if data.has_density(df):
            row["r_fm_rho"] = stats.pearsonr(g["fm"], g["rho"])[0]
            row["r_E_rho"] = stats.pearsonr(g["E"], g["rho"])[0]
        rows.append(row)
    r, _ = stats.pearsonr(df["fm"], df["E"])
    lo, hi = fisher_ci(r, len(df))
    rows.append({"product": "all", "family": "all", "N": len(df), "r_fm_E": r, "r_ci_lo": lo, "r_ci_hi": hi,
                 "rho_spearman": stats.spearmanr(df["fm"], df["E"]).statistic})
    # pooled within-product correlation (product-centred variables)
    c = df.assign(fmc=df["fm"] - df.groupby("product", observed=True)["fm"].transform("mean"),
                  Ec=df["E"] - df.groupby("product", observed=True)["E"].transform("mean"))
    rows.append({"product": "pooled within", "family": "all", "N": len(df),
                 "r_fm_E": stats.pearsonr(c["fmc"], c["Ec"])[0]})
    return pd.DataFrame(rows)


def fig_fm_E(df, rel):
    fig, axes = gp.figure("double", 0.50, 2, 4, sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, p in zip(axes, config.PRODUCTS):
        g = df[df["product"] == p]
        st = gp.product_style(p)
        ax.plot(df["E"] / 1e3, df["fm"], ls="none", marker=".", ms=1.2, color="#d9d8d4", zorder=1)
        ax.plot(g["E"] / 1e3, g["fm"], ls="none", marker=st["marker"], ms=1.8, mew=0, color=st["color"], zorder=2)
        b1, b0 = np.polyfit(np.log(g["E"]), np.log(g["fm"]), 1)
        e = np.linspace(g["E"].min(), g["E"].max(), 50)
        ax.plot(e / 1e3, np.exp(b0) * e ** b1, color=gp.INK, lw=0.9, zorder=3)
        r = rel.loc[rel["product"] == p, "r_fm_E"].iloc[0]
        ax.set_title(f"{p} ({g['family'].iloc[0]}), $N={len(g)}$", fontsize=7, pad=2)
        ax.text(0.04, 0.95, f"$r={r:.2f}$", transform=ax.transAxes, va="top", fontsize=7)
        ax.axhline(config.FMK_DECLARED[g["family"].iloc[0]], color=gp.INK_2, lw=0.6, ls="--")
    ax = axes[-1]
    ax.plot(df["E"] / 1e3, df["fm"], ls="none", marker=".", ms=1.2, color=gp.INK_2)
    r = rel.loc[rel["product"] == "all", "r_fm_E"].iloc[0]
    rw = rel.loc[rel["product"] == "pooled within", "r_fm_E"].iloc[0]
    ax.set_title(f"All products, $N={len(df)}$", fontsize=7, pad=2)
    ax.text(0.04, 0.95, f"$r={r:.2f}$\n$r_{{\\mathrm{{within}}}}={rw:.2f}$", transform=ax.transAxes, va="top",
            fontsize=7)
    for i, ax in enumerate(axes):
        if i >= 4:
            ax.set_xlabel(gp.LABEL["E"])
        if i % 4 == 0:
            ax.set_ylabel(gp.LABEL["fm"])
    return gp.save(fig, "fig_fm_E_products")


def main():
    gp.use_latex_style()
    df = data.load()
    pear, spear = correlation_matrix(df)
    rel = within_product_relation(df)
    gt.save_csv(pear.reset_index(), "T10_correlation_pearson")
    gt.save_csv(spear.reset_index(), "T10_correlation_spearman")
    gt.save_csv(rel, "T10_strength_stiffness")
    fig_fm_E(df, rel)
    print("density available:", data.has_density(df))
    print(pear.round(2).to_string())
    with pd.option_context("display.width", 200):
        print(rel.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
