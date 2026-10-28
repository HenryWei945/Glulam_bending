"""Product overview (Table 1 from the data), descriptive statistics, distribution and box plots."""

import numpy as np
import pandas as pd

from glulam import config, data, plotting as gp, stats as gst, tables as gt


def rng_str(s, digits=0):
    lo, hi = s.min(), s.max()
    f = f"{{:.{digits}f}}"
    return f.format(lo) if np.isclose(lo, hi) else f"{f.format(lo)}--{f.format(hi)}"


def product_overview(df):
    rows = []
    for p, g in df.groupby("product", observed=True):
        fam = g["family"].iloc[0]
        rows.append({
            "Product": p, "Family": fam, "N": len(g),
            "fmk_declared": config.FMK_DECLARED[fam], "E_declared": config.E_DECLARED[p],
            "b": rng_str(g["b"]), "h": rng_str(g["h"]), "hext": rng_str(g["hext"], 1), "hint": rng_str(g["hint"], 1),
            "class_ext_int": f"{g['cext'].iloc[0]}/{g['cint'].iloc[0]}",
            "fj_ext_int": f"{g['fjext'].iloc[0]}/{g['fjint'].iloc[0]}",
            "glue": "/".join(sorted(g["glue"].unique())),
            "n_depths": g["h"].nunique(),
        })
    return pd.DataFrame(rows)


def descriptive_table(df, by):
    rows = []
    for key, g in df.groupby(by, observed=True):
        for var, scale in [("fm", 1.0), ("E", 1e-3)]:
            d = gst.describe(g[var] * scale)
            rows.append({by: key, "response": var, **d})
    return pd.DataFrame(rows)


def fig_family_ecdf(df):
    fig, axes = gp.figure("double", 0.36, 1, 2)
    for ax, var, scale, lab in [(axes[0], "fm", 1, gp.LABEL["fm"]), (axes[1], "E", 1e-3, gp.LABEL["E"])]:
        for fam in config.FAMILIES:
            x = np.sort(df.loc[df["family"] == fam, var].to_numpy() * scale)
            st = gp.FAMILY_STYLE[fam]
            ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", color=st["color"], ls=st["ls"],
                    lw=1.0, label=f"{fam} ($N={x.size}$)")
        ax.axhline(0.05, color=gp.INK_2, lw=0.5, ls=(0, (1, 2)))
        ax.set_xlabel(lab)
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(0, 1)
    axes[0].legend(loc="lower right")
    axes[0].text(axes[0].get_xlim()[0], 0.07, r"$p=0.05$", fontsize=6, color=gp.INK_2)
    gp.panel_label(axes[0], "a")
    gp.panel_label(axes[1], "b")
    return gp.save(fig, "fig_ecdf_family")


def fig_box_product(df):
    fig, axes = gp.figure("double", 0.34, 1, 2)
    prods = config.PRODUCTS
    for ax, var, scale, lab in [(axes[0], "fm", 1, gp.LABEL["fm"]), (axes[1], "E", 1e-3, gp.LABEL["E"])]:
        vals = [df.loc[df["product"] == p, var].to_numpy() * scale for p in prods]
        bp = ax.boxplot(vals, positions=np.arange(len(prods)), widths=0.55, patch_artist=True,
                        whis=1.5, flierprops=dict(marker=".", markersize=2, markerfacecolor=gp.INK_2,
                                                  markeredgecolor="none"),
                        medianprops=dict(color=gp.INK, lw=0.8), boxprops=dict(lw=0.5),
                        whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
        for patch, p in zip(bp["boxes"], prods):
            fam = df.loc[df["product"] == p, "family"].iloc[0]
            patch.set_facecolor(gp.FAMILY_STYLE[fam]["color"])
            patch.set_alpha(0.55)
            patch.set_edgecolor(gp.INK)
        means = [v.mean() for v in vals]
        ax.plot(np.arange(len(prods)), means, ls="none", marker="D", ms=2.5, color=gp.INK, label="Mean")
        if var == "fm":
            for i, p in enumerate(prods):
                fam = df.loc[df["product"] == p, "family"].iloc[0]
                ax.plot([i - 0.3, i + 0.3], [config.FMK_DECLARED[fam]] * 2, color=gp.INK, lw=0.8, ls="--")
            ax.plot([], [], color=gp.INK, lw=0.8, ls="--", label=r"Declared $f_{\mathrm{m,k}}$")
        else:
            for i, p in enumerate(prods):
                ax.plot([i - 0.3, i + 0.3], [config.E_DECLARED[p] * 1e-3] * 2, color=gp.INK, lw=0.8, ls="--")
            ax.plot([], [], color=gp.INK, lw=0.8, ls="--", label=r"Declared $E_{0,\mathrm{mean}}$")
        fams = [df.loc[df["product"] == p, "family"].iloc[0] for p in prods]
        ax.set_xticks(np.arange(len(prods)), [f"{p}\n{f}" for p, f in zip(prods, fams)])
        ax.set_ylabel(lab)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        ax.grid(axis="x", visible=False)
    gp.panel_label(axes[0], "a", x=0.97 - 0.03, y=0.97)
    axes[0].texts[-1].set_ha("right")
    gp.panel_label(axes[1], "b", x=0.94, y=0.97)
    axes[1].texts[-1].set_ha("right")
    return gp.save(fig, "fig_box_product")


def main():
    gp.use_latex_style()
    df = data.load()
    ov = product_overview(df)
    gt.save_csv(ov, "T1_product_overview")
    tex = ov[["Product", "Family", "N", "fmk_declared", "E_declared", "b", "h", "hext", "hint",
              "class_ext_int", "fj_ext_int", "glue"]].copy()
    tex.columns = ["No.", "Family", "$N$", "$f_{\\mathrm{m,k}}$", "$E_{0,\\mathrm{mean}}$", "$b$", "$h$",
                   "$h_{\\mathrm{ext}}$", "$h_{\\mathrm{int}}$", "Class ext/int", "Finger joint ext/int",
                   "Lamination adhesive"]
    gt.save_tex(tex.astype(str), "T1_product_overview",
                "Tested products. Declared $f_{\\mathrm{m,k}}$ and $E_{0,\\mathrm{mean}}$ (MPa) are the values "
                "recorded for each product; dimensions (mm), grades and adhesives are those in the test database.",
                "tab:products", column_format="l" * len(tex.columns))

    fam = descriptive_table(df, "family")
    prod = descriptive_table(df, "product")
    gt.save_csv(fam, "T2_descriptive_family")
    gt.save_csv(prod, "T2_descriptive_product")
    for name, d, key in [("T2_descriptive_family", fam, "family"), ("T2_descriptive_product", prod, "product")]:
        t = d[[key, "response", "N", "mean", "sd", "cov", "skew", "exkurt", "q05"]].copy()
        t["response"] = t["response"].map({"fm": "$f_{\\mathrm{m}}$ [MPa]", "E": "$E_{\\mathrm{m,g}}$ [GPa]"})
        t = gt.fmt(t, {"mean": "{:.2f}", "sd": "{:.2f}", "cov": "{:.3f}", "skew": "{:.3f}",
                       "exkurt": "{:.3f}", "q05": "{:.2f}"})
        t.columns = [key.capitalize(), "Response", "$N$", "$\\mu$", "$\\sigma$", "COV", "$\\gamma_1$",
                     "$\\gamma_2$", "$Q_{0.05}$"]
        gt.save_tex(t.astype(str), name,
                    f"Descriptive statistics by {key}: mean $\\mu$, standard deviation $\\sigma$, "
                    "coefficient of variation, skewness $\\gamma_1$, excess kurtosis $\\gamma_2$ and "
                    "empirical 5\\,\\% fractile $Q_{0.05}$.", f"tab:desc_{key}")
    fig_family_ecdf(df)
    fig_box_product(df)
    print(ov.to_string(index=False))
    print(fam.round(3).to_string(index=False))
    print(prod.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
