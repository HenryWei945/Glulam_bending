"""Distribution fitting with bootstrap goodness-of-fit, lower-tail diagnostics and the GL24* mixture check."""

import numpy as np
import pandas as pd
from scipy import optimize

from glulam import config, data, plotting as gp, stats as gst, tables as gt

DIST_STYLE = {
    "Normal": dict(color=gp.PALETTE[0], ls="-", marker="o"),
    "Lognormal": dict(color=gp.PALETTE[1], ls="--", marker="^"),
    "Gamma": dict(color=gp.PALETTE[2], ls="-.", marker="s"),
    "Weibull": dict(color=gp.PALETTE[3], ls=":", marker="D"),
}


def gof_all(df, group, rng):
    rows = []
    for key, g in df.groupby(group, observed=True):
        for var, scale in [("fm", 1.0), ("E", 1e-3)]:
            t = gst.gof_table(g[var].to_numpy() * scale, config.N_BOOT, rng)
            t.insert(0, "response", var)
            t.insert(0, group, key)
            t.insert(2, "N", len(g))
            rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out["dAIC"] = out["AIC"] - out.groupby([group, "response"])["AIC"].transform("min")
    return out


def mixture_q05(df):
    """5 % fractile of GL24*: pooled lognormal vs. a mixture of product-wise lognormals vs. empirical."""
    g = df[df["family"] == "GL24*"]
    x = g["fm"].to_numpy()
    pooled = gst.fit("Lognormal", x)
    comps = [(len(gp_) / len(g), gst.fit("Lognormal", gp_["fm"].to_numpy()))
             for _, gp_ in g.groupby("product", observed=True)]

    def mix_cdf(v):
        return sum(w * d.cdf(v) for w, d in comps)

    q_mix = optimize.brentq(lambda v: mix_cdf(v) - 0.05, 1, 100)
    return pd.DataFrame([{
        "Q05_empirical": gst.q05_order(x),
        "Q05_pooled_lognormal": pooled.ppf(0.05),
        "Q05_product_mixture": q_mix,
        "sigma_ln_pooled": np.log(x).std(),
        "sigma_ln_within_mean": np.sqrt(sum(w * np.log(gp_["fm"]).var(ddof=0)
                                            for (w, _), (_, gp_) in zip(comps, g.groupby("product", observed=True)))),
    }])


def fig_gof_gl24(df):
    g = df[df["family"] == "GL24*"]
    fig, axes = gp.figure("double", 0.62, 2, 3)
    for row, (var, scale, lab) in enumerate([("fm", 1, gp.LABEL["fm"]), ("E", 1e-3, gp.LABEL["E"])]):
        x = np.sort(g[var].to_numpy() * scale)
        n = x.size
        u = (np.arange(1, n + 1) - 0.5) / n
        fits = {d: gst.fit(d, x) for d in gst.DISTRIBUTIONS}
        grid = np.linspace(x.min() - 0.1 * np.ptp(x), x.max() + 0.1 * np.ptp(x), 400)
        grid = grid[grid > 0]

        ax = axes[row, 0]
        ax.hist(x, bins=30, density=True, color="#d9d8d4", edgecolor="white", lw=0.3, label="Data")
        for d, fz in fits.items():
            st = DIST_STYLE[d]
            ax.plot(grid, fz.pdf(grid), color=st["color"], ls=st["ls"], lw=0.9, label=d)
        ax.set_xlabel(lab)
        ax.set_ylabel("Probability density")

        ax = axes[row, 1]
        for d, fz in fits.items():
            st = DIST_STYLE[d]
            ax.plot(fz.ppf(u), x, ls="none", marker=st["marker"], ms=1.6, mew=0, color=st["color"], label=d)
        lim = [min(x.min(), min(f.ppf(u[0]) for f in fits.values())),
               max(x.max(), max(f.ppf(u[-1]) for f in fits.values()))]
        ax.plot(lim, lim, color=gp.INK, lw=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Fitted quantile" + (" [MPa]" if var == "fm" else " [GPa]"))
        ax.set_ylabel("Empirical quantile" + (" [MPa]" if var == "fm" else " [GPa]"))

        ax = axes[row, 2]
        k = int(np.ceil(0.10 * n))                       # lowest 10 % of the observations
        for d, fz in fits.items():
            st = DIST_STYLE[d]
            ax.plot(fz.ppf(u[:k]), x[:k], ls="none", marker=st["marker"], ms=2.0, mew=0, color=st["color"],
                    label=d)
        lim = [min(x[0], min(f.ppf(u[0]) for f in fits.values())) * 0.97,
               max(x[k - 1], max(f.ppf(u[k - 1]) for f in fits.values())) * 1.02]
        ax.plot(lim, lim, color=gp.INK, lw=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Fitted quantile, lower 10\\,\\%")
        ax.set_ylabel("Empirical quantile, lower 10\\,\\%")
    axes[0, 0].legend(loc="upper right")
    axes[0, 2].legend(loc="lower right", markerscale=1.8, handletextpad=0.2)
    for ax, lab in zip(axes.ravel(), "abcdef"):
        gp.panel_label(ax, lab)
    return gp.save(fig, "fig_gof_gl24")


def main():
    gp.use_latex_style()
    df = data.load()
    rng = np.random.default_rng(config.SEED)
    fam = gof_all(df, "family", rng)
    prod = gof_all(df, "product", rng)
    gt.save_csv(fam, "T3_gof_family")
    gt.save_csv(prod, "T3_gof_product")
    mix = mixture_q05(df)
    gt.save_csv(mix, "T3_gl24_mixture_q05")

    t = fam[["family", "response", "dist", "KS", "KS_p", "CvM", "CvM_p", "AD", "AD_p", "AIC", "BIC"]].copy()
    t["response"] = t["response"].map({"fm": "$f_{\\mathrm{m}}$", "E": "$E_{\\mathrm{m,g}}$"})
    t = gt.fmt(t, {"KS": "{:.3f}", "KS_p": "{:.3f}", "CvM": "{:.3f}", "CvM_p": "{:.3f}", "AD": "{:.3f}",
                   "AD_p": "{:.3f}", "AIC": "{:.0f}", "BIC": "{:.0f}"})
    t.columns = ["Family", "Response", "Distribution", "KS", "$p_{\\mathrm{KS}}$", "CvM", "$p_{\\mathrm{CvM}}$",
                 "AD", "$p_{\\mathrm{AD}}$", "AIC", "BIC"]
    gt.save_tex(t.astype(str), "T3_gof_family",
                "Goodness of fit by family (maximum-likelihood fits). $p$-values from a parametric bootstrap "
                f"with re-estimated parameters ($B={config.N_BOOT}$).", "tab:gof")
    fig_gof_gl24(df)

    best = fam.loc[fam.groupby(["family", "response"])["AIC"].idxmin(), ["family", "response", "dist", "CvM_p", "AD_p"]]
    print(best.to_string(index=False))
    bestp = prod[prod.response == "fm"].loc[lambda d: d.groupby("product")["AIC"].idxmin()]
    print(bestp[["product", "dist", "KS_p", "CvM_p", "AD_p", "dAIC"]].to_string(index=False))
    print(prod[(prod.response == "fm") & (prod.dist == "Lognormal")][["product", "CvM_p", "AD_p", "dAIC"]].to_string(index=False))
    print(mix.round(3).to_string(index=False))
    print(fam[fam.family == "GL24*"][["response", "dist", "KS", "KS_p", "CvM", "CvM_p", "AD", "AD_p", "AIC"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
