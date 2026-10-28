"""Hierarchical Bayesian lognormal model of the product populations and implied material partial factors.

ln f_m,ij ~ N(mu_j, sigma_j^2),  mu_j = mu_0 + tau z_j,  ln sigma_j = m_s + s_s w_j   (j = product)

The resistance is the population strength of a product (the stiffness of a member is unknown at design),
parameter uncertainty is propagated, and partial pooling replaces the pooled GL24* distribution. For each product:
    X_d = exp(mu_j - alpha_R beta sigma_j)                        (EN 1990 design value, resistance side)
    gamma_M,req = f_m,k,declared / X_d                             (partial factor implied by the declared value)
    gamma_M,var = exp((alpha_R beta - 1.645) sigma_j)              (variability part only)
A frequentist check uses the EN 1990 Annex D predictive factor k_d,n (V_X unknown).
"""

import numpy as np
import pandas as pd
from scipy import stats

from glulam import config, data, plotting as gp, tables as gt

AB = config.ALPHA_R * config.BETA_TARGET
P_D = stats.norm.cdf(-AB)


def fit_hierarchical(df, response="fm", seed=config.SEED):
    import arviz as az
    import pymc as pm
    prods = list(df["product"].cat.categories)
    idx = df["product"].cat.codes.to_numpy()
    y = np.log(df[response].to_numpy())
    with pm.Model():
        mu0 = pm.Normal("mu0", mu=np.log(40.0), sigma=0.5)
        tau = pm.HalfNormal("tau", sigma=0.3)
        ms = pm.Normal("m_s", mu=np.log(0.2), sigma=0.5)
        ss = pm.HalfNormal("s_s", sigma=0.5)
        z = pm.Normal("z", 0, 1, shape=len(prods))
        w = pm.Normal("w", 0, 1, shape=len(prods))
        mu = pm.Deterministic("mu", mu0 + tau * z)
        sig = pm.Deterministic("sigma", pm.math.exp(ms + ss * w))
        pm.Normal("y", mu=mu[idx], sigma=sig[idx], observed=y)
        idata = pm.sample(draws=2000, tune=2000, chains=4, target_accept=0.95, random_seed=seed,
                          progressbar=False)
    summ = az.summary(idata, var_names=["mu0", "tau", "m_s", "s_s", "mu", "sigma"], hdi_prob=0.90)
    post = idata.posterior
    draws = {k: post[k].values.reshape(-1, *post[k].shape[2:]) for k in ["mu0", "tau", "m_s", "s_s", "mu", "sigma"]}
    return prods, draws, summ


def product_reliability(df, prods, d):
    rows = []
    q = lambda a: np.percentile(a, [5, 50, 95])
    for j, p in enumerate(prods):
        fam = df.loc[df["product"] == p, "family"].iloc[0]
        mu, s = d["mu"][:, j], d["sigma"][:, j]
        f05 = np.exp(mu - 1.645 * s)
        xd = np.exp(mu - AB * s)
        g_req = config.FMK_DECLARED[fam] / xd
        g_var = np.exp((AB - 1.645) * s)
        lo, med, hi = q(g_req)
        rows.append({"product": p, "family": fam, "fmk_declared": config.FMK_DECLARED[fam],
                     "sigma_ln_med": np.median(s), "f05_med": np.median(f05), "f05_lo": q(f05)[0],
                     "f05_hi": q(f05)[2], "P_f05_below_declared": np.mean(f05 < config.FMK_DECLARED[fam]),
                     "Xd_med": np.median(xd), "gamma_req_lo": lo, "gamma_req_med": med, "gamma_req_hi": hi,
                     "gamma_var_lo": q(g_var)[0], "gamma_var_med": q(g_var)[1], "gamma_var_hi": q(g_var)[2]})
    return pd.DataFrame(rows)


def new_product(d, rng, n_mc=400):
    """Predictive design quantities for a product not in the database (random mu_j, sigma_j)."""
    n = d["mu0"].size
    take = rng.choice(n, 2000, replace=False)
    mu_new = d["mu0"][take] + d["tau"][take] * rng.standard_normal(take.size)
    s_new = np.exp(d["m_s"][take] + d["s_s"][take] * rng.standard_normal(take.size))
    xd = np.exp(mu_new - AB * s_new)
    f05 = np.exp(mu_new - 1.645 * s_new)
    g_var = np.exp((AB - 1.645) * s_new)
    # specimen drawn from a random new product: mixture over products, per posterior draw
    mix_f05, mix_xd = [], []
    for i in range(0, take.size, 10):
        mus = d["mu0"][take[i]] + d["tau"][take[i]] * rng.standard_normal(n_mc)
        ss = np.exp(d["m_s"][take[i]] + d["s_s"][take[i]] * rng.standard_normal(n_mc))
        yv = (mus[:, None] + ss[:, None] * rng.standard_normal((n_mc, 200))).ravel()
        mix_f05.append(np.exp(np.quantile(yv, 0.05)))
        mix_xd.append(np.exp(np.quantile(yv, P_D)))
    q = lambda a: np.percentile(a, [5, 50, 95])
    return pd.DataFrame([
        {"quantity": "new product: sigma_ln", **dict(zip(["lo", "med", "hi"], q(s_new)))},
        {"quantity": "new product: f05 [MPa]", **dict(zip(["lo", "med", "hi"], q(f05)))},
        {"quantity": "new product: X_d [MPa]", **dict(zip(["lo", "med", "hi"], q(xd)))},
        {"quantity": "new product: gamma_var", **dict(zip(["lo", "med", "hi"], q(g_var)))},
        {"quantity": "random product mixture: f05 [MPa]", **dict(zip(["lo", "med", "hi"], q(mix_f05)))},
        {"quantity": "random product mixture: X_d [MPa]", **dict(zip(["lo", "med", "hi"], q(mix_xd)))},
        {"quantity": "random product mixture: f05/X_d", **dict(zip(["lo", "med", "hi"],
                                                                     q(np.array(mix_f05) / np.array(mix_xd))))},
    ])


def annex_d(df, response="fm"):
    """EN 1990 Annex D design value with V_X unknown (Bayesian predictive with vague prior)."""
    rows = []
    for p, g in df.groupby("product", observed=True):
        fam = g["family"].iloc[0]
        y = np.log(g[response].to_numpy())
        n = y.size
        kdn = stats.t.ppf(1 - P_D, n - 1) * np.sqrt(1 + 1 / n)
        xd = np.exp(y.mean() - kdn * y.std(ddof=1))
        rows.append({"product": p, "N": n, "k_dn": kdn, "Xd_annexD": xd,
                     "gamma_req_annexD": config.FMK_DECLARED[fam] / xd})
    return pd.DataFrame(rows)


def fig_forest(rel, newp):
    fig, axes = gp.figure("double", 0.40, 1, 2, sharey=True)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=10)
    ylab = list(rel["product"] + " (" + rel["family"] + ")") + ["New product"]
    yy = np.arange(len(ylab))[::-1]
    ax = axes[0]
    for k, r in rel.iterrows():
        st = gp.product_style(r["product"])
        ax.errorbar(r["f05_med"], yy[k], xerr=[[r["f05_med"] - r["f05_lo"]], [r["f05_hi"] - r["f05_med"]]],
                    fmt=st["marker"], color=st["color"], ms=3.5, mec=gp.INK, mew=0.3, elinewidth=0.9, capsize=1.5)
        ax.plot(r["fmk_declared"], yy[k], marker="|", ms=8, mew=1.2, color=gp.INK)
    nf = newp.set_index("quantity").loc["new product: f05 [MPa]"]
    ax.errorbar(nf["med"], yy[-1], xerr=[[nf["med"] - nf["lo"]], [nf["hi"] - nf["med"]]], fmt="o", mfc="white",
                color=gp.INK, ms=3.5, elinewidth=0.9, capsize=1.5)
    ax.plot([], [], marker="|", ls="none", ms=8, mew=1.2, color=gp.INK, label=r"Declared $f_{\mathrm{m,k}}$")
    ax.set_yticks(yy, ylab)
    ax.set_xlim(21.5, 36.8)
    # Tested-product bars are 90% equal-tailed credible intervals. The new-product
    # bar is a 90% posterior predictive interval. One axis must name both.
    ax.set_xlabel("5\\% fractile of $f_{\\mathrm{m}}$ [MPa]\n(tested: 90\\% CrI; new: predictive)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=9)
    ax = axes[1]
    for k, r in rel.iterrows():
        st = gp.product_style(r["product"])
        ax.errorbar(r["gamma_req_med"], yy[k] + 0.12,
                    xerr=[[r["gamma_req_med"] - r["gamma_req_lo"]], [r["gamma_req_hi"] - r["gamma_req_med"]]],
                    fmt=st["marker"], color=st["color"], ms=3.5, mec=gp.INK, mew=0.3, elinewidth=0.9, capsize=1.5,
)
        ax.errorbar(r["gamma_var_med"], yy[k] - 0.12,
                    xerr=[[r["gamma_var_med"] - r["gamma_var_lo"]], [r["gamma_var_hi"] - r["gamma_var_med"]]],
                    fmt=st["marker"], mfc="white", color=st["color"], ms=3.2, mew=0.6, elinewidth=0.6, capsize=1.0,
)
    ng = newp.set_index("quantity").loc["new product: gamma_var"]
    ax.errorbar(ng["med"], yy[-1] - 0.12, xerr=[[ng["med"] - ng["lo"]], [ng["hi"] - ng["med"]]], fmt="o",
                mfc="white", color=gp.INK, ms=3.2, elinewidth=0.6, capsize=1.0)
    ax.axvline(config.GAMMA_M_EC5, color=gp.INK_2, lw=0.7, ls="--")
    ax.set_xlim(0.85, 1.65)
    ax.set_xlabel("Implied material partial factor\n($\\beta=3.8$, $\\alpha_R=0.8$)")
    from matplotlib.lines import Line2D
    # Filled and open symbols share one shape per product, so a single circle
    # swatch would not identify them. The caption states which fill is which.
    handles = [Line2D([], [], color=gp.INK_2, ls="--", lw=0.7, label=r"EN 1995-1-1: $\gamma_M=1.25$")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=9)
    gp.panel_label(axes[0], "a", x=0.03, y=0.97)
    gp.panel_label(axes[1], "b", x=0.97, y=0.97)
    axes[0].texts[-1].set_fontsize(10)
    axes[1].texts[-1].set_fontsize(10)
    axes[1].texts[-1].set_ha("right")
    for ax in axes:
        ax.xaxis.label.set_fontsize(10)
    return gp.save(fig, "fig_bayes_reliability")


def main(response="fm", tag=""):
    gp.use_latex_style()
    df = data.load()
    rng = np.random.default_rng(config.SEED)
    prods, d, summ = fit_hierarchical(df, response)
    rel = product_reliability(df, prods, d)
    newp = new_product(d, rng)
    ad = annex_d(df, response)
    rel = rel.merge(ad, on="product")
    gt.save_csv(summ.reset_index(), f"T13_bayes_summary{tag}")
    gt.save_csv(rel, f"T13_bayes_reliability{tag}")
    gt.save_csv(newp, f"T13_bayes_new_product{tag}")
    if not tag:
        fig_forest(rel, newp)
        t = rel[["product", "family", "fmk_declared", "sigma_ln_med", "f05_med", "P_f05_below_declared",
                 "Xd_med", "gamma_req_med", "gamma_req_lo", "gamma_req_hi", "gamma_var_med", "gamma_req_annexD"]].copy()
        t["gamma_req"] = t.apply(lambda r: f"{r.gamma_req_med:.2f} [{r.gamma_req_lo:.2f}, {r.gamma_req_hi:.2f}]", axis=1)
        t = gt.fmt(t, {"fmk_declared": "{:.0f}", "sigma_ln_med": "{:.3f}", "f05_med": "{:.1f}",
                       "P_f05_below_declared": "{:.2f}", "Xd_med": "{:.1f}", "gamma_var_med": "{:.2f}",
                       "gamma_req_annexD": "{:.2f}"})
        t = t[["product", "family", "fmk_declared", "sigma_ln_med", "f05_med", "P_f05_below_declared", "Xd_med",
               "gamma_req", "gamma_var_med", "gamma_req_annexD"]]
        t.columns = ["Product", "Family", "$f_{\\mathrm{m,k,decl}}$", "$\\sigma_{\\ln}$", "$f_{0.05}$",
                     "$P(f_{0.05}<f_{\\mathrm{m,k,decl}})$", "$X_d$", "$\\gamma_{M,\\mathrm{req}}$ [90\\,\\% CrI]",
                     "$\\gamma_{M,\\mathrm{var}}$", "$\\gamma_{M,\\mathrm{req}}$ (Annex D)"]
        gt.save_tex(t.astype(str), "T13_bayes_reliability",
                    "Hierarchical Bayesian lognormal model of the product populations (posterior medians; "
                    "strengths in MPa, as tested) and material partial factors implied by $\\beta=3.8$, "
                    "$\\alpha_R=0.8$ on the resistance side.", "tab:bayes")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summ.round(3).to_string())
        print(rel.round(3).to_string(index=False))
        print(newp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
