"""Bayesian calibration of the log-scale model error of manuscript M7.

ln f_m = ln eta(x) - sigma^2/2 + sigma * eps,  eps ~ N(0, 1);  eta fixed at the OLS coefficients.
This conditional error is not the population model used for the partial factors in the paper.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats

from glulam import config, data, models as gm, plotting as gp, tables as gt
from glulam.analyses.regression import selected_m7_terms


def sigma_mle(e):
    """MLE of sigma for e ~ N(-sigma^2/2, sigma^2)."""
    nll = lambda s: -stats.norm.logpdf(e, -s ** 2 / 2, s).sum()
    return optimize.minimize_scalar(nll, bounds=(1e-3, 2), method="bounded").x


def posterior_sigma(e, seed):
    import arviz as az
    import pymc as pm
    with pm.Model():
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        pm.Normal("e", mu=-sigma ** 2 / 2, sigma=sigma, observed=e)
        idata = pm.sample(draws=2000, tune=2000, chains=4, target_accept=0.95, random_seed=seed,
                          progressbar=False)
    s = idata.posterior["sigma"].values.ravel()
    hdi = az.hdi(idata, hdi_prob=0.95)["sigma"].values
    rhat = float(az.rhat(idata)["sigma"].values)
    ess = float(az.ess(idata)["sigma"].values)
    return s, hdi, rhat, ess


def out_of_fold_eta(df, terms, seed):
    """Cross-fitted conditional means (each specimen predicted by a model that did not see it)."""
    _, oof = gm.cross_validate(df, lambda: gm.FixedOLS(terms), config.K_FOLDS, 1, seed)
    return oof


def fig_posterior(df, s, hdi, eta, mu_s):
    fig, axes = gp.figure("double", 0.40, 1, 2)
    ax = axes[0]
    kde = stats.gaussian_kde(s)
    xs = np.linspace(s.min(), s.max(), 300)
    ax.plot(xs, kde(xs), color=gp.PALETTE[0], lw=1.0)
    m = (xs >= hdi[0]) & (xs <= hdi[1])
    ax.fill_between(xs[m], kde(xs[m]), color=gp.PALETTE[0], alpha=0.25, lw=0, label=r"95\,\% HDI")
    ax.axvline(mu_s, color=gp.INK, lw=0.8, label=rf"$\mu_\sigma={mu_s:.3f}$")
    ax.set_xlabel(r"Log-scale model error $\sigma$")
    ax.set_ylabel("Posterior density")
    ax.legend(loc="upper right")
    ax = axes[1]
    med = eta * np.exp(-mu_s ** 2 / 2)
    ax.plot(med, df["fm"], ls="none", marker="o", ms=1.6, mew=0, color=gp.PALETTE[0], alpha=0.7)
    g = np.linspace(20, 75, 100)
    ax.plot(g, g, color=gp.INK, lw=0.8, label="1:1")
    ax.plot(g, g * np.exp(mu_s), color=gp.INK_2, lw=0.6, ls="--", label=r"$\times\exp(\pm\mu_\sigma)$")
    ax.plot(g, g * np.exp(-mu_s), color=gp.INK_2, lw=0.6, ls="--")
    ax.set_xlim(15, 80)
    ax.set_ylim(15, 80)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Conditional median, M8 " + gp.LABEL["fm"])
    ax.set_ylabel("Measured " + gp.LABEL["fm"])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
    gp.panel_label(axes[0], "a")
    gp.panel_label(axes[1], "b")
    return gp.save(fig, "fig_model_error_posterior")


def main():
    gp.use_latex_style()
    df = data.load()
    terms = selected_m7_terms(df)
    m8 = gm.FixedOLS(terms).fit(df)
    eta = m8.predict(df)
    assert (eta > 0).all()
    e_in = np.log(df["fm"].to_numpy()) - np.log(eta)
    s, hdi, rhat, ess = posterior_sigma(e_in, config.SEED)
    eta_oof = out_of_fold_eta(df, terms, config.SEED)
    e_oof = np.log(df["fm"].to_numpy()) - np.log(eta_oof)
    s1 = gm.FixedOLS([("ln_E",), ("ln_h600",), ("cext",)], "log").fit(df)
    res = pd.DataFrame([
        {"quantity": "posterior mean sigma (in-sample residuals)", "value": s.mean()},
        {"quantity": "posterior sd sigma", "value": s.std(ddof=1)},
        {"quantity": "95% HDI lower", "value": hdi[0]},
        {"quantity": "95% HDI upper", "value": hdi[1]},
        {"quantity": "R-hat", "value": rhat},
        {"quantity": "ESS", "value": ess},
        {"quantity": "MLE sigma, in-sample", "value": sigma_mle(e_in)},
        {"quantity": "MLE sigma, cross-fitted (out-of-fold eta)", "value": sigma_mle(e_oof)},
        {"quantity": "residual sd, S1 log model", "value": float(np.sqrt(s1.tr.s2))},
        {"quantity": "sd ln fm, GL24* (no model)", "value": np.log(df.loc[df.family == "GL24*", "fm"]).std(ddof=1)},
        {"quantity": "sd ln fm, all (no model)", "value": np.log(df["fm"]).std(ddof=1)},
    ])
    gt.save_csv(res, "T11_model_error")
    pd.DataFrame({"eta_M8": eta, "eta_M8_oof": eta_oof}).to_csv(config.RESULTS / "M8_eta.csv", index=False)
    fig_posterior(df, s, hdi, eta, s.mean())
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
