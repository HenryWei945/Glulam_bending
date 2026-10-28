"""Material partial factor: conditional k_M check and the population calculation.

The conditional check uses the manuscript M7 mean. It is not reported in the paper or in
Online Resource 1. The partial factors in the paper come from bayes_reliability.py.

(1) Check: conditional resistance m_R = eta_M7(x) (specimen stiffness known), sigma = mu_sigma.
(2) Population-based (EN 1990 Annex C/D, resistance side only): the design value of the bending strength
    of a product population, X_d = exp(mu_ln - alpha_R * beta * sigma_ln), with strengths adjusted to h = 600 mm;
    gamma_M,req = f_m,k / X_d, either with the declared f_m,k or with the product's own characteristic value.
"""

import numpy as np
import pandas as pd

from glulam import config, data, models as gm, plotting as gp, stats as gst, tables as gt
from glulam.analyses.regression import selected_m7_terms


def replication(df, mu_sigma):
    m8 = gm.FixedOLS(selected_m7_terms(df)).fit(df)
    g = df[df["family"] == "GL24*"].copy()
    eta = m8.predict(g)
    fd = eta * np.exp(-0.5 * mu_sigma ** 2 - config.ALPHA_R * config.BETA_TARGET * mu_sigma)
    fmd_ec5 = config.K_MOD * g["kh"] * 24.0 / config.GAMMA_M_EC5
    kM = fmd_ec5 / fd
    summ = {"N": len(g), "kh_mean": g["kh"].mean(), "fd_mean": fd.mean(), "fmd_ec5_mean": fmd_ec5.mean(),
            "kM_mean": kM.mean(), "kM_50": np.median(kM), "kM_sd": kM.std(ddof=1),
            "kM_05": np.quantile(kM, 0.05), "kM_95": np.quantile(kM, 0.95), "frac_below_1": (kM < 1).mean()}
    summ["gamma_M"] = config.GAMMA_M_EC5 * summ["kM_95"]
    return pd.DataFrame([summ]), g.assign(kM=kM.to_numpy())


def population_based(df, group):
    ab = config.ALPHA_R * config.BETA_TARGET
    rows = []
    for key, g in df.groupby(group, observed=True):
        fam = g["family"].iloc[0] if group == "product" else key
        y = np.log(g["fm600"].to_numpy())
        mu, s = y.mean(), y.std(ddof=1)
        xd = np.exp(mu - ab * s)
        fk_own = gst.fk_lognormal(g["fm600"])
        rows.append({group: key, "N": len(g), "sigma_ln": s, "X_d": xd, "fmk_declared": config.FMK_DECLARED[fam],
                     "fk_own": fk_own, "gamma_req_declared": config.FMK_DECLARED[fam] / xd,
                     "gamma_req_own": fk_own / xd,
                     "gamma_lognormal_formula": np.exp((ab - 1.645) * s)})
    return pd.DataFrame(rows)


def fig_kM(g, summ):
    fig, ax = gp.figure("single", 0.80)
    ax.plot(g["fm"], g["kM"], ls="none", marker="o", ms=1.6, mew=0, color=gp.PALETTE[0], alpha=0.7)
    for q, ls in [("kM_50", "--"), ("kM_95", ":")]:
        ax.axhline(summ[q].iloc[0], color=gp.INK, lw=0.7, ls=ls,
                   label=f"$k_{{M,{q[3:]}}}={summ[q].iloc[0]:.3f}$")
    ax.axhline(1.0, color=gp.INK_2, lw=0.5)
    ax.set_xlabel("Measured " + gp.LABEL["fm"])
    ax.set_ylabel(r"$k_M^\ast$")
    ax.legend(loc="upper right")
    return gp.save(fig, "fig_kM_replication")


def main():
    gp.use_latex_style()
    df = data.load()
    me = pd.read_csv(config.RESULTS / "T11_model_error.csv") if (config.RESULTS / "T11_model_error.csv").exists() else None
    mu_sigma = float(me.loc[me["quantity"].str.startswith("posterior mean"), "value"].iloc[0]) if me is not None else 0.178
    summ, g = replication(df, mu_sigma)
    pp = population_based(df, "product")
    pf = population_based(df, "family")
    gt.save_csv(summ, "T12_kM_replication")
    gt.save_csv(pp, "T12_gamma_population_product")
    gt.save_csv(pf, "T12_gamma_population_family")
    fig_kM(g, summ)
    with pd.option_context("display.width", 200):
        print(summ.round(3).to_string(index=False))
        print(pp.round(3).to_string(index=False))
        print(pf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
