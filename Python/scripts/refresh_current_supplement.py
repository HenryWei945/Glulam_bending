"""Regenerate supplementary results for the seven formulas in the current paper.

This uses the reported full-sample formulas, not a new stepwise search. Outputs
are isolated from the legacy model-numbered tables. The main hierarchical
Bayesian population analysis and the manuscript's numbers.tex are not changed.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy import integrate, optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glulam import config, data, models as gm, plotting as gp
from glulam.analyses.model_error import sigma_mle
from glulam.analyses.partial_factor import population_based

OUT = ROOT / "outputs" / "consistency_20261028"
TABLES = OUT / "tables"
RESULTS = OUT / "results"

NUM5 = ["E", "h", "hext", "hint", "b"]
TERMS = {
    "M1": [("E",)],
    "M2": [(v,) for v in NUM5],
    "M3": [(v,) for v in NUM5 + ["l", "a", "c"]],
    "M4": [("E",), ("b",), ("h",), ("hext",), ("hint",),
           ("E", "hext"), ("b", "h"), ("h", "hext")],
    "M5": [("E",), ("b",), ("h",), ("product",), ("b", "h")],
    "M6": [("E",), ("Eb",), ("Eh",), ("Eoh",), ("hint",),
           ("hint_frac",), ("Eb", "Eh")],
    "M7": [("EZ7",), ("Eb",), ("Eh",), ("Eh2",), ("Eoh",),
           ("hint_frac",), ("Eh", "hint_frac"), ("hint_frac", "hint_frac")],
}
SYMBOL = {
    "E": r"E_{\mathrm{m,g}}", "h": "h", "b": "b", "l": "l", "a": "a", "c": "c",
    "hext": r"h_{\mathrm{ext}}", "hint": r"h_{\mathrm{int}}",
    "Eb": r"E_{\mathrm{m,g}}b", "Eh": r"E_{\mathrm{m,g}}h",
    "Eh2": r"E_{\mathrm{m,g}}h^{2}", "Eoh": r"E_{\mathrm{m,g}}/h",
    "EZ7": r"E_{\mathrm{m,g}}h^{2}/b", "hint_frac": r"h_{\mathrm{int}}/h",
    "product": r"\text{product indicators}",
}


def atom_tex(name):
    """Parenthesise ratios so a later product does not change the grouping."""
    symbol = SYMBOL[name]
    if "/" in symbol:
        return "(" + symbol + ")"
    return symbol


def term_tex(term):
    if len(term) == 2 and term[0] == term[1]:
        return "$(" + SYMBOL[term[0]] + ")^{2}$"
    return "$" + r"\times ".join(atom_tex(v) for v in term) + "$"


def table(frame, stem, caption, colfmt=None, note=None, size=r"\footnotesize"):
    body = frame.to_latex(index=False, escape=False, column_format=colfmt, na_rep="--")
    parts = [r"\begin{table}[htbp]", r"\centering", size,
             r"\caption{" + caption + "}", body]
    if note:
        parts += [r"\par\smallskip\raggedright\scriptsize " + note]
    parts.append(r"\end{table}")
    (TABLES / (stem + ".tex")).write_text("\n".join(parts) + "\n", encoding="utf-8")


def posterior_quadrature(residual, points=60001):
    """Same N(-sigma^2/2,sigma^2) likelihood and HN(1) prior as model_error.py.

    Terms independent of sigma are omitted. All summaries are deterministic
    integrals of the scalar posterior; no MCMC diagnostics apply.
    """
    n = len(residual)
    sum_squares = float(np.dot(residual, residual))

    def logp(sigma):
        return -n * np.log(sigma) - sum_squares / (2 * sigma ** 2) - (n / 8 + .5) * sigma ** 2

    mode = optimize.minimize_scalar(lambda v: -logp(v), bounds=(.001, 2), method="bounded").x
    grid = np.linspace(mode * .3, mode * 3, points)
    density = np.exp(logp(grid) - logp(mode))
    density /= integrate.trapezoid(density, grid)
    cdf = integrate.cumulative_trapezoid(density, grid, initial=0)
    cdf /= cdf[-1]
    q = lambda p: np.interp(p, cdf, grid)
    tail = optimize.minimize_scalar(lambda a: q(a + .95) - q(a), bounds=(0, .05), method="bounded").x
    hdi = np.array([q(tail), q(tail + .95)])
    mean = integrate.trapezoid(grid * density, grid)
    sd = np.sqrt(integrate.trapezoid((grid - mean) ** 2 * density, grid))
    return grid, density, cdf, float(mean), float(sd), hdi


def figures(df, eta, grid, density, mean_sigma, hdi, km, summary):
    old_output = config.FIGURES
    config.FIGURES = OUT / "figures"
    config.FIGURES.mkdir(exist_ok=True)
    gp.use_latex_style()
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": False, "font.serif": ["STIXGeneral"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False})
    fig, axes = gp.figure("double", 0.40, 1, 2)
    central = density > density.max() * 1e-6
    xs, density = grid[central], density[central]
    axes[0].plot(xs, density, color=gp.PALETTE[0])
    mask = (xs >= hdi[0]) & (xs <= hdi[1])
    axes[0].fill_between(xs[mask], density[mask], alpha=.25, color=gp.PALETTE[0], label="95% HDI")
    axes[0].axvline(mean_sigma, color=gp.INK, lw=.8)
    axes[0].set_xlabel(r"Log-scale model-error standard deviation $\sigma_\varepsilon$")
    axes[0].set_ylabel("Posterior density")
    axes[0].legend()
    median = eta * np.exp(-mean_sigma ** 2 / 2)
    axes[1].plot(median, df.fm, ls="none", marker="o", ms=1.6, mew=0, alpha=.65)
    g = np.linspace(15, 80, 100)
    axes[1].plot(g, g, color=gp.INK, lw=.8)
    axes[1].set(xlim=(15, 80), ylim=(15, 80),
                xlabel="Conditional median, M7 [MPa]", ylabel="Measured bending strength [MPa]")
    axes[1].set_aspect("equal", adjustable="box")
    for ax, letter in zip(axes, ["a", "b"]):
        gp.panel_label(ax, letter)
    gp.save(fig, "fig_current_model_error")

    fig, ax = gp.figure("single", .80)
    ax.plot(km.fm, km.kM, ls="none", marker="o", ms=1.6, mew=0, alpha=.65)
    for key, label, ls in [("kM_50", "50", "--"), ("kM_95", "95", ":")]:
        val = summary[key]
        ax.axhline(val, color=gp.INK, lw=.7, ls=ls, label=f"$k_{{M,{label}}}={val:.3f}$")
    ax.axhline(1., color=gp.INK_2, lw=.5)
    ax.set_xlabel("Measured " + gp.LABEL["fm"])
    ax.set_ylabel(r"$k_M^\ast$")
    ax.legend()
    gp.save(fig, "fig_current_kM")
    config.FIGURES = old_output


def main():
    for p in [TABLES, RESULTS]:
        p.mkdir(parents=True, exist_ok=True)
    df = data.load()
    assert len(df) == 1054
    selected = pd.DataFrame({"Model": list(TERMS),
                             "Retained terms": [" + ".join(term_tex(t) for t in ts) for ts in TERMS.values()]})
    table(selected, "current_terms", "Formulas used in the current manuscript. All formulas include an intercept. "
          "Terms refer to predictors, not to the number of estimated coefficients; the product factor in M5 "
          "has six indicator columns. Lengths are in mm and $E_{\\mathrm{m,g}}$ is in MPa.",
          colfmt=r"lp{0.82\linewidth}", size=r"\scriptsize")

    coefficients, vifs, comparison = [], [], []
    fits = {}
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    for model, ts in TERMS.items():
        fit = gm.FixedOLS(ts).fit(df)
        fits[model] = fit
        for name, value in zip(fit.colnames, fit.beta):
            coefficients.append({"model": model, "column": name, "estimate": value})
        folds, _ = gm.cross_validate(df, lambda ts=ts: gm.FixedOLS(ts), 5, 10, config.SEED)
        comparison.append({"model": model, "terms": len(ts), **gm.summarise_folds(folds)})
        if model in {"M6", "M7"}:
            X, names = gm.design(df, ts, fit.lv)
            X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)
            for k in range(1, X.shape[1]):
                vifs.append({"model": model, "column": names[k],
                             "VIF": variance_inflation_factor(X, k)})
    pd.DataFrame(comparison).to_csv(RESULTS / "current_model_comparison.csv", index=False)
    co = pd.DataFrame(coefficients)
    co.to_csv(RESULTS / "current_coefficients.csv", index=False)
    for model in TERMS:
        t = co[co.model == model][["column", "estimate"]].copy()
        t["column"] = t.column.str.replace("_", r"\_", regex=False)
        t["estimate"] = t.estimate.map(lambda v: f"{v:.8g}")
        t.columns = ["Design-matrix column", "Coefficient"]
        table(t, "current_coefficients_" + model, f"Least-squares coefficients of {model} on the full sample. "
              "Code E denotes $E_{\\mathrm{m,g}}$; other code names are defined in Section S3.")
    vf = pd.DataFrame(vifs)
    vf.to_csv(RESULTS / "current_vif.csv", index=False)
    t = vf.copy()
    t["column"] = t.column.str.replace("_", r"\_", regex=False)
    t["VIF"] = t.VIF.map(lambda v: f"{v:.2f}")
    t.columns = ["Model", "Design-matrix column", "VIF"]
    table(t, "current_vif", "Variance inflation factors for the reported M6 and M7 formulas.")

    m7 = fits["M7"]
    eta = m7.predict(df)
    assert np.all(eta > 0)
    residual = np.log(df.fm.to_numpy()) - np.log(eta)
    print("Integrating the scalar model-error posterior for the current M7 formula", flush=True)
    grid, density, cdf, mean_sigma, sd_sigma, hdi = posterior_quadrature(residual)
    _, _, _, mean_coarse, _, _ = posterior_quadrature(residual, points=30001)
    assert abs(mean_sigma - mean_coarse) < 1e-8
    _, eta_oof = gm.cross_validate(df, lambda: gm.FixedOLS(TERMS["M7"]), 5, 1, config.SEED)
    assert np.all(eta_oof > 0)
    out_residual = np.log(df.fm.to_numpy()) - np.log(eta_oof)
    me = pd.DataFrame([
        ("Posterior mean of sigma_epsilon", mean_sigma),
        ("Posterior SD of sigma_epsilon", sd_sigma),
        ("95% HDI lower", hdi[0]), ("95% HDI upper", hdi[1]),
        ("MLE sigma_epsilon, full-sample residuals", sigma_mle(residual)),
        ("MLE sigma_epsilon, fixed-formula OOF residuals", sigma_mle(out_residual)),
        ("SD of ln f_m, GL24*", np.log(df.loc[df.family == "GL24*", "fm"]).std(ddof=1)),
        ("SD of ln f_m, all specimens", np.log(df.fm).std(ddof=1)),
    ], columns=["quantity", "value"])
    me.to_csv(RESULTS / "current_model_error.csv", index=False)
    pd.DataFrame({"sigma_epsilon": grid, "density": density, "cdf": cdf}).to_csv(
        RESULTS / "current_model_error_posterior.csv", index=False)
    t = me.copy()
    t["quantity"] = t.quantity.str.replace("%", r"\%", regex=False).str.replace("_", r"\_", regex=False)
    t["value"] = t.value.map(lambda x: f"{x:.4f}")
    t.columns = ["Quantity", "Value"]
    table(t, "current_model_error", "Lognormal model error of the current M7 conditional-mean formula, "
          "with its least-squares coefficients held fixed. The one-dimensional posterior is integrated "
          "numerically; $\\sigma_\\varepsilon$ has a half-normal prior with scale 1.")

    g = df[df.family == "GL24*"].copy()
    eta_g = m7.predict(g)
    fd = eta_g * np.exp(-.5 * mean_sigma ** 2 - config.ALPHA_R * config.BETA_TARGET * mean_sigma)
    fmd = config.K_MOD * g.kh * 24. / config.GAMMA_M_EC5
    g["kM"] = np.asarray(fmd / fd)
    summary = {"N": len(g), "sigma_epsilon_mean": mean_sigma,
               "kh_mean": float(g.kh.mean()), "fd_mean": float(fd.mean()), "fmd_ec5_mean": float(fmd.mean()),
               "kM_50": float(g.kM.median()), "kM_95": float(g.kM.quantile(.95))}
    summary["gamma_M_conditional"] = config.GAMMA_M_EC5 * summary["kM_95"]
    pd.DataFrame([summary]).to_csv(RESULTS / "current_conditional_factor.csv", index=False)
    pd.DataFrame({"eta_M7": eta, "eta_M7_oof": eta_oof}).to_csv(RESULTS / "current_M7_predictions.csv", index=False)
    t = pd.DataFrame([(k.replace("_", r"\_"), f"{v:.4f}") for k, v in summary.items()], columns=["Quantity", "Value"])
    table(t, "current_conditional_factor", "Conditional partial-factor calculation for GL24* using the "
          "current M7 conditional mean and posterior mean of $\\sigma_\\varepsilon$. "
          "This conditional calculation is not used for the population-based design conclusions.")
    pp = population_based(df, "product")
    pp.to_csv(RESULTS / "current_frequentist_population.csv", index=False)
    t = pp[["product", "N", "sigma_ln", "X_d", "gamma_req_declared"]].copy()
    for c in ["sigma_ln", "X_d", "gamma_req_declared"]:
        t[c] = t[c].map(lambda v: f"{v:.4f}")
    t.columns = ["Product", "$N$", "$s_y$", "$X_d$ [MPa]", r"$\gamma_{M,\mathrm{req}}$"]
    table(t, "current_frequentist_population", "Separate frequentist population calculation after dividing "
          "strengths by the 2004 depth factor. This reference calculation differs from the main paper's "
          "hierarchical model of strength as tested and is independent of M7.")
    figures(df, eta, grid, density, mean_sigma, hdi, g, summary)
    manifest = {"sample_size": len(df), "models": {k: [list(t) for t in v] for k, v in TERMS.items()},
                "model_error_method": "one-dimensional posterior quadrature",
                "quadrature_mean_grid_difference": abs(mean_sigma - mean_coarse), "results": comparison}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Current supplement regenerated", flush=True)
    print(pd.DataFrame(comparison)[["model", "terms", "RMSE_mean", "R2_mean"]].to_string(index=False), flush=True)
    print("VIF maxima", vf.groupby("model").VIF.max().to_dict(), flush=True)
    print("Posterior mean sigma_epsilon / HDI", mean_sigma, hdi.tolist(), flush=True)


if __name__ == "__main__":
    main()
