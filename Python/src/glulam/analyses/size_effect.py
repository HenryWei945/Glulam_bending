"""Depth effect on the cleaned sample, using the legacy family-mean power law.

The fit follows ANOVA TESTS/SIZE EFFECT/MAIN.m: one row per family and depth,
an uncapped law mean f_m = A (600/h)^n with weights proportional to the cell
counts, a 95% band from the same local linearisation, and one-amplitude
scalings of the EN 1995-1-1:2004 and Vida et al. (2023) shapes. Two coding
errors in that script are not reproduced: each cell is stored once, and the
counts stay aligned with the cells. Specimen scatter is drawn as +/- 1 SD.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize, stats

from glulam import config, data, plotting as gp, tables as gt


# Grey band of the fitted mean.
BAND_RGB = (0.62, 0.62, 0.62)


def kh_ec5(h, cap=True):
    """EN 1995-1-1:2004 depth factor; the cap at 1.1 is optional."""
    k = (config.H_REF / np.asarray(h, float)) ** config.KH_EXPONENT
    return np.where(np.asarray(h) < config.H_REF, np.minimum(k, config.KH_CAP) if cap else k, 1.0)


def family_depth_table(df):
    """Arithmetic mean, sample SD and count at each family-depth combination.

    MATLAB std (and pandas std) divides by N-1. A cell with one specimen has
    no sample SD; the legacy script plotted that bar as zero.
    """
    grouped = (df.groupby(["family", "h"], observed=True)["fm"]
               .agg(mean="mean", std="std", size="size").reset_index())
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def _power_fit(H, M, N):
    """Weighted nonlinear least squares and the legacy parameter covariance.

    Weights are N/mean(N), so they average to 1 and the residual variance is
    on the scale of the cell means. The Jacobian is that of the weighted
    residuals, and the t critical value uses G-2 degrees of freedom.
    """
    H = np.asarray(H, float)
    M = np.asarray(M, float)
    weight = np.asarray(N, float)
    weight = weight / weight.mean()
    sqrt_weight = np.sqrt(weight)

    def mean_strength(h, amplitude, exponent):
        return amplitude * (config.H_REF / h) ** exponent

    start = np.array([float(np.mean(M)), config.KH_EXPONENT])
    estimate = optimize.minimize(
        lambda p: np.sum(weight * (M - mean_strength(H, p[0], p[1])) ** 2),
        start, method="Nelder-Mead")
    amplitude, exponent = estimate.x
    ratio = config.H_REF / H
    # r = sqrt(w) * (M - y); J = d r / d(A, n) = -sqrt(w) * dy/d(A, n).
    dy_damplitude = ratio ** exponent
    dy_dexponent = amplitude * ratio ** exponent * np.log(ratio)
    jacobian = -np.column_stack([sqrt_weight * dy_damplitude, sqrt_weight * dy_dexponent])
    weighted_residual = sqrt_weight * (M - mean_strength(H, amplitude, exponent))
    df_resid = max(len(M) - 2, 1)
    sigma2 = float(weighted_residual @ weighted_residual) / df_resid
    covariance = sigma2 * np.linalg.inv(jacobian.T @ jacobian)
    half_width = stats.t.ppf(0.975, df_resid) * np.sqrt(np.diag(covariance))
    predicted = mean_strength(H, amplitude, exponent)
    residual_ss = float(np.sum(weight * (M - predicted) ** 2))
    total_ss = float(np.sum(weight * (M - np.average(M, weights=weight)) ** 2))
    return {
        "gamma": amplitude, "n": exponent,
        "gamma_ci_lo": amplitude - half_width[0], "gamma_ci_hi": amplitude + half_width[0],
        "n_ci_lo": exponent - half_width[1], "n_ci_hi": exponent + half_width[1],
        "R2": 1.0 - residual_ss / total_ss,
        "RMSE": np.sqrt(residual_ss / weight.sum()),
        "df_resid": df_resid, "n_cells": len(M), "N": int(np.sum(N)),
        "var_gamma": covariance[0, 0], "cov_gamma_n": covariance[0, 1], "var_n": covariance[1, 1],
    }


def original_fit(df):
    """Family-depth power laws: equal cell weights, and weights proportional to N."""
    grouped = family_depth_table(df)
    H = grouped["h"].to_numpy(float)
    M = grouped["mean"].to_numpy(float)
    counts = grouped["size"].to_numpy(float)
    fits = {
        "unweighted": _power_fit(H, M, np.ones_like(counts)),
        "N-weighted": _power_fit(H, M, counts),
    }
    return grouped, fits


def fitted_mean_band(h, fit):
    """Pointwise 95% interval of the fitted mean (delta method), not a prediction band."""
    ratio = config.H_REF / np.asarray(h, dtype=float)
    fitted = fit["gamma"] * ratio ** fit["n"]
    d_amplitude = ratio ** fit["n"]
    d_exponent = fitted * np.log(ratio)
    variance = (d_amplitude ** 2 * fit["var_gamma"]
                + 2.0 * d_amplitude * d_exponent * fit["cov_gamma_n"]
                + d_exponent ** 2 * fit["var_n"])
    half_width = stats.t.ppf(0.975, fit["df_resid"]) * np.sqrt(np.maximum(variance, 0.0))
    return fitted, fitted - half_width, fitted + half_width


def scaled_amplitude(H, M, factor):
    """One-amplitude ordinary least squares, y = alpha * factor(h), as in the legacy script."""
    factor = np.asarray(factor, float)
    return float(np.dot(factor, M) / np.dot(factor, factor))


def within_product(df):
    """Retained only as a recorded sensitivity; it is not the estimate drawn in Fig. 8.

    ln f_m = a_product - n ln(h/600). Products tested at a single depth do not inform n.
    """
    multi = df.groupby("product", observed=True)["h"].nunique()
    d = df[df["product"].isin(multi[multi > 1].index)].copy()
    d["product"] = d["product"].astype(str)
    rows = []
    for label, formula in [("mean, product effects", "ln_fm ~ ln_h600 + C(product)"),
                           ("mean, product effects + ln E", "ln_fm ~ ln_h600 + ln_E + C(product)")]:
        model = smf.ols(formula, d).fit(cov_type="HC3")
        lo, hi = model.conf_int().loc["ln_h600"]
        rows.append({"estimator": label, "n": -model.params["ln_h600"], "ci_lo": -hi, "ci_hi": -lo,
                     "p": model.pvalues["ln_h600"], "N": len(d), "products": d["product"].nunique()})
    rng = np.random.default_rng(config.SEED)
    groups = [np.where(d["product"].to_numpy() == p)[0] for p in sorted(d["product"].unique())]
    for tau in [0.05, 0.10]:
        formula = "ln_fm ~ ln_h600 + C(product)"
        quantile_fit = smf.quantreg(formula, d).fit(q=tau, max_iter=5000)
        boots = []
        for _ in range(499):
            idx = np.concatenate([rng.choice(gi, gi.size) for gi in groups])
            boots.append(-smf.quantreg(formula, d.iloc[idx]).fit(q=tau, max_iter=5000).params["ln_h600"])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"estimator": f"{int(tau * 100)} % quantile, product effects",
                     "n": -quantile_fit.params["ln_h600"], "ci_lo": lo, "ci_hi": hi,
                     "p": np.nan, "N": len(d), "products": d["product"].nunique()})
    return pd.DataFrame(rows), d


def code_vs_simulation(exponent):
    """Dimensionless k_h at selected depths, including the fitted family-mean exponent."""
    depths = np.array([100, 150, 200, 300, 450, 600])
    return pd.DataFrame({
        "h": depths,
        "kh_EC5_capped": kh_ec5(depths),
        "kh_EC5_uncapped": kh_ec5(depths, cap=False),
        "kh_Vida_GL24h": (config.H_REF / depths) ** (1.0 / config.VIDA_M3["GL24h"]),
        "kh_EC5_2025": (config.H_REF / depths) ** config.KH_EXPONENT_2025,
        "kh_fitted": (config.H_REF / depths) ** exponent,
    })


def fig_size(df, grouped, fits):
    """Single-column panel of family-depth means in MPa."""
    # Slightly narrower than the 84 mm column. The manuscript scales the file
    # to \columnwidth. A 68 mm canvas made the 8 pt labels print too large.
    width_mm = 78
    fig, ax = gp.plt.subplots(figsize=(width_mm * gp.MM, width_mm * gp.MM * 1.05),
                              constrained_layout=True)
    _panel_means(ax, df, grouped, fits["N-weighted"])
    return gp.save(fig, "fig_size_effect")


def _panel_means(ax, df, grouped, fit):
    """Classwise means +/- 1 SD, the weighted fit, and the two scaled code shapes."""
    h_grid = np.linspace(df["h"].min(), df["h"].max(), 400)
    fitted, low, high = fitted_mean_band(h_grid, fit)
    band = ax.fill_between(h_grid, low, high, color=BAND_RGB, alpha=0.28, linewidth=0,
                           label="95\\% CI of fitted mean", zorder=1)
    family_handles = []
    for family in config.FAMILIES:
        cells = grouped[grouped["family"] == family].sort_values("h")
        if cells.empty:
            continue
        style = gp.FAMILY_STYLE[family]
        handle = ax.errorbar(
            cells["h"], cells["mean"], yerr=cells["std"],
            fmt=style["marker"], color=style["color"], linestyle="none",
            elinewidth=0.6, capsize=2.0, capthick=0.6, markersize=4.0,
            markeredgecolor=gp.INK, markeredgewidth=0.3, alpha=0.9,
            label=family.replace("*", r"$^{*}$"), zorder=3)
        family_handles.append(handle)
        # A short connector shows which depths belong to one family; it is not a fit.
        if len(cells) > 1:
            ax.plot(cells["h"], cells["mean"], color=style["color"], lw=0.55, alpha=0.8, zorder=2)
    # "Weighted fit" is the short name used in the manuscript. The curve is still
    # the power law fitted by weighted nonlinear least squares.
    power_line, = ax.plot(h_grid, fitted, color=gp.INK, lw=1.15,
                          label=rf"Weighted fit, $n={fit['n']:.2f}$ "
                                rf"[{fit['n_ci_lo']:.2f}, {fit['n_ci_hi']:.2f}]", zorder=4)
    depths = grouped["h"].to_numpy(float)
    means = grouped["mean"].to_numpy(float)
    ec5_line, = ax.plot(
        h_grid, scaled_amplitude(depths, means, kh_ec5(depths)) * kh_ec5(h_grid),
        color=gp.INK_2, lw=0.9, ls="--", label="EN 1995-1-1:2004, scaled", zorder=2)
    vida_factor = (config.H_REF / depths) ** (1.0 / config.VIDA_M3["GL24h"])
    vida_line, = ax.plot(
        h_grid, scaled_amplitude(depths, means, vida_factor)
        * (config.H_REF / h_grid) ** (1.0 / config.VIDA_M3["GL24h"]),
        color=gp.INK_2, lw=0.9, ls=":", label="Vida et al.\\ (2023), scaled", zorder=2)
    ax.set_xlabel(gp.LABEL["h"])
    ax.set_ylabel(r"Mean bending strength $\bar f_{\mathrm{m}}$ [MPa]")
    # Family markers sit in the empty upper-right corner. The curve labels stay
    # above the axes: at this size they are too long for that corner.
    # 7 pt on the 68 mm canvas prints slightly smaller than the 8 pt axis labels.
    family_legend = ax.legend(handles=family_handles, loc="upper right", fontsize=7,
                              handlelength=1.4, borderaxespad=0.3)
    ax.add_artist(family_legend)
    ax.legend(handles=[power_line, band, ec5_line, vida_line],
              loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, fontsize=7,
              handlelength=2.0, borderaxespad=0.2)
    ax.margins(x=0.04, y=0.08)


def main():
    gp.use_latex_style()
    df = data.load()
    grouped, fits = original_fit(df)
    within, _subset = within_product(df)
    comparison = code_vs_simulation(fits["N-weighted"]["n"])
    gt.save_csv(pd.DataFrame([{"fit": name, **values} for name, values in fits.items()]),
                "T9_size_original_fit")
    gt.save_csv(grouped, "T9_family_depth_summary")
    gt.save_csv(within, "T9_size_within_product")
    gt.save_csv(comparison, "T9_kh_comparison")
    h_grid = np.linspace(df["h"].min(), df["h"].max(), 400)
    fitted, low, high = fitted_mean_band(h_grid, fits["N-weighted"])
    gt.save_csv(pd.DataFrame({"h": h_grid, "fitted_mean": fitted, "ci_lo": low, "ci_hi": high}),
                "T9_family_fitted_curve")
    fig_size(df, grouped, fits)
    print(f"Cleaned sample: {len(df)} specimens; {len(grouped)} family-depth cells")
    print(grouped.round(3).to_string(index=False))
    summary = [{"fit": name, "n": values["n"], "lo": values["n_ci_lo"], "hi": values["n_ci_hi"],
                "gamma": values["gamma"], "R2": values["R2"], "RMSE": values["RMSE"],
                "df": values["df_resid"]} for name, values in fits.items()]
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(comparison.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
