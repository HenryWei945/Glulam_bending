"""Write every number quoted in the manuscript as a LaTeX key-value macro (results/tex/numbers.tex).

Usage in the manuscript:  \\input{numbers}  ...  \\V{fm-cov-P2}
An undefined key prints ??key?? in bold so that it cannot go unnoticed.
"""

import numpy as np
import pandas as pd

from glulam import config

R = config.RESULTS
vals = {}


def put(key, value, fmt="{:.2f}"):
    if isinstance(value, str):
        vals[key] = value
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        vals[key] = "--"
    else:
        txt = fmt.format(value)
        if txt.startswith("-") and float(txt.replace(",", "")) == 0:
            txt = txt[1:]                                   # no negative zero
        vals[key] = txt.replace("-", "\\ensuremath{-}")   # proper minus sign in text and math


def p_tex(p):
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return f"{p:.3f}"


def read(name):
    return pd.read_csv(R / f"{name}.csv")


def main():
    ov = read("T1_product_overview")
    put("N-spec", int(ov["N"].sum()), "{:d}")
    put("N-records", int(ov["N"].sum()), "{:d}")
    for _, r in ov.iterrows():
        put(f"N-{r['Product']}", r["N"], "{:d}")

    for name, key in [("T2_descriptive_product", "product"), ("T2_descriptive_family", "family")]:
        d = read(name)
        for _, r in d.iterrows():
            k = r[key].replace("*", "")
            put(f"N-{k}", r["N"], "{:d}")
            put(f"{r['response']}-mean-{k}", r["mean"], "{:.1f}" if r["response"] == "fm" else "{:.2f}")
            put(f"{r['response']}-sd-{k}", r["sd"], "{:.1f}" if r["response"] == "fm" else "{:.2f}")
            put(f"{r['response']}-cov-{k}", r["cov"])
            put(f"{r['response']}-q05-{k}", r["q05"], "{:.1f}")
            put(f"{r['response']}-skew-{k}", r["skew"])
        if key == "product":
            fm = d[d["response"] == "fm"]
            put("fm-cov-min", fm["cov"].min())
            put("fm-cov-max", fm["cov"].max())
            put("fm-mean-min", fm["mean"].min(), "{:.1f}")
            put("fm-mean-max", fm["mean"].max(), "{:.1f}")

    ow = read("T4_oneway")
    for _, r in ow.iterrows():
        k = f"{r['factor'][:4]}-{r['response']}"
        put(f"anova-{k}-F", r["F"])
        put(f"anova-{k}-df1", r["df1"], "{:.0f}")
        put(f"anova-{k}-df2", r["df2"], "{:.0f}")
        put(f"anova-{k}-p", p_tex(r["p"]))
        put(f"anova-{k}-eta2", r["eta2"], "{:.3f}")
        put(f"welch-{k}-F", r["Welch_F"])
        put(f"welch-{k}-df2", r["Welch_df2"], "{:.0f}")
        put(f"welch-{k}-p", p_tex(r["Welch_p"]))
        put(f"kw-{k}-p", p_tex(r["KW_p"]))
        put(f"bf-{k}-p", p_tex(r["BF_p"]))
    adj = read("T4_adjusted_anova").set_index("term")
    put("adj-family-F", adj.loc["C(family)", "F"])
    put("adj-family-p", adj.loc["C(family)", "PR(>F)"], "{:.3f}")
    vc = pd.read_csv(R / "T4_variance_components.csv", keep_default_na=False).set_index("model")
    put("icc-null", 100 * vc.loc["null", "ICC"], "{:.0f}")
    put("icc-cond", 100 * vc.loc["ln E + ln h", "ICC"], "{:.0f}")
    nt = read("T4_nested_tests")
    r = nt[(nt["base"] == "E, h, outer class") & (nt["added"] == "family")].iloc[0]
    put("lr-family-p", p_tex(r["p"]))
    put("lr-family-dR2", r["dR2"], "{:.3f}")
    lt = read("T4_lower_tail_family")
    for _, r in lt[lt["model"] == "family only"].iterrows():
        put(f"q05reg-{r['term'].replace('*', '')}-ratio", r["ratio"])
        put(f"q05reg-{r['term'].replace('*', '')}-p", r["p"], "{:.3f}")

    gof = read("T3_gof_family")
    g = gof[(gof["family"] == "GL24*") & (gof["response"] == "fm")].set_index("dist")
    put("gl24-ln-ksp", g.loc["Lognormal", "KS_p"], "{:.3f}")
    put("gl24-ln-cvmp", g.loc["Lognormal", "CvM_p"], "{:.3f}")
    put("gl24-ln-adp", g.loc["Lognormal", "AD_p"], "{:.3f}")
    put("gl24-ln-dAICgamma", g.loc["Gamma", "AIC"] - g.loc["Lognormal", "AIC"], "{:.0f}")
    put("gl24-ln-dAICnormal", g.loc["Normal", "AIC"] - g.loc["Lognormal", "AIC"], "{:.0f}")
    put("gl24-dAIC-ln-gam", g.loc["Lognormal", "AIC"] - g.loc["Gamma", "AIC"], "{:.1f}")
    for d, k in [("Gamma", "gam"), ("Lognormal", "ln")]:
        put(f"gl24-{k}-ksp2", g.loc[d, "KS_p"], "{:.3f}")
        put(f"gl24-{k}-cvmp2", g.loc[d, "CvM_p"], "{:.3f}")
        put(f"gl24-{k}-adp2", g.loc[d, "AD_p"], "{:.3f}")
        put(f"gl24-{k}-q05fit", g.loc[d, "q05_fit"], "{:.1f}")
    gE = gof[(gof["family"] == "GL24*") & (gof["response"] == "E")].set_index("dist")
    put("gl24-E-gamma-cvmp", gE.loc["Gamma", "CvM_p"], "{:.3f}")
    gp_ = read("T3_gof_product")
    gp_ = gp_[(gp_["response"] == "fm") & (gp_["dist"] == "Normal")]
    put("prod-normal-rejected", int((gp_["AD_p"] < 0.05).sum()), "{:d}")
    mix = read("T3_gl24_mixture_q05").iloc[0]
    put("q05-emp-gl24", mix["Q05_empirical"], "{:.1f}")
    put("q05-ln-gl24", mix["Q05_pooled_lognormal"], "{:.1f}")
    put("q05-mix-gl24", mix["Q05_product_mixture"], "{:.1f}")
    put("sigln-pooled-gl24", mix["sigma_ln_pooled"], "{:.3f}")
    put("sigln-within-gl24", mix["sigma_ln_within_mean"], "{:.3f}")

    ch = read("T8_characteristic_product")
    for _, r in ch.iterrows():
        p = r["product"]
        put(f"fk-{p}", r["fk_raw"], "{:.1f}")
        put(f"fk600-{p}", r["fk_600"], "{:.1f}")
        put(f"ratio-{p}", r["fk_ref_over_declared"])
        put(f"Eratio-{p}", r["E_ratio"])
        put(f"Eenratio-{p}", r["E0_en384_ratio"])
    put("ratio-min-others", ch.loc[ch["product"] != "P7", "fk_ref_over_declared"].min())

    ss = read("T10_strength_stiffness").set_index("product")
    for p in config.PRODUCTS:
        put(f"r-{p}", ss.loc[p, "r_fm_E"])
        put(f"slope-{p}", ss.loc[p, "slope_lnfm_lnE"])
    put("r-all", ss.loc["all", "r_fm_E"])
    put("r-all-lo", ss.loc["all", "r_ci_lo"])
    put("r-all-hi", ss.loc["all", "r_ci_hi"])
    put("r-within", ss.loc["pooled within", "r_fm_E"])
    put("r-min", ss.loc[config.PRODUCTS, "r_fm_E"].min())
    put("r-max", ss.loc[config.PRODUCTS, "r_fm_E"].max())
    put("sigres-min", ss.loc[config.PRODUCTS, "sigma_ln_resid"].min(), "{:.3f}")
    put("sigres-max", ss.loc[config.PRODUCTS, "sigma_ln_resid"].max(), "{:.3f}")

    mc = read("T5_model_comparison").set_index("model")
    for m in mc.index:
        put(f"A-R2-{m}", mc.loc[m, "A_R2_mean"])
        put(f"A-R2sd-{m}", mc.loc[m, "A_R2_sd"])
        put(f"A-RMSE-{m}", mc.loc[m, "A_RMSE_mean"])
        put(f"A-RMSEsd-{m}", mc.loc[m, "A_RMSE_sd"])
        put(f"A-RMSEsdrep-{m}", mc.loc[m, "A_RMSE_sd_across_repeats"], "{:.3f}")
        put(f"B-R2-{m}", mc.loc[m, "B_R2_mean"])
        put(f"B-RMSE-{m}", mc.loc[m, "B_RMSE_mean"])
        put(f"C-R2-{m}", mc.loc[m, "C_R2"])
        put(f"C-RMSE-{m}", mc.loc[m, "C_RMSE"])
        put(f"C-n-{m}", mc.loc[m, "C_products_predictable"], "{:.0f}")
        put(f"terms-{m}", mc.loc[m, "n_terms"], "{:.0f}")
    vif = read("T6_M7_vif")
    put("vif-max-M7", vif["VIF"].max(), "{:,.0f}")
    put("vif-n10-M7", int((vif["VIF"] > 10).sum()), "{:d}")
    co = read("T6_S1_coefficients").set_index("term")
    put("s1-lnE", co.loc["ln_E", "estimate"])
    put("s1-lnE-se", co.loc["ln_E", "se"])
    put("s1-lnh", co.loc["ln_h600", "estimate"])
    put("s1-lnh-se", co.loc["ln_h600", "se"])
    put("s1-C40", co.loc["outer class C40", "estimate"])
    put("s1-C40-pct", 100 * (np.exp(co.loc["outer class C40", "estimate"]) - 1), "{:.0f}")
    put("s1-vif-max", co["VIF"].max(), "{:.1f}")

    ly = read("T7_layup_P2_P7")
    a = ly[(ly["subset"] == "matched b = 48 mm") & (ly["variable"] == "fm")].iloc[0]
    e = ly[(ly["subset"] == "matched b = 48 mm") & (ly["variable"] == "E")].iloc[0]
    put("lay-N-P2", a["N_P2"], "{:.0f}")
    put("lay-fm-P2", a["mean_P2"], "{:.1f}")
    put("lay-fm-P7", a["mean_P7"], "{:.1f}")
    put("lay-fm-diff", a["diff"], "{:.1f}")
    put("lay-fm-cilo", a["ci_lo"], "{:.1f}")
    put("lay-fm-cihi", a["ci_hi"], "{:.1f}")
    put("lay-fm-pmw", a["p_mw"], "{:.2f}")
    put("lay-q05-P2", a["q05_P2"], "{:.1f}")
    put("lay-q05-P7", a["q05_P7"], "{:.1f}")
    put("lay-q05-ci", "[{:.1f}, {:.1f}]".format(*[float(v) for v in str(a["q05_diff_ci"]).strip("()").split(",")])
        .replace("-", "\\ensuremath{-}"))
    put("lay-cov-P2", a["cov_P2"])
    put("lay-cov-P7", a["cov_P7"])
    put("lay-E-P2", e["mean_P2"] / 1e3)
    put("lay-E-P7", e["mean_P7"] / 1e3)
    put("lay-E-pct", 100 * (e["ratio"] - 1), "{:.0f}")
    aa = ly[(ly["subset"] == "all") & (ly["variable"] == "fm")].iloc[0]
    ea = ly[(ly["subset"] == "all") & (ly["variable"] == "E")].iloc[0]
    put("layall-fm-diff", aa["diff"], "{:.1f}")
    put("layall-fm-p", aa["p_welch"], "{:.2f}")
    put("layall-E-pct", 100 * (ea["ratio"] - 1), "{:.0f}")
    an = read("T7_layup_P2_P7_ancova").iloc[0]
    put("lay-shift", -100 * an["shift_P7_vs_P2_at_equal_E"], "{:.0f}")
    put("lay-shift-lo", -100 * an["ci_hi"], "{:.0f}")
    put("lay-shift-hi", -100 * an["ci_lo"], "{:.0f}")
    put("lay-rel", 100 * an["shift_P7_vs_P2_at_equal_E"], "{:.0f}")      # P7 relative to P2 at equal E [%]
    put("lay-rel-lo", 100 * an["ci_lo"], "{:.0f}")
    put("lay-rel-hi", 100 * an["ci_hi"], "{:.0f}")
    put("lay-fm-pct", 100 * a["diff"] / a["mean_P2"], "{:.0f}")
    put("lay-q05-diff", a["q05_P7"] - a["q05_P2"], "{:.1f}")
    put("lay-slope-P2", an["slope_P2"])
    put("lay-slope-P7", an["slope_P7"])

    sz = read("T9_size_within_product").set_index("estimator")
    for key, est in [("mean", "mean, product effects"), ("meanE", "mean, product effects + ln E"),
                     ("q05", "5 % quantile, product effects"), ("q10", "10 % quantile, product effects")]:
        put(f"n-{key}", sz.loc[est, "n"])
        put(f"n-{key}-lo", sz.loc[est, "ci_lo"])
        put(f"n-{key}-hi", sz.loc[est, "ci_hi"])
    put("n-products", sz["products"].iloc[0], "{:.0f}")
    put("n-N", sz["N"].iloc[0], "{:.0f}")
    so = read("T9_size_original_fit").set_index("fit")
    put("n-orig-unw", so.loc["unweighted", "n"])
    put("n-orig-w", so.loc["N-weighted", "n"])
    kh = read("T9_kh_comparison").set_index("h")
    for h in [150, 200, 300, 450]:
        put(f"kh-ec5-{h}", kh.loc[h, "kh_EC5_capped"], "{:.3f}")
        put(f"kh-vida-{h}", kh.loc[h, "kh_Vida_GL24h"], "{:.3f}")
        put(f"kh-2025-{h}", kh.loc[h, "kh_EC5_2025"], "{:.3f}")

    me = read("T11_model_error").set_index("quantity")["value"]
    put("sig-post", me["posterior mean sigma (in-sample residuals)"], "{:.3f}")
    put("sig-hdi-lo", me["95% HDI lower"], "{:.3f}")
    put("sig-hdi-hi", me["95% HDI upper"], "{:.3f}")
    put("sig-cv", me["MLE sigma, cross-fitted (out-of-fold eta)"], "{:.3f}")
    put("sig-s1", me["residual sd, S1 log model"], "{:.3f}")
    km = read("T12_kM_replication").iloc[0]
    put("kM50", km["kM_50"], "{:.3f}")
    put("kM95", km["kM_95"], "{:.3f}")
    put("gammaM-repl", km["gamma_M"])

    br = read("T13_bayes_reliability").set_index("product")
    for p in config.PRODUCTS:
        put(f"greq-{p}", br.loc[p, "gamma_req_med"])
        put(f"greq-lo-{p}", br.loc[p, "gamma_req_lo"])
        put(f"greq-hi-{p}", br.loc[p, "gamma_req_hi"])
        put(f"gvar-{p}", br.loc[p, "gamma_var_med"])
        put(f"f05b-{p}", br.loc[p, "f05_med"], "{:.1f}")
        put(f"pbelow-{p}", br.loc[p, "P_f05_below_declared"])
        put(f"gannexD-{p}", br.loc[p, "gamma_req_annexD"])
        put(f"sigb-{p}", br.loc[p, "sigma_ln_med"], "{:.3f}")
    put("gvar-min", br["gamma_var_med"].min())
    put("gvar-max", br["gamma_var_med"].max())
    put("greq-min", br["gamma_req_med"].min())
    put("greq-max", br["gamma_req_med"].max())
    nb = read("T13_bayes_new_product").set_index("quantity")
    put("new-gvar", nb.loc["new product: gamma_var", "med"])
    put("new-gvar-lo", nb.loc["new product: gamma_var", "lo"])
    put("new-gvar-hi", nb.loc["new product: gamma_var", "hi"])
    put("new-sig", nb.loc["new product: sigma_ln", "med"], "{:.3f}")
    put("new-f05", nb.loc["new product: f05 [MPa]", "med"], "{:.1f}")
    put("new-f05-lo", nb.loc["new product: f05 [MPa]", "lo"], "{:.1f}")
    put("new-f05-hi", nb.loc["new product: f05 [MPa]", "hi"], "{:.1f}")
    put("mix-ratio", nb.loc["random product mixture: f05/X_d", "med"])
    bs = read("T13_bayes_summary").set_index("index")
    put("tau-post", bs.loc["tau", "mean"], "{:.3f}")
    put("ms-post", np.exp(bs.loc["m_s", "mean"]), "{:.3f}")

    lines = ["% Automatically generated by Code/python/export_numbers.py -- do not edit by hand",
             "\\makeatletter",
             "\\def\\V#1{\\ifcsname gv@#1\\endcsname\\csname gv@#1\\endcsname\\else\\textbf{??#1??}\\fi}"]
    lines += [f"\\expandafter\\def\\csname gv@{k}\\endcsname{{{v}}}" for k, v in sorted(vals.items())]
    lines.append("\\makeatother")
    (config.TEX / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(vals)} values written to {config.TEX / 'numbers.tex'}")


if __name__ == "__main__":
    main()
