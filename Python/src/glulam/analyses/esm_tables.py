"""LaTeX tables for Online Resource 1 (supplementary material), written to results/tex/esm_*.tex."""

import numpy as np
import pandas as pd

from glulam import config

R = config.RESULTS
T = config.TEX


def tab(df, name, caption, colfmt=None, size="\\footnotesize"):
    body = df.to_latex(index=False, escape=False, column_format=colfmt, na_rep="--")
    txt = "\n".join(["\\begin{table}[htbp]", "\\centering", size, f"\\caption{{{caption}}}", body.strip(),
                     "\\end{table}", ""])
    (T / f"esm_{name}.tex").write_text(txt)


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def main():
    g = pd.read_csv(R / "T3_gof_family.csv")
    g = g.assign(response=g["response"].map({"fm": "$f_{\\mathrm{m}}$", "E": "$E_{\\mathrm{m,g}}$"}))
    family = {"GL24*": r"\GL{24}", "GL27*": r"\GL{27}", "GL28*": r"\GL{28}", "GL33*": r"\GL{33}"}
    t = pd.DataFrame({"Family": g["family"].map(family), "Resp.": g["response"], "Distribution": g["dist"],
                      "KS": g["KS"].map(f), "$p$": g["KS_p"].map(f), "CvM": g["CvM"].map(f),
                      "$p$ ": g["CvM_p"].map(f), "AD": g["AD"].map(f), "$p$  ": g["AD_p"].map(f),
                      "$\\Delta$AIC": g["dAIC"].map(lambda v: f(v, 1))})
    tab(t, "gof_family", "Goodness of fit by family. Maximum-likelihood fits; $p$-values from a parametric bootstrap "
        f"with re-estimated parameters ($B={config.N_BOOT}$); $\\Delta$AIC relative to the best distribution. "
        "$E_{\\mathrm{m,g}}$ in GPa.", size="\\scriptsize")

    g = pd.read_csv(R / "T3_gof_product.csv")
    g = g[g["response"] == "fm"]
    t = pd.DataFrame({"Product": g["product"], "Distribution": g["dist"], "KS $p$": g["KS_p"].map(f),
                      "CvM $p$": g["CvM_p"].map(f), "AD $p$": g["AD_p"].map(f),
                      "$\\Delta$AIC": g["dAIC"].map(lambda v: f(v, 1)), "$Q_{0.05}$ fit": g["q05_fit"].map(lambda v: f(v, 1)),
                      "$Q_{0.05}$ emp.": g["q05_emp"].map(lambda v: f(v, 1))})
    tab(t, "gof_product", "Goodness of fit of the bending strength by product (bootstrap $p$-values).",
        size="\\scriptsize")

    o = pd.read_csv(R / "T4_oneway.csv")
    t = pd.DataFrame({"Factor": o["factor"], "Response": o["response"].map(
        {"fm": "$f_{\\mathrm{m}}$", "fm600": "$f_{\\mathrm{m}}/k_{h}$", "E": "$E_{\\mathrm{m,g}}$"}),
        "$F$ (df)": [f"{a:.2f} ({b:.0f}, {c:.0f})" for a, b, c in zip(o["F"], o["df1"], o["df2"])],
        "$p$": o["p"].map(lambda v: f"{v:.1e}"), "$\\eta^2$": o["eta2"].map(f),
        "Welch $F$ (df$_2$)": [f"{a:.2f} ({b:.0f})" for a, b in zip(o["Welch_F"], o["Welch_df2"])],
        "Welch $p$": o["Welch_p"].map(lambda v: f"{v:.1e}"), "KW $p$": o["KW_p"].map(lambda v: f"{v:.1e}"),
        "BF $p$": o["BF_p"].map(lambda v: f"{v:.1e}")})
    tab(t, "oneway", "One-way comparisons: classical ANOVA, Welch ANOVA, Kruskal--Wallis (KW) and Brown--Forsythe "
        "(BF) tests.", size="\\scriptsize")

    a = pd.read_csv(R / "T4_adjusted_anova.csv")
    a["term"] = a["term"].replace({"C(family)": "family", "b": "$b$", "h": "$h$", "l": "$l$"})
    t = pd.DataFrame({"Term": a["term"], "SS": a["sum_sq"].map(lambda v: f(v, 1)), "df": a["df"].map(lambda v: f(v, 0)),
                      "$F$": a["F"].map(lambda v: f(v, 2)), "$p$": a["PR(>F)"].map(f)})
    tab(t, "adjusted_anova", "Type II analysis of variance of $f_{\\mathrm{m}}$ with family, width, depth and span.")

    n = pd.read_csv(R / "T4_nested_tests.csv")
    base = {
        "E, h, outer class": r"$\ln E_{\mathrm{m,g}}$, $\ln(h/600)$, outer grade",
        "E, h, outer class, layup": r"$\ln E_{\mathrm{m,g}}$, $\ln(h/600)$, outer grade, layup",
    }
    t = pd.DataFrame({"Base model": n["base"].map(base), "Added": n["added"], "LR": n["LR"].map(lambda v: f(v, 2)),
                      "df": n["df"].map(lambda v: f(v, 0)), "$p$": n["p"].map(lambda v: "--" if pd.isna(v) else f"{v:.1e}"),
                      "$R^2$ base": n["R2_base"].map(f), "$\\Delta R^2$": n["dR2"].map(f)})
    tab(t, "nested", "Likelihood-ratio tests for adding the family label or the product identity to models of "
        "$\\ln f_{\\mathrm{m}}$. Layup adds the inner-lamella grade, $h_{\\mathrm{ext}}$, $h_{\\mathrm{int}}$ and $b$. "
        "With that set included, family and product are aliased (df = 0).",
        colfmt="p{0.30\\linewidth}llllll", size="\\scriptsize")

    v = pd.read_csv(R / "T4_variance_components.csv", keep_default_na=False)
    t = pd.DataFrame({"Fixed effects": v["model"].replace({
                          "null": "intercept only",
                          "ln E + ln h": r"$\ln E_{\mathrm{m,g}}+\ln(h/600)$"}),
                      "$\\tau^2$ (between)": v["tau2_between"].map(lambda x: f(float(x), 4)),
                      "$\\sigma^2$ (within)": v["sigma2_within"].map(lambda x: f(float(x), 4)),
                      "ICC": v["ICC"].map(lambda x: f(float(x), 3))})
    tab(t, "icc", "Random-intercept models for $\\ln f_{\\mathrm{m}}$ with product as grouping factor.")

    s = pd.read_csv(R / "T5_selected_terms.csv")
    t = pd.DataFrame({"Model": s["model"], "Terms": s["terms"].str.replace("_", "\\_", regex=False)
                      .str.replace("^2", "$^2$", regex=False)})
    tab(t, "selected_terms", "Terms of the models fitted to all data (stepwise selection for M4--M7). "
        "Variables: E modulus, h depth, b width, hext/hint lamella thicknesses, cext/cint lamella grades, "
        "fjext/fjint finger joints, glue lamination adhesive; logE, hb, bh, Eh, Eb, EZ, EZ7, Eh2, h2b, Eoh, hint\\_frac "
        "engineered variables.", colfmt="lp{0.75\\linewidth}", size="\\scriptsize")

    vf = pd.read_csv(R / "T6_M7_vif.csv")
    t = pd.DataFrame({"Column": vf["column"].str.replace("_", "\\_", regex=False), "VIF": vf["VIF"].map(lambda x: f"{x:,.0f}")})
    tab(t, "vif", "Variance inflation factors of the columns of the M7 formula selected on all data.")

    c = pd.read_csv(R / "T6_S1_coefficients.csv")
    t = pd.DataFrame({"Term": c["term"].str.replace("_", "\\_", regex=False), "Estimate": c["estimate"].map(lambda x: f(x, 4)),
                      "SE": c["se"].map(lambda x: f(x, 4)), "$t$": c["t"].map(lambda x: f(x, 2)),
                      "$p$": c["p"].map(lambda x: f"{x:.1e}"), "VIF": c["VIF"].map(lambda x: f(x, 2))})
    tab(t, "s1", "Coefficients of the parsimonious model S1 fitted to all data ($\\ln f_{\\mathrm{m}}$; reference outer grade C24).")

    me = pd.read_csv(R / "T11_model_error.csv")
    t = pd.DataFrame({"Quantity": me["quantity"].str.replace("%", "\\%", regex=False).str.replace("*", "$^*$", regex=False),
                      "Value": me["value"].map(lambda x: f(x, 4))})
    tab(t, "model_error", "Log-scale model error of the M7 formula with fixed least-squares coefficients "
        "(posterior from PyMC, 4 chains $\\times$ 2000 draws) and comparison with cross-fitted and unconditional values.")

    km = pd.read_csv(R / "T12_kM_replication.csv").iloc[0]
    rows = [("$N$ (\\GL{24})", f"{km['N']:.0f}"), ("mean $k_h$", f(km["kh_mean"])), ("mean $f_d(\\beta)$ [MPa]", f(km["fd_mean"], 2)),
            ("mean $f_{\\mathrm{m,d}}$ EC5 [MPa]", f(km["fmd_ec5_mean"], 2)), ("median $k_M^*$", f(km["kM_50"])),
            ("95\\,\\% fractile $k_M^*$", f(km["kM_95"])), ("$\\gamma_M=1.25\\,k_{M,95}$", f(km["gamma_M"]))]
    tab(pd.DataFrame(rows, columns=["Quantity", "Value"]), "kM",
        "Conditional calibration of $k_M$ for \\GL{24} using the M7 conditional mean as resistance mean "
        "(not used in the paper, see Section 3.8).")

    pp = pd.read_csv(R / "T12_gamma_population_product.csv")
    t = pd.DataFrame({"Product": pp["product"], "$N$": pp["N"], "$\\sigma_{\\ln}$": pp["sigma_ln"].map(f),
                      "$X_d$": pp["X_d"].map(lambda x: f(x, 1)), "$f_{\\mathrm{k}}$ own": pp["fk_own"].map(lambda x: f(x, 1)),
                      "$\\gamma_{M,\\mathrm{req}}$ (declared)": pp["gamma_req_declared"].map(lambda x: f(x, 2)),
                      "$\\gamma_{M}$ (own $f_{\\mathrm{k}}$)": pp["gamma_req_own"].map(lambda x: f(x, 2))})
    tab(t, "gamma_freq", "Frequentist population-based design values (strengths divided by $k_h$, $\\alpha_R\\beta=3.04$).")
    print("ESM tables written:", len(list(T.glob("esm_*.tex"))))


if __name__ == "__main__":
    main()
