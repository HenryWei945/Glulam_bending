"""Characteristic bending strength per product and family, adjusted to h = 600 mm, versus declared values."""

import numpy as np
import pandas as pd

from glulam import config, data, stats as gst, tables as gt


def characteristic(df, group):
    rows = []
    for key, g in df.groupby(group, observed=True):
        fam = g["family"].iloc[0] if group == "product" else key
        fk_decl = config.FMK_DECLARED[fam]
        r = {group: key, "family": fam, "N": len(g), "fmk_declared": fk_decl,
             "mean_fm": g["fm"].mean(), "cov_fm": g["fm"].std(ddof=1) / g["fm"].mean(),
             "sigma_ln": np.log(g["fm"]).std(ddof=1), "kh_mean": g["kh"].mean(),
             "q05_raw": gst.q05_order(g["fm"]), "q05_600": gst.q05_order(g["fm600"]),
             "f05_ln_raw": gst.f05_lognormal_point(g["fm"]), "f05_ln_600": gst.f05_lognormal_point(g["fm600"]),
             "ks75": gst.ks_factor(len(g)),
             "fk_raw": gst.fk_lognormal(g["fm"]), "fk_600": gst.fk_lognormal(g["fm600"])}
        r["fk_600_over_declared"] = r["fk_600"] / fk_decl
        r["fk_raw_over_declared"] = r["fk_raw"] / fk_decl
        if group == "product":
            # Every product is compared as tested. fk_600 is a uniform EN 1995-1-1 reference
            # and does not depend on an approval-specific depth-factor rule.
            r["fk_ref"] = r["fk_raw"]
            r["fk_ref_over_declared"] = r["fk_ref"] / fk_decl
            e_decl = config.E_DECLARED[key]
            r.update({"E_mean": g["E"].mean(), "E_declared": e_decl, "E_ratio": g["E"].mean() / e_decl,
                      # sensitivity only: EN 384 shear correction from global to pure-bending MOE (solid timber)
                      "E0_en384": 1.3 * g["E"].mean() - 2690,
                      "E0_en384_ratio": (1.3 * g["E"].mean() - 2690) / e_decl})
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    df = data.load()
    prod = characteristic(df, "product")
    fam = characteristic(df, "family")
    gt.save_csv(prod, "T8_characteristic_product")
    gt.save_csv(fam, "T8_characteristic_family")
    t = prod[["product", "family", "N", "fmk_declared", "mean_fm", "cov_fm", "q05_raw", "fk_raw",
              "fk_600", "fk_ref_over_declared", "E_mean", "E_declared"]].copy()
    t["E_mean"] = t["E_mean"] / 1e3
    t["E_declared"] = t["E_declared"] / 1e3
    t = gt.fmt(t, {"fmk_declared": "{:.0f}", "mean_fm": "{:.1f}", "cov_fm": "{:.2f}", "q05_raw": "{:.1f}",
                   "fk_raw": "{:.1f}", "fk_600": "{:.1f}", "fk_ref_over_declared": "{:.2f}",
                   "E_mean": "{:.2f}", "E_declared": "{:.1f}"})
    t.columns = ["Product", "Family", "$N$", "$f_{\\mathrm{m,k,decl}}$", "$\\bar f_{\\mathrm{m}}$",
                 "COV", "$Q_{0.05}$", "$f_{\\mathrm{m,k}}$", "$f_{\\mathrm{m,k,600}}$", "Ratio",
                 "$\\bar E_{\\mathrm{m,g}}$", "$E_{\\mathrm{decl}}$"]
    gt.save_tex(t.astype(str), "T8_characteristic_product",
                "Characteristic bending strength per product. $Q_{0.05}$: empirical 5\\,\\% fractile; "
                "$f_{\\mathrm{m,k}}$: lognormal 5\\,\\% fractile at 75\\,\\% confidence (sample-size factor "
                "$k_s(n)$) of the values as tested; subscript 600: the same fractile after dividing each "
                "observation by the EN 1995-1-1 depth factor, reported for every product. "
                "Ratio: $f_{\\mathrm{m,k}}$ over the declared value. Strengths in MPa, moduli in GPa.",
                "tab:fk")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(prod.round(3).to_string(index=False))
        print(fam.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
