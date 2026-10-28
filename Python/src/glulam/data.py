"""Loading and preparation of the four-point bending database."""

import numpy as np
import pandas as pd

from . import config

RAW_COLUMNS = {
    "l [mm]": "l", "a [mm]": "a", "c [mm]": "c", "hn [mm]": "h", "bn [mm]": "b",
    "fm [MPa]": "fm", "MoE [MPa]": "E", "class": "family_code", "product": "product_name",
    "h_ext_mm": "hext", "h_int_mm": "hint", "class_ext": "cext", "class_int": "cint",
    "fing_joint_ext": "fjext", "fing_joint_int": "fjint", "lam_glue": "glue",
}
# Optional columns that are used automatically when present in the database
OPTIONAL_COLUMNS = {
    "rho [kg/m3]": "rho", "rhokgm3": "rho", "density [kg/m3]": "rho", "rho": "rho",
    "MC [%]": "mc", "MC": "mc", "u [%]": "mc",
}


def kh_ec5(h):
    """EN 1995-1-1 depth factor for rectangular glulam."""
    h = np.asarray(h, dtype=float)
    return np.where(h < config.H_REF,
                    np.minimum((config.H_REF / h) ** config.KH_EXPONENT, config.KH_CAP), 1.0)


def load(path=None, drop_duplicates=None, exclude_cluster=None):
    """Return the manuscript analysis sample (1054 specimens).

    `data/glulam_cleaned.csv` already uses anonymised product IDs and the derived
    columns. A raw export with the original column `fm [MPa]` still goes through
    the renaming below; row filters stay off unless the caller sets them.
    """
    drop_duplicates = config.DROP_DUPLICATES if drop_duplicates is None else drop_duplicates
    df = pd.read_csv(path or config.DATA_FILE, encoding="utf-8-sig")
    if "fm" in df.columns and "product" in df.columns and "fm [MPa]" not in df.columns:
        df["product"] = pd.Categorical(df["product"], categories=config.PRODUCTS, ordered=True)
        df["family"] = pd.Categorical(df["family"], categories=config.FAMILIES, ordered=True)
        return df
    df.insert(0, "row", np.arange(len(df)))
    rename = {**RAW_COLUMNS, **{k: v for k, v in OPTIONAL_COLUMNS.items() if k in df.columns}}
    df = df.rename(columns=rename)
    missing = [v for v in RAW_COLUMNS.values() if v not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in database: {missing}")

    if not config.PRODUCT_ID:
        raise FileNotFoundError(f"Product mapping not found: {config.PRODUCT_MAP_FILE} (see data/README.md)")
    unknown = set(df["product_name"]) - set(config.PRODUCT_ID)
    if unknown:
        raise KeyError(f"Products without anonymised ID: {unknown}")
    df["product"] = pd.Categorical(df["product_name"].map(config.PRODUCT_ID),
                                   categories=config.PRODUCTS, ordered=True)
    df["family"] = pd.Categorical(df["family_code"].map(config.FAMILY_FROM_CODE),
                                  categories=config.FAMILIES, ordered=True)
    df = df.drop(columns=["product_name"])  # keep the commercial names out of every output

    for col in ["cext", "cint", "fjext", "fjint", "glue"]:
        df[col] = df[col].astype(str).str.strip()
    # Keep class_int as recorded. P2 and P7 are coded C24 in the database.

    if drop_duplicates:   # same product and identical bending strength = same specimen entered twice
        df = df[~df.duplicated(subset=["product", "fm"], keep="first")]
    df = df.reset_index(drop=True)

    df["kh"] = kh_ec5(df["h"])
    df["fm600"] = df["fm"] / df["kh"]          # bending strength adjusted to h_ref = 600 mm
    df["ln_fm"] = np.log(df["fm"])
    df["ln_E"] = np.log(df["E"])
    df["ln_h600"] = np.log(df["h"] / config.H_REF)
    df["l_over_h"] = df["l"] / df["h"]
    # engineered predictors used by the original stepwise models (M7-M8)
    df["logE"] = np.log(df["E"])
    df["bh"] = df["b"] / df["h"]
    df["hb"] = df["h"] / df["b"]
    df["Eh"] = df["E"] * df["h"]
    df["Eb"] = df["E"] * df["b"]
    df["EZ"] = df["E"] * df["h"] ** 3 / (12 * df["b"])     # stiffness proxy used in M8
    df["EZ7"] = df["E"] * df["h"] ** 2 / df["b"]           # stiffness proxy used in M7
    df["h2b"] = df["h"] ** 2 / df["b"]
    df["Eoh"] = df["E"] / df["h"]
    df["Eh2"] = df["E"] * df["h"] ** 2
    # Lamella thickness relative to beam depth. Used by manuscript M6 and M7.
    df["hext_frac"] = df["hext"] / df["h"]
    df["hint_frac"] = df["hint"] / df["h"]
    return df


def has_density(df):
    return "rho" in df.columns and df["rho"].notna().any()
