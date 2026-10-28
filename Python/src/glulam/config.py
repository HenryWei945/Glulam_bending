"""Project-wide constants: paths, anonymisation, declared values, analysis settings."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # Python/ directory
DATA_DIR = ROOT / "data"
# Analysis sample used in the manuscript: 1054 specimens, product IDs P1-P7.
# Override with GLULAM_DATA=/path/to/file.csv if needed.
DATA_FILE = Path(os.environ.get("GLULAM_DATA", DATA_DIR / "glulam_cleaned.csv"))
PHOTO_DIR = DATA_DIR / "photos"                       # optional fig2_a.jpg ... fig2_d.jpg
OUTPUT = Path(os.environ.get("GLULAM_OUTPUT", ROOT / "outputs"))
RESULTS = OUTPUT / "results"
TEX = OUTPUT / "tex"
FIGURES = OUTPUT / "figures"

SEED = 20261011
K_FOLDS = 5
CV_REPEATS = 10          # repeated K-fold for fixed-formula models
NESTED_REPEATS = 4       # repeated K-fold with model selection inside each training fold
N_BOOT = 999             # parametric bootstrap replicates for GoF p-values

# Product column of the analysis file is already the anonymised ID used in the manuscript.
PRODUCTS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
PRODUCT_ID = {name: name for name in PRODUCTS}

FAMILIES = ["GL24*", "GL27*", "GL28*", "GL33*"]
FAMILY_FROM_CODE = {24: "GL24*", 27: "GL27*", 28: "GL28*", 33: "GL33*"}
# Declared characteristic bending strength of each approval-based family [MPa]
FMK_DECLARED = {"GL24*": 24.0, "GL27*": 27.0, "GL28*": 28.0, "GL33*": 33.0}
# Declared mean modulus of elasticity per product [MPa] (Table 1, from the TG/ETA documents)
E_DECLARED = {"P1": 12000, "P2": 11000, "P3": 9000, "P4": 11000,
              "P5": 11000, "P6": 12500, "P7": 14000}

# Approval notes below are not used by the manuscript or by the comparison rules.
# Depth factor, declared density and product-specific gamma_M are not analysis inputs.
# Internal use only: approval numbers would identify the products and must not appear in outputs.
APPROVAL = {
    "P1": dict(basis="glulam-type beam, tension side marked", kh_allowed=False, gamma_M=1.25, rho_mean=500,
               layup="T22 tension lamellae (>= h/6), C14 others; 33-45 mm"),
    "P2": dict(basis="glulam-type beam", kh_allowed=False, gamma_M=1.15, rho_mean=460,
               layup="47 mm outer C24 (EN 338), 17/19 mm inner G4-2 (EN 1611-1)"),
    "P3": dict(basis="glulam-type stud (two producers)", kh_allowed=None, gamma_M=1.25, rho_mean=450,
               layup="17-25 mm boards, inner G4-2 (EN 1611-1)"),
    "P4": dict(basis="stud, equivalent to C24 (EN 338)", kh_allowed=None, gamma_M=1.25, rho_mean=420,
               layup="14-25 mm lamellae, outer two per side unjointed"),
    "P5": dict(basis="glulam-type beam", kh_allowed=None, gamma_M=1.25, rho_mean=450,
               layup="46 mm lamellae, no finger joints"),
    "P6": dict(basis="not available", kh_allowed=None, gamma_M=1.25, rho_mean=None,
               layup="T22 outer, T15 inner (database coding)"),
    "P7": dict(basis="glulam-type beam", kh_allowed=False, gamma_M=1.15, rho_mean=470,
               layup="47 mm outer C40 (EN 338), 17/19 mm inner G4-2 (EN 1611-1)"),
}
# P2 and P7 inner boards are appearance grade G4-2 in the approvals, but the
# test database records class_int = C24. The recorded coding is kept.

# glulam_cleaned.csv is already the manuscript sample. Do not drop rows again.
DROP_DUPLICATES = False
EXCLUDE_P2_CLUSTER = False

# Eurocode 5 / EN 1990 constants
H_REF = 600.0            # reference depth for glulam [mm]
KH_EXPONENT = 0.1        # EN 1995-1-1 depth-factor exponent for glulam
KH_CAP = 1.1
KH_EXPONENT_2025 = 0.04  # EN 1995-1-1:2025, Table 5.6 (k_h may be taken as 1.0 for h <= 600 mm)
GAMMA_M_EC5 = 1.25
ALPHA_R = 0.8
BETA_TARGET = 3.8
K_MOD = 1.0
VIDA_M3 = {"GL24h": 13.5, "GL30h": 12.69}   # Vida et al. (2023) depth-law parameters
