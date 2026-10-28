"""Copy generated figures to outputs/manuscript_figures after scripts/run_all.py.

The manuscript in ../Submit already contains its numbers. This script does not
edit that file. Compare the copied figures with ../Submit before replacing any
file there. Fig. 2 is copied only when the photographs were available.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DEST = OUT / "manuscript_figures"
FIGS = {"fig_setup.pdf": "Fig1.pdf", "fig_failures.jpg": "Fig2.jpg", "fig_box_product.pdf": "Fig3.pdf",
        "fig_gof_gl24.pdf": "Fig4.pdf", "fig_fm_E_products.pdf": "Fig5.pdf", "fig_layup_P2_P7.pdf": "Fig6.pdf",
        "fig_obs_pred.pdf": "Fig7.pdf", "fig_size_effect.pdf": "Fig8.pdf", "fig_bayes_reliability.pdf": "Fig9.pdf"}

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    for src, dst in FIGS.items():
        source = OUT / "figures" / src
        if source.is_file():
            shutil.copy(source, DEST / dst)
            print(dst)
        else:
            print("missing", src)

