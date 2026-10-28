# Analysis code for the glulam bending-strength manuscript

The scripts reproduce the tables and figures in the manuscript and in Online Resource 1
from `data/glulam_cleaned.csv`. That file is the analysis sample: 1054 specimens.
Products are the anonymised IDs P1–P7 used in the paper. Commercial names are not stored.

Model numbers in `src/glulam/analyses/regression.py` are the manuscript numbers:

- M1: global modulus of elasticity only
- M2: modulus, depth, width and the two lamella thicknesses
- M3: M2 plus span and the two load positions
- M4: stepwise search from the M2 main effects
- M5: stepwise search with layup categories and product
- M6: stepwise search with the engineered variables
- M7: M6 with squares of the numerical predictors

The partial factors in Section 3.8 come from `bayes_reliability.py`. The conditional
M7 calculation in `model_error.py` and `partial_factor.py` is not reported in the paper.

## Run

```text
pip install -r requirements.txt
python scripts/run_all.py
python scripts/refresh_current_supplement.py
```

`run_all.py` writes `outputs/results`, `outputs/tex` and `outputs/figures`.
Fig. 2 is skipped unless `data/photos/fig2_a.jpg` through `fig2_d.jpg` are added;
the figure in `../Submit` is the one used in the manuscript.
The hierarchical Bayesian step uses the seed in `src/glulam/config.py`.

`scripts/sync_paper.py` copies figure files to `outputs/manuscript_figures`.
Compare them with `../Submit` before replacing a submitted figure.
