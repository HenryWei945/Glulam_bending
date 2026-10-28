"""CSV export and booktabs LaTeX snippets for the manuscript."""

import pandas as pd

from . import config


def save_csv(df, name):
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_tex(df, name, caption, label, column_format=None, escape=False, note=None):
    """Write a complete table environment (booktabs) that can be \\input into the manuscript."""
    config.TEX.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=False, escape=escape, column_format=column_format,
                       na_rep="--")
    lines = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", body.strip()]
    if note:
        lines += [f"\\par\\smallskip\\raggedright\\scriptsize {note}"]
    lines.append("\\end{table}")
    path = config.TEX / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def fmt(df, spec):
    """Format numeric columns: spec = {column: format string or callable}."""
    out = df.copy()
    for c, f in spec.items():
        if c in out:
            out[c] = out[c].map((lambda v, f=f: f.format(v) if pd.notna(v) else "--") if isinstance(f, str) else f)
    return out


def pval(p):
    if pd.isna(p):
        return "--"
    if p < 1e-3:
        mant, exp = f"{p:.1e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return f"{p:.3f}"
