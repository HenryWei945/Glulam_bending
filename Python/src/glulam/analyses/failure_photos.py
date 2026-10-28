"""Fig. 2: typical failures, composed from the original photographs (data/photos) at full resolution."""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from glulam import config, plotting as gp

# panels (a)-(d) of Fig. 2, stored locally as data/photos/fig2_a.jpg ... fig2_d.jpg
PANELS = [config.PHOTO_DIR / f"fig2_{k}.jpg" for k in "abcd"]
ASPECT = 1.6   # width / height of every panel after cropping


def crop(img):
    h, w = img.shape[:2]
    if w / h > ASPECT:                      # too wide: trim the sides
        nw = int(h * ASPECT)
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(w / ASPECT)                    # too tall: keep the upper part, where the beam is
    return img[:nh, :]


def main():
    missing = [path for path in PANELS if not path.is_file()]
    if missing:
        print("Fig. 2 photographs are not in data/photos; the submitted Fig2.jpg is left unchanged.")
        return
    gp.use_latex_style()
    fig, axes = plt.subplots(2, 2, figsize=(gp.DOUBLE, gp.DOUBLE / ASPECT + 0.35),
                             gridspec_kw=dict(wspace=0.03, hspace=0.12))
    for ax, name, lab in zip(axes.ravel(), PANELS, "abcd"):
        ax.imshow(crop(mpimg.imread(name)), interpolation="none")
        ax.set_axis_off()
        ax.set_title(f"({lab})", y=-0.11, fontsize=8)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / "fig_failures.pdf", dpi=600)
    fig.savefig(config.FIGURES / "fig_failures.jpg", dpi=600, pil_kwargs=dict(quality=92))
    plt.close(fig)


if __name__ == "__main__":
    main()
