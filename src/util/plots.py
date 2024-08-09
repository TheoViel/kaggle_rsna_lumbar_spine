import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from params import CLASSES_SEG


def get_injury_mask(img, coords, level, m=3):
    mask = np.zeros_like(img).astype(np.uint8)
    mask[:, 0, 0] = 1
    mask[:, 1, 0] = 10

    for lvl, coord in zip(level, coords):
        v = CLASSES_SEG.index(lvl) + 1
        x, y = int(coord[2]), int(coord[1])
        mask[int(coord[0]), x - m: x + m + 1, y - m: y + m + 1] = v

    return mask


def plot_mask(img, mask, mask_2=None):
    plt.imshow(img, cmap="gray")
    plt.imshow(
        np.where(mask, mask, np.nan),
        cmap="tab10",
        interpolation="none",
        alpha=0.2,
    )
    if mask_2 is not None:
        plt.imshow(
            np.where(mask_2 > 0, mask_2, np.nan),
            cmap="tab10",
            interpolation="none",
            alpha=0.9,
        )

    plt.axis(False)


def add_rect(x0, x1, y0, y1, z0, z1, f, col="skyblue"):
    rect = Rectangle(
        (z0, y0),
        z1 - z0,
        y1 - y0,
        linewidth=2,
        facecolor="none",
        edgecolor=col,
    )
    if f > x0 and f < x1:
        plt.gca().add_patch(rect)
