import numpy as np

from params import CLASSES_SEG


def get_start_end(x):
    return np.argmax(x), len(x) - np.argmax(x[::-1])


def custom_expand(x0, x1, y0, y1, z0, z1, img_shape, m=1):

    mx = 0
    my = 3
    mz = 1
    x0, x1 = max(0, x0 - mx), min(img_shape[0], x1 + mx)
    y0, y1 = max(0, y0 - my), min(img_shape[1], y1 + my)
    z0, z1 = max(0, z0 - mz), min(img_shape[2], z1 + mz)

    dx = x1 - x0
    x0 -= dx * 0.1
    x1 += dx * 0.1

    dy = y1 - y0

    if dy < 30:
        y0 -= dy * 0.75
        y1 += dy * 0.5
    else:
        y0 -= dy * 0.5
        y1 += dy * 0.25

    dz = z1 - z0
    z0 += dz * 0.25  # -0.1 in v1
    z1 += dz * 0.75  # 0.5 in v1

    # to square - new in v2
    delta = (z1 - z0) - (y1 - y0)
    if delta > 0:
        y1 += delta / 2
        y0 -= delta / 2
    else:
        z0 += delta / 2
        z1 -= delta / 2

    x0, x1 = max(0, x0), min(img_shape[0], x1)
    y0, y1 = max(0, y0), min(img_shape[1], y1)
    z0, z1 = max(0, z0), min(img_shape[2], z1)

    x0, x1, y0, y1, z0, z1 = list(map(int, [x0, x1, y0, y1, z0, z1]))

    return x0, x1, y0, y1, z0, z1


def get_crops(mask, disk="L1/L2", m=5):
    msk = (mask == (CLASSES_SEG.index(disk) + 1)).astype(int)

    x0, x1 = get_start_end(msk.sum((1, 2)) > 3)
    y0, y1 = get_start_end(msk.sum((0, 2)) > 3)
    z0, z1 = get_start_end(msk.sum((0, 1)) > 3)

    x0, x1, y0, y1, z0, z1 = custom_expand(x0, x1, y0, y1, z0, z1, mask.shape)
    return x0, x1, y0, y1, z0, z1
