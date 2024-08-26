import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Rectangle, FancyArrowPatch

from params import CLASSES_SEG, LEVELS


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


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def draw_slice(
    ax,
    df,
    is_slice=True,
    scolor=[[1, 0, 0]],
    salpha=[0.1],
    is_border=True,
    bcolor=[[1, 0, 0]],
    balpha=[0.1],
    is_origin=True,
    ocolor=[[1, 0, 0]],
    oalpha=[0.1],
    is_arrow=True,
    h=640,
    w=640,
):
    df = df.copy()
    df = df.reset_index(drop=True)

    D = len(df)
    if len(scolor) == 1:
        scolor = scolor * D
    if len(salpha) == 1:
        salpha = salpha * D
    if len(bcolor) == 1:
        bcolor = bcolor * D
    if len(balpha) == 1:
        balpha = balpha * D
    if len(ocolor) == 1:
        ocolor = bcolor * D
    if len(oalpha) == 1:
        oalpha = balpha * D

    # for i,d in df.iterrows():
    for i in range(D):
        d = df.iloc[i]
        o0, o1, o2, o3, o4, o5 = d.ImageOrientationPatient
        ox = np.array([o0, o1, o2])
        oy = np.array([o3, o4, o5])
        sx, sy, sz = d.ImagePositionPatient
        s = np.array([sx, sy, sz])
        delx, dely = d.PixelSpacing

        p0 = s
        p1 = s + w * delx * ox
        p2 = s + h * dely * oy
        p3 = s + h * dely * oy + w * delx * ox

        grid = np.stack([p0, p1, p2, p3]).reshape(2, 2, 3)
        gx = grid[:, :, 0]
        gy = grid[:, :, 1]
        gz = grid[:, :, 2]

        # outline
        if is_slice:
            ax.plot_surface(gx, gy, gz, color=scolor[i], alpha=salpha[i])

        if is_border:
            line = np.stack([p0, p1, p3, p2])
            ax.plot(
                line[:, 0],
                line[:, 1],
                zs=line[:, 2],
                color=ocolor[i],
                alpha=oalpha[i],
                label=LEVELS[i] if not is_arrow else None,
            )

        if is_origin:
            ax.scatter([sx], [sy], [sz], color=ocolor[i], alpha=oalpha[i])

    # check ordering of slice
    if is_arrow:
        sx0, sy0, sz0 = df.iloc[0].ImagePositionPatient
        sx1, sy1, sz1 = df.iloc[-1].ImagePositionPatient
        arrow_prop_dict = dict(
            mutation_scale=20, arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0
        )
        a = Arrow3D([sx0, sx1], [sy0, sy1], [sz0, sz1], **arrow_prop_dict)
        ax.add_artist(a)


def plot_coords(world_point, assigned_level, closest_z, h, w, df_axial, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(world_point[:, 0], world_point[:, 1], world_point[:, 2])

    level_color = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
    ]

    level_ncolor = np.array(level_color) / 255
    coloring = level_ncolor[assigned_level].tolist()

    draw_slice(
        ax,
        df_axial,
        is_slice=True,
        scolor=coloring,
        salpha=[0.1],
        is_border=True,
        bcolor=coloring,
        balpha=[0.2],
        is_origin=False,
        ocolor=[[0, 0, 0]],
        oalpha=[0.0],
        is_arrow=True,
        h=h,
        w=w,
    )

    coloring = level_ncolor[1:].tolist()
    draw_slice(
        ax,
        df_axial.iloc[closest_z],
        is_slice=True,
        scolor=coloring,
        salpha=[0.1],
        is_border=True,
        bcolor=coloring,
        balpha=[1],
        is_origin=False,
        ocolor=[[1, 0, 0]],
        oalpha=[0],
        is_arrow=False,
        h=h,
        w=w,
    )

    ax.set_aspect("equal")
    # ax.set_title(f'axial slice assignment\n series_id:{sagittal_t2_id}')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=0, azim=-10, roll=0)

    plt.legend()
    if title:
        plt.title(title)
    plt.show()
