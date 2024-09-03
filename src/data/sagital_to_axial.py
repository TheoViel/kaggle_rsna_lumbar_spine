import numpy as np
from data.processing import read_series_metadata


def project_to_3d(x, y, z, df):
    d = df.iloc[z]
    # h, w = d.h, d.w
    sx, sy, sz = [float(v) for v in d.ImagePositionPatient]
    (
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
    ) = [float(v) for v in d.ImageOrientationPatient]
    delx, dely = d.PixelSpacing

    xx = o0 * delx * x + o3 * dely * y + sx
    yy = o1 * delx * x + o4 * dely * y + sy
    zz = o2 * delx * x + o5 * dely * y + sz
    return xx, yy, zz


def view_to_world(sagittal_t2_point, z, sagittal_t2_df):
    xxyyzz = []
    for i in range(1, 6):
        x, y = sagittal_t2_point[i - 1]
        xx, yy, zz = project_to_3d(x, y, z, sagittal_t2_df)
        xxyyzz.append((xx, yy, zz))

    xxyyzz = np.array(xxyyzz)
    return xxyyzz


def point_to_level(world_point, axial_df, return_closest_frame=False):
    """
    Get closest axial slices (z) to the CSC world points
    """
    xxyyzz = world_point
    orientation = np.array(axial_df.ImageOrientationPatient.values.tolist())
    position = np.array(axial_df.ImagePositionPatient.values.tolist())
    ox = orientation[:, :3]
    oy = orientation[:, 3:]
    oz = np.cross(ox, oy)
    t = xxyyzz.reshape(-1, 1, 3) - position.reshape(1, -1, 3)
    dis = (oz.reshape(1, -1, 3) * t).sum(-1)  # np.dot(point-s,oz)
    fdis = np.fabs(dis)
    closest_z = fdis.argmin(-1)

    if return_closest_frame:
        return closest_z

    # hard/soft assigment, multi/single assigment
    # no assignment based on distance
    # allow point found in multi group
    num_group = len(axial_df["group"].unique())
    point_group = axial_df.group.values[fdis.argsort(-1)[:, :3]].tolist()
    point_group = [list(set(g)) for g in point_group]
    group_point = [[] for g in range(num_group)]
    for i in range(5):
        for k in point_group[i]:
            group_point[k].append(i)
    group_point = [sorted(list(set(g))) for g in group_point]

    D = len(axial_df)
    assigned_level = np.full(D, fill_value=0, dtype=int)
    for group in range(num_group):
        point_in_this_group = np.array(
            group_point[group]
        )  # np.where(closest_df['group'] == group)[0]
        slice_in_this_group = np.where(axial_df["group"] == group)[0]
        if len(point_in_this_group) == 0:
            continue  # unassigned, level=0

        level = (
            point_in_this_group[
                fdis[point_in_this_group][:, slice_in_this_group].argmin(0)
            ]
            + 1
        )
        assigned_level[slice_in_this_group] = level

    return assigned_level, closest_z, dis


# back project 3d to 2d
def backproject_to_2d(xx, yy, zz, df):

    d = df.iloc[0]
    sx, sy, sz = [float(v) for v in d.ImagePositionPatient]
    (
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
    ) = [float(v) for v in d.ImageOrientationPatient]
    delx, dely = d.PixelSpacing
    delz = d.SpacingBetweenSlices

    ox = np.array([o0, o1, o2])
    oy = np.array([o3, o4, o5])
    oz = np.cross(ox, oy)

    p = np.array([xx - sx, yy - sy, zz - sz])
    x = np.dot(ox, p) / delx
    y = np.dot(oy, p) / dely
    z = np.dot(oz, p) / delz
    x = int(round(x))
    y = int(round(y))
    z = int(round(z))

    D, h, w = len(df), d.h, d.w
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (z >= 0) & (z < D)
    if not inside:
        # print('out-of-bound')
        return False, 0, 0, 0, 0

    n = df.instance_number.values[int(round(z))]
    return True, x, y, z, n


def get_axial_coords(
    study,
    series,
    coords,
    h,
    w,
    df,
    data_path="../input/train_images/",
    world_point=None,
):
    df_sagittal, _ = read_series_metadata(
        study,
        series,
        "sagittal",
        data_path=data_path,
        advanced_sorting=False,
        return_imgs=False,
    )

    coords[:, 0] *= w
    coords[:, 1] *= h

    df_s = df[df["study_id"] == study]

    df_axial, _ = read_series_metadata(
        study,
        df_s[df_s["orient"] == "Axial"].series_id.values[0],
        "axial",
        data_path=data_path,
    )

    if world_point is None:
        world_point = view_to_world(coords, len(df_sagittal) // 2, df_sagittal)
    assigned_level, closest_z, dis = point_to_level(world_point, df_axial)

    return world_point, assigned_level, closest_z, df_axial
