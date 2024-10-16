import re
import numpy as np
import pandas as pd
from collections import defaultdict

from params import (
    CLASSES_SCS,
    SEVERITIES,
    CLASSES_CROP,
    LEVELS,
)


def get_coords(row):
    """
    Extracts coordinates from a row of data.

    Args:
        row (pandas Series): Row containing 'instance_number', 'x', 'y', and 'frames' columns.

    Returns:
        np.array: Coordinates.
    """
    coords = []
    for i, x, y in zip(row["instance_number"], row["x"], row["y"]):
        coords.append([row["frames"].index(i), x, y])
    return np.array(coords)


def prepare_data(data_path="../input/"):
    """
    Prepares data for training by loading and processing CSV files.

    Args:
        data_path (str, optional): The path to the input data directory. Defaults to "../input/".

    Returns:
        pd.DataFrame: A DataFrame containing the prepared data with additional columns for
                      orientation, weighting, image paths, and coordinates.
    """
    df = pd.read_csv(data_path + "train_series_descriptions.csv")

    df["orient"] = df["series_description"].apply(lambda x: x.split()[0])
    df["weighting"] = df["series_description"].apply(lambda x: x.split()[1])

    df["img_path"] = df["study_id"].astype(str) + "_" + df["series_id"].astype(str)
    df["img_path"] = data_path + "npy2/" + df["img_path"] + ".npy"

    labels = pd.read_csv(data_path + "train_label_coordinates.csv")
    labels = labels.groupby(["study_id", "series_id"]).agg(list).reset_index()

    frames = pd.read_csv("../input/df_frames.csv")
    frames["frames"] = frames["frames"].apply(eval)

    labels = labels.merge(frames, how="left")

    labels["coords"] = labels.apply(get_coords, axis=1)
    df = df.merge(
        labels[["study_id", "series_id", "condition", "level", "coords"]], how="left"
    )
    return df


def get_target(row):
    """
    Extracts the target value from a row based on the condition and level.

    Args:
        row (pandas Series): A row containing 'level', 'condition', and relevant target columns.

    Returns:
        float or np.nan: The target value if found, otherwise np.nan.
    """
    la, lb = row.level.lower().split("/")
    c = re.sub(" ", "_", row.condition.lower())
    try:
        return row[f"{c}_{la}_{lb}"]
    except KeyError:
        return np.nan


def prepare_data_scs(data_path="../input/", crop_folder=None, explode=True):
    """
    Prepares data for the SCS task by loading and processing CSV files.

    Args:
        data_path (str, optional): The path to the input data directory. Defaults to "../input/".
        crop_folder (str, optional): Path to the folder containing cropped images. Defaults to None.
        explode (bool, optional): Whether to explode the DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared data.
    """
    df = prepare_data(data_path)
    df = df[df["weighting"] == "T2/STIR"].reset_index(drop=True)

    df_train = pd.read_csv(data_path + "train.csv")
    df_train = df_train[["study_id"] + CLASSES_SCS]

    for c in df_train.columns[1:]:
        df_train[c] = df_train[c].map(dict(zip(SEVERITIES, [0, 1, 2]))).fillna(-1)
    df_train = df_train.astype(int)
    df = df.merge(df_train, on="study_id", how="left")

    if explode:
        df = (
            df.explode(["condition", "level", "coords"]).dropna().reset_index(drop=True)
        )
        df["target"] = df.apply(get_target, axis=1)
        df.drop(CLASSES_SCS, axis=1, inplace=True)

        if crop_folder is not None:
            df["img_path"] = (
                df["study_id"].astype(str) + "_" + df["series_id"].astype(str)
            )
            lvl = df["level"].apply(lambda x: re.sub("/", "_", x.lower()))
            df["img_path"] = crop_folder + df["img_path"] + "_" + lvl + ".npy"

        try:
            df_coords_crops = pd.read_csv(crop_folder + "df_injury_coords.csv")
            df = df.merge(
                df_coords_crops, on=["study_id", "series_id", "level"], how="left"
            )
            df["side"] = df["side"].fillna("Center")

        except FileNotFoundError:
            df["side"] = "Center"
            pass

    return df


def prepare_data_lvl2(data_path="../input/"):
    """
    Prepares data for the Level 2 task by loading and processing CSV files.

    Args:
        data_path (str, optional): The path to the input data directory. Defaults to "../input/".

    Returns:
        pd.DataFrame: A DataFrame containing the prepared data with additional columns for targets.
    """
    df = pd.read_csv(data_path + "train_series_descriptions.csv")
    df = df.groupby("study_id").agg(list)

    df_train = pd.read_csv(data_path + "train.csv")
    for c in df_train.columns[1:]:
        df_train[c] = df_train[c].map(dict(zip(SEVERITIES, [0, 1, 2]))).fillna(-1)
    df_train = df_train.astype(int)
    df = df.merge(df_train, on="study_id", how="left")

    return df


def get_target_crop(row):
    """
    Extracts the target values for cropping from a row based on the condition and level.

    Args:
        row (pandas Series): A row from a DataFrame containing 'level' and relevant target columns.

    Returns:
        list or np.nan: A list of target values if found, otherwise np.nan.
    """
    la, lb = row.level.lower().split("/")
    try:
        labels = []
        for c in CLASSES_CROP:
            labels.append(row[f"{c}_{la}_{lb}"])
        return labels
    except KeyError:
        return np.nan


def simplify_coords(x):
    """
    Simplifies the coordinates by calculating the median of the x-coordinates for each side.

    Args:
        x (pandas Series): A row from a DataFrame containing 'condition' and 'coords' columns.

    Returns:
        np.array: An array of simplified coordinates for the left, center, and right sides.
    """
    coords = -1 * np.ones(3)
    d = defaultdict(list)

    if isinstance(x.condition, float):
        return coords

    for i, c in enumerate(x.condition):
        side = c.split(" ")[0]
        side = "Center" if side == "Spinal" else side
        d[side].append(x.coords[i][0])

    for k in d:
        d[k] = int(np.median(d[k]))

    coords[0] = d.get("Right", -1)
    coords[1] = d.get("Center", -1)
    coords[2] = d.get("Left", -1)
    return coords


def prepare_data_crop(data_path, crop_folder=None, axial=False):
    """
    Prepares data for the crop task by loading and processing CSV files.

    Args:
        data_path (str): The path to the input data directory.
        crop_folder (str, optional): Path to the folder containing cropped images. Defaults to None.
        axial (bool, optional): Whether to use axial orientation. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared data.
    """
    df = prepare_data(data_path)

    df["level"] = [
        ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"] for _ in range(len(df))
    ]
    df["side"] = "Center"

    if axial:
        df = df[df["orient"] == "Axial"].reset_index(drop=True)
    else:
        df = df[df["orient"] == "Sagittal"].reset_index(drop=True)

    df["coords"] = df.apply(simplify_coords, axis=1)
    df["coords"] = df["coords"].apply(
        lambda x: int(np.mean(x[x > -1])) if x.max() > -1 else -1
    )
    df.drop("condition", axis=1, inplace=True)

    df_train = pd.read_csv(data_path + "train.csv")
    for c in df_train.columns[1:]:
        df_train[c] = df_train[c].map(dict(zip(SEVERITIES, [0, 1, 2]))).fillna(-1)
    df_train = df_train.astype(int)
    df = df.merge(df_train, on="study_id", how="left")

    df = df.explode("level").reset_index(drop=True)
    df["target"] = df.apply(get_target_crop, axis=1)
    df.drop(df_train.columns[1:], axis=1, inplace=True)

    if crop_folder is not None:
        df["img_path"] = df["study_id"].astype(str) + "_" + df["series_id"].astype(str)
        lvl = df["level"].apply(lambda x: re.sub("/", "_", x.lower()))
        df["img_path"] = crop_folder + df["img_path"] + "_" + lvl + ".npy"

    return df


def get_coords_target(row, axial=False, relative=False):
    """
    Extracts the target coordinates from a row based on the side or level.

    Args:
        row (pandas Series): A row from a DataFrame containing 'side', 'x', 'y',
                             'level', 'relative_x', and 'relative_y' columns.
        axial (bool, optional): Whether to use axial orientation. Defaults to False.
        relative (bool, optional): Whether to use relative coordinates. Defaults to False.

    Returns:
        np.array: An array of target coordinates.
    """
    if axial:
        target = np.zeros((2, 2), dtype=float) - 1
        for s, side in enumerate(["Left", "Right"]):
            i = row.side.index(side)
            if relative:
                target[s, 0] = row.relative_x[i]
                target[s, 1] = row.relative_y[i]
            else:
                target[s, 0] = row.x[i]
                target[s, 1] = row.y[i]
    else:
        target = np.zeros((5, 2), dtype=float) - 1
        for lvl, level in enumerate(LEVELS):
            if level in row.level:
                i = row.level.index(level)
                if relative:
                    target[lvl, 0] = row.relative_x[i]
                    target[lvl, 1] = row.relative_y[i]
                else:
                    target[lvl, 0] = row.x[i]
                    target[lvl, 1] = row.y[i]
    return target


def prepare_coords_data(data_path="../input/coords/", axial=False, use_ext=True):
    """
    Prepares coordinate data by loading and processing CSV files.

    Args:
        data_path (str, optional): Path to the input data directory. Defaults to "../input/coords/".
        axial (bool, optional): Whether to use axial orientation. Defaults to False.
        use_ext (bool, optional): Whether to use external data. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared coordinate data.
    """
    if axial:
        df = pd.read_csv(data_path + "coords_ax.csv")
        df = df.groupby(["study_id", "series_id", "img_path"]).agg(list).reset_index()

        df["target"] = df.apply(
            lambda x: get_coords_target(x, axial=True, relative=False), axis=1
        ).tolist()
        df["target_rel"] = df.apply(
            lambda x: get_coords_target(x, axial=True, relative=True), axis=1
        ).tolist()
    else:
        if use_ext:
            df = pd.read_csv(data_path + "coords_pretrain.csv")
            df["img_path"] = (
                data_path + "data/processed_" + df["source"] + "_jpgs/" + df["filename"]
            )
            df = df.sort_values(["source", "filename", "level"], ignore_index=True)
            df = df.rename(columns={"source": "study_id", "filename": "series_id"})

            df = pd.concat(
                [df, pd.read_csv(data_path + "coords_comp.csv")], ignore_index=True
            )
        else:
            df = pd.read_csv(data_path + "coords_comp_3ch.csv")

        df = df.groupby(["study_id", "series_id", "img_path"]).agg(list).reset_index()

        df["target"] = df.apply(
            lambda x: get_coords_target(x, relative=False), axis=1
        ).tolist()
        df["target_rel"] = df.apply(
            lambda x: get_coords_target(x, relative=True), axis=1
        ).tolist()
    return df
