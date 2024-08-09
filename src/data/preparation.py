import re
import numpy as np
import pandas as pd

from params import (
    CLASSES_SCS,
    CLASSES_NFN,
    SEVERITIES,
    CLASSES_NFN_NOSIDE,
    CLASSES_SS,
    LEVELS,
)


def get_coords(row):
    coords = []
    for i, x, y in zip(row["instance_number"], row["x"], row["y"]):
        coords.append([row["frames"].index(i), x, y])
    return np.array(coords)


def prepare_data(data_path="../input/"):
    df = pd.read_csv(data_path + "train_series_descriptions.csv")

    df["orient"] = df["series_description"].apply(lambda x: x.split()[0])
    df["weighting"] = df["series_description"].apply(lambda x: x.split()[1])

    df["img_path"] = df["study_id"].astype(str) + "_" + df["series_id"].astype(str)
    df["img_path"] = data_path + "npy/" + df["img_path"] + ".npy"

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
    la, lb = row.level.lower().split("/")
    c = re.sub(' ', '_', row.condition.lower())
    try:
        return row[f"{c}_{la}_{lb}"]
    except KeyError:
        return np.nan


def prepare_data_scs(data_path="../input", crop_folder=None, explode=True):
    df = prepare_data(data_path)
    df = df[df["weighting"] == "T2/STIR"].reset_index(drop=True)

    # Add train data
    df_train = pd.read_csv(data_path + "train.csv")
    df_train = df_train[["study_id"] + CLASSES_SCS]

    for c in df_train.columns[1:]:
        df_train[c] = df_train[c].map(dict(zip(SEVERITIES, [0, 1, 2]))).fillna(-1)
    df_train = df_train.astype(int)
    df = df.merge(df_train, on="study_id", how="left")

    # To disk level
    if explode:
        df = df.explode(["condition", "level", "coords"]).dropna().reset_index(drop=True)
        df["target"] = df.apply(get_target, axis=1)
        df.drop(CLASSES_SCS, axis=1, inplace=True)

        if crop_folder is not None:
            df["img_path"] = df["study_id"].astype(str) + "_" + df["series_id"].astype(str)
            lvl = df["level"].apply(lambda x: re.sub('/', '_', x.lower()))
            df["img_path"] = crop_folder + df["img_path"] + "_" + lvl + ".npy"

        try:
            df_coords_crops = pd.read_csv(crop_folder + "df_injury_coords.csv")
            df = df.merge(
                df_coords_crops, on=["study_id", "series_id", "level"], how="left"
            )
            # df["coords_crop"] = df["coords_crop"].apply(lambda x: eval(x)[0])
            df["side"] = df["side"].fillna("Center")

        except FileNotFoundError:
            pass

    return df


def split_left_right(df):
    df_left = df[[c for c in df.columns if "right" not in c.lower()]].copy()
    df_right = df[[c for c in df.columns if "left" not in c.lower()]].copy()

    df_left = df_left.rename(
        columns={c: re.sub("left_", "", c) for c in df_left.columns}
    )
    df_right = df_right.rename(
        columns={c: re.sub("right_", "", c) for c in df_right.columns}
    )

    conds, lvls, coords = [], [], []
    for i, row in df_left.iterrows():
        is_left = np.array(["Left" in c for c in row.condition])
        conds.append(np.array(row.condition)[is_left])
        lvls.append(np.array(row.level)[is_left])
        coords.append(np.array(row.coords)[is_left])
    df_left["condition"] = conds
    df_left["level"] = lvls
    df_left["coords"] = coords

    conds, lvls, coords = [], [], []
    for i, row in df_right.iterrows():
        is_right = np.array(["Right" in c for c in row.condition])
        conds.append(np.array(row.condition)[is_right])
        lvls.append(np.array(row.level)[is_right])
        coords.append(np.array(row.coords)[is_right])
    df_right["condition"] = conds
    df_right["level"] = lvls
    df_right["coords"] = coords

    df_right["side"] = "Right"
    df_left["side"] = "Left"

    df = pd.concat([df_left, df_right], ignore_index=True)
    df = df.sort_values(["study_id", "series_id", "side"], ignore_index=True)

    return df


def prepare_data_nfn(data_path="../input", crop_folder=None, explode=True, left_right=False):
    df = prepare_data(data_path)
    df = df[df["weighting"] == "T1"].reset_index(drop=True)

    # Add train data
    df_train = pd.read_csv(data_path + "train.csv")
    df_train = df_train[["study_id"] + CLASSES_NFN]

    for c in df_train.columns[1:]:
        df_train[c] = df_train[c].map(dict(zip(SEVERITIES, [0, 1, 2]))).fillna(-1)
    df_train = df_train.astype(int)
    df = df.merge(df_train, on="study_id", how="left")

    # To disk level
    if explode:
        df = df.explode(["condition", "level", "coords"]).dropna().reset_index(drop=True)
        df["side"] = df["condition"].apply(lambda x: x.split()[0])
        df["target"] = df.apply(get_target, axis=1)
        df.drop(CLASSES_NFN, axis=1, inplace=True)

        if crop_folder is not None:
            df["img_path"] = df["study_id"].astype(str) + "_" + df["series_id"].astype(str)
            lvl = df["level"].apply(lambda x: re.sub('/', '_', x.lower()))
            df["img_path"] = crop_folder + df["img_path"] + "_" + lvl + ".npy"

        df = df[df["target"] >= 0]  # Remove not visible
        df["target"] = df["target"].astype(int)

        try:
            df_coords_crops = pd.read_csv(crop_folder + "df_injury_coords.csv")
            df = df.merge(
                df_coords_crops,
                on=["study_id", "series_id", "side", "level"],
                how="left",
            )
            df = df.dropna().reset_index(drop=True)  # Removes some SCS
            df["coords_crop"] = df["coords_crop"].apply(lambda x: eval(x)[0])

        except FileNotFoundError:
            pass

    elif left_right:  # No side
        df = df.dropna()
        df = split_left_right(df)
        df = df[df.condition.apply(len) != 0]
        df = df[df[CLASSES_NFN_NOSIDE].max(1) >= 0].reset_index(drop=True)

    return df


def get_target_ss(row):
    la, lb = row.level.lower().split("/")

    try:
        labels = []
        for i, condition in enumerate(row.condition):
            c = re.sub(' ', '_', condition.lower())
            labels.append(row[f"{c}_{la}_{lb}"][i])
        return labels
    except KeyError:
        return np.nan


def prepare_data_ss(data_path="../input/", crop_folder=None, explode=True, left_right=False):
    df = prepare_data(data_path)
    df = df[df["weighting"] == "T2"].reset_index(drop=True)

    if crop_folder is not None:
        df["img_path"] = df["study_id"].astype(str) + "_" + df["series_id"].astype(str)
        df["img_path"] = crop_folder + df["img_path"] + ".npy"

    # Add train data
    df_train = pd.read_csv(data_path + "train.csv")
    df_train = df_train[["study_id"] + CLASSES_SS]

    for c in df_train.columns[1:]:
        df_train[c] = df_train[c].map(dict(zip(SEVERITIES, [0, 1, 2]))).fillna(-1)
    df_train = df_train.astype(int)
    df = df.merge(df_train, on="study_id", how="left")

    # To disk level
    if explode:
        df = df.explode(["condition", "level", "coords"]).dropna().reset_index(drop=True)
        df["side"] = df["condition"].apply(lambda x: x.split()[0])

        df = df.sort_values(['study_id', 'series_id', 'level', 'side'])
        df = df.groupby(
            [
                'study_id', 'series_id', 'series_description',
                'orient', 'weighting', 'img_path', 'level'
            ]
        ).agg(list).reset_index()

        df["target"] = df.apply(get_target_ss, axis=1)
        df.drop(CLASSES_SS, axis=1, inplace=True)

        df['left_subarticular_stenosis'] = df.apply(
            lambda x: x.target[x.side.index("Left")] if "Left" in x.side else -1, axis=1
        )
        df['right_subarticular_stenosis'] = df.apply(
            lambda x: x.target[x.side.index("Right")] if "Right" in x.side else -1, axis=1
        )

        df.drop("target", axis=1, inplace=True)
        df['target_aux'] = df['level'].apply(lambda x: LEVELS.index(x))

    return df


def prepare_data_lvl2(data_path="../input/",):
    df = pd.read_csv(data_path + "train_series_descriptions.csv")
    df = df.groupby('study_id').agg(list)

    # Add train data
    df_train = pd.read_csv(data_path + "train.csv")
    for c in df_train.columns[1:]:
        df_train[c] = df_train[c].map(dict(zip(SEVERITIES, [0, 1, 2]))).fillna(-1)
    df_train = df_train.astype(int)
    df = df.merge(df_train, on="study_id", how="left")

    return df


def prepare_folds(data_path="../input/", k=4):
    """
    Prepare data folds for cross-validation.
    MultilabelStratifiedKFold is used.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../input/".
        k (int, optional): Number of cross-validation folds. Defaults to 4.

    Returns:
        pandas DataFrame: DataFrame containing the study IDs and their respective fold.
    """
    df = pd.read_csv(data_path + "train.csv").fillna(0)
    cols = df.columns[1:]

    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    splits = mskf.split(df, y=df[cols])

    df["fold"] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i

    df_folds = df[["study_id", "fold"]]
    df_folds.to_csv(data_path + f"folds_{k}.csv", index=False)
    return df_folds
