# import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
from natsort import natsorted
from collections import Counter


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def np_dot(a, b):
    return np.sum(a * b, 1)


def read_series_metadata(
    study_id,
    series_id,
    series_description,
    data_path="../input/train_images/",
    advanced_sorting=True,
    return_imgs=True,
):
    dicom_dir = data_path + f"{study_id}/{series_id}"

    # read dicom file
    dicom_file = natsorted(glob.glob(f"{dicom_dir}/*.dcm"))
    instance_number = [int(f.split("/")[-1].split(".")[0]) for f in dicom_file]
    dicom = [pydicom.dcmread(f, stop_before_pixels=False) for f in dicom_file]

    dicom_df, images = [], {}
    for i, d in zip(instance_number, dicom):  # d__.dict__
        images[i] = d.pixel_array if return_imgs else None
        dicom_df.append(
            dotdict(
                study_id=study_id,
                series_id=series_id,
                series_description=series_description,
                instance_number=i,
                ImagePositionPatient=[float(v) for v in d.ImagePositionPatient],
                ImageOrientationPatient=[float(v) for v in d.ImageOrientationPatient],
                PixelSpacing=[float(v) for v in d.PixelSpacing],
                SpacingBetweenSlices=float(d.SpacingBetweenSlices),
                SliceThickness=float(d.SliceThickness),
                grouping=str([round(float(v), 3) for v in d.ImageOrientationPatient]),
            )
        )
    dicom_df = pd.DataFrame(dicom_df)

    # Sort slices
    if advanced_sorting:
        # First cluster by orientation
        dicom_df = [d for _, d in dicom_df.groupby("grouping")]

        # Sort inside each cluster by projection
        data = []
        sort_data_by_group = []
        for df in dicom_df:
            position = np.array(df["ImagePositionPatient"].values.tolist())
            orientation = np.array(df["ImageOrientationPatient"].values.tolist())
            normal = np.cross(orientation[:, :3], orientation[:, 3:])
            projection = np_dot(normal, position)

            df.loc[:, "projection"] = projection
            df = df.sort_values("projection")
            data.append(dotdict(df=df))

            if "sagittal" in series_description.lower():
                sort_data_by_group.append(position[0, 0])  # x
            if "axial" in series_description.lower():
                sort_data_by_group.append(position[0, 2])  # z

        # Sort clusters by position
        data = [r for _, r in sorted(zip(sort_data_by_group, data))]
        for i, r in enumerate(data):
            r.df.loc[:, "group"] = i

        df = pd.concat([r.df for r in data])
    else:  # Sort by z
        if "sagittal" in series_description.lower():
            dicom_df["order"] = dicom_df["ImagePositionPatient"].apply(lambda x: x[0])
        if "axial" in series_description.lower():
            dicom_df["order"] = dicom_df["ImagePositionPatient"].apply(lambda x: x[2])
        df = dicom_df.sort_values("order", ignore_index=True)

    df.loc[:, "z"] = np.arange(len(df))

    images = [images[k] for k in df["instance_number"]]
    return df.reset_index(drop=True), images


def process_2(study, series, orient, data_path="", on_gpu=False):
    df, imgs = read_series_metadata(
        study, series, orient, data_path=data_path, advanced_sorting=(orient == "axial")
    )

    try:
        imgs = np.array(imgs)
    except Exception:
        shapes = Counter([img.shape for img in imgs])
        shape = shapes.most_common()[0][0]
#         print("Different shapes:", shapes, f"resize to {shape} - {orient}")
        imgs = np.array(
            [cv2.resize(img, shape) if img.shape != shape else img for img in imgs]
        )
    return imgs, df
