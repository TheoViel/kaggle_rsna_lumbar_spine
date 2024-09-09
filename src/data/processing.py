# import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
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
    dicom_file = sorted(
        glob.glob(f"{dicom_dir}/*.dcm"), key=lambda x: int(x.split("/")[-1][:-4])
    )
    instance_number = [int(f.split("/")[-1].split(".")[0]) for f in dicom_file]
    dicom = [pydicom.dcmread(f, stop_before_pixels=False) for f in dicom_file]

    dicom_df, images = [], {}
    for i, d in zip(instance_number, dicom):  # d__.dict__
        images[i] = d.pixel_array if return_imgs else None
        try:
            g = (
                str([round(float(v), 3) for v in d.ImageOrientationPatient])
                if advanced_sorting
                else 0
            )
        except Exception:
            g = 0

        try:
            dicom_df.append(
                dotdict(
                    study_id=study_id,
                    series_id=series_id,
                    series_description=series_description,
                    instance_number=i,
                    ImagePositionPatient=[float(v) for v in d.ImagePositionPatient],
                    ImageOrientationPatient=[
                        float(v) for v in d.ImageOrientationPatient
                    ],
                    PixelSpacing=[float(v) for v in d.PixelSpacing],
                    SpacingBetweenSlices=float(d.SpacingBetweenSlices),
                    SliceThickness=float(d.SliceThickness),
                    grouping=g,
                )
            )
        except Exception:  # Missing fields, fall back to sorting by IPP
            advanced_sorting = False
            dicom_df.append(
                dotdict(
                    study_id=study_id,
                    series_id=series_id,
                    series_description=series_description,
                    instance_number=i,
                    ImagePositionPatient=[float(v) for v in d.ImagePositionPatient],
                    ImageOrientationPatient=-1,
                    PixelSpacing=-1,
                    SpacingBetweenSlices=-1,
                    SliceThickness=-1,
                    grouping=g
                )
            )

    dicom_df = pd.DataFrame(dicom_df)

    # Sort slices
    if advanced_sorting:
        try:
            # Sort inside each cluster by projection
            data = []
            sort_data_by_group = []
            for _, df in dicom_df.groupby("grouping"):
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
        except Exception:
            if "sagittal" in series_description.lower():
                dicom_df["order"] = dicom_df["ImagePositionPatient"].apply(
                    lambda x: x[0]
                )
            if "axial" in series_description.lower():
                dicom_df["order"] = dicom_df["ImagePositionPatient"].apply(
                    lambda x: x[2]
                )
            df = dicom_df.sort_values("order", ignore_index=True)
            df["group"] = 0
    else:  # Sort by z
        if "sagittal" in series_description.lower():
            dicom_df["order"] = dicom_df["ImagePositionPatient"].apply(lambda x: x[0])
        if "axial" in series_description.lower():
            dicom_df["order"] = dicom_df["ImagePositionPatient"].apply(lambda x: x[2])
        df = dicom_df.sort_values("order", ignore_index=True)
        df["group"] = 0

    df.loc[:, "z"] = np.arange(len(df))

    images = [images[k] for k in df["instance_number"]]
    return df.reset_index(drop=True), images


def process_2(study, series, orient, data_path="", on_gpu=False):
    df, imgs = read_series_metadata(
        study,
        series,
        orient,
        data_path=data_path,
        advanced_sorting=False,  # (orient == "axial")
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


def process_and_save(
    study,
    series,
    orient,
    data_path,
    save_folder="",
    save_meta=False,
    save_middle_frame=False,
):
    imgs, df_series = process_2(int(study), int(series), orient, data_path=data_path)
    if save_folder:
        np.save(save_folder + f"npy/{study}_{series}.npy", imgs)
        if save_meta:
            df_series.to_csv(save_folder + f"csv/{study}_{series}.csv", index=False)
        if save_middle_frame:
            img = imgs[len(imgs) // 2]
            img = np.clip(
                img, np.percentile(img.flatten(), 0), np.percentile(img.flatten(), 98)
            )
            max_, min_ = img.max(), img.min()
            if max_ != min_:
                img = (img - min_) / (max_ - min_)
            else:
                img = img - min_
            img = (img * 255).astype(np.uint8)
            cv2.imwrite(save_folder + f"mid/{study}_{series}.png", img)

    return {
        "study_id": study,
        "series_id": series,
        "frames": df_series.instance_number.values.tolist(),
    }
