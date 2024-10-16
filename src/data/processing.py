import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
from collections import Counter


class dotdict(dict):
    """
    A dictionary that allows accessing its elements as attributes.
    """

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        """
        Get an attribute from the dictionary.

        Args:
            name (str): The name of the attribute to get.

        Returns:
            The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def read_series_metadata(
    study_id,
    series_id,
    series_description,
    data_path="../input/train_images/",
    return_imgs=True,
):
    """
    Reads the metadata and images from a DICOM series.

    Args:
        study_id (int): The ID of the study.
        series_id (int): The ID of the series.
        series_description (str): The description of the series.
        data_path (str, optional): The path to the input data. Defaults to "../input/train_images/".
        return_imgs (bool, optional): Whether to return the images. Defaults to True.

    Returns:
        tuple: A tuple containing the DataFrame with metadata and the list of images.
    """
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
                    grouping=0,
                )
            )
        except Exception:  # Missing fields
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
                    grouping=0,
                )
            )

    dicom_df = pd.DataFrame(dicom_df)

    # Sort slices
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
    """
    Processes the DICOM series and returns the images and metadata.

    Args:
        study (int): The ID of the study.
        series (int): The ID of the series.
        orient (str): The orientation of the series.
        data_path (str, optional): The path to the input data directory. Defaults to "".
        on_gpu (bool, optional): Whether to process on GPU. Defaults to False.

    Returns:
        tuple: A tuple containing the processed images and the DataFrame with metadata.
    """
    df, imgs = read_series_metadata(
        study,
        series,
        orient,
        data_path=data_path,
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
    """
    Processes the DICOM series and saves the images and metadata.

    Args:
        study (int): The ID of the study.
        series (int): The ID of the series.
        orient (str): The orientation of the series.
        data_path (str): The path to the input data directory.
        save_folder (str, optional): Path to the saving folder. Defaults to "".
        save_meta (bool, optional): Whether to save the metadata. Defaults to False.
        save_middle_frame (bool, optional): Whether to save the middle frame. Defaults to False.

    Returns:
        dict: A dictionary containing the study ID, series ID, and the list of frame numbers.
    """
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
