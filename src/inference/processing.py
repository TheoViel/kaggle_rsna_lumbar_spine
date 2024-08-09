import os
import cv2
import gdcm  # noqa
import pydicom
import numpy as np
from collections import Counter

SO = np.array([0, 1, 0, 0, 0, -1])
AO = np.array([1, 0, 0, 0, 1, 0])


def process(study, series, data_path="", on_gpu=False):
    folder = data_path + f"{study}/{series}/"
    files = os.listdir(folder)
    files.sort(key=lambda x: int(x[:-4]))

    imgs = {}
    for frame, file in enumerate(files):
        dicom = pydicom.dcmread(folder + file)

        weighting = dicom[(0x008, 0x103e)].value

        orient = np.round(dicom[(0x20, 0x37)].value, 1)
        orient = "Sagittal" if (orient * SO).sum() > (orient * AO).sum() else "Axial"

        # Retrieve frame order
        pos = int(file.split("/")[-1][:-4])
        if orient == "Axial":
            pos = -dicom[(0x20, 0x32)].value[-1]
        else:  # Sagittal
            pos = dicom[(0x20, 0x32)].value[0]

        img = dicom.pixel_array

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            print("inv")
            img = 1 - img

        try:
            _ = imgs[pos]
            # print(f"Pos {pos} is already in keys")
            imgs[pos + 0.1] = img  # pos is the same, offset by 0.1
        except KeyError:
            imgs[pos] = img

    assert len(imgs) == len(files), "Missing frames!"

    # order = np.argsort(list(imgs.keys()))

    try:
        imgs = np.array([img for k, img in sorted(imgs.items())])
    except Exception:
        imgs = [img for k, img in sorted(imgs.items())]

        shapes = Counter([img.shape for img in imgs])
        shape = shapes.most_common()[0][0]
        print("Different shapes:", shapes, f"resize to {shape}")

        imgs = np.array(
            [cv2.resize(img, shape) if img.shape != shape else img for img in imgs]
        )
    return imgs, orient, weighting
