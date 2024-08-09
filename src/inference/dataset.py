import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import Dataset

from data.dataset import get_frames
from params import LEVELS_


class ImageInfDataset(Dataset):
    """
    Dataset for infering 2D classification models.
    """
    def __init__(
        self,
        imgs,
        transforms=None,
        frames_chanel=0,
        n_frames=1,
        stride=1,
    ):
        """
        Constructor.

        Args:
            imgs (np array): Images stack.
            transforms (albu transforms, optional): Transforms to apply. Defaults to None.
            frames_chanel (int, optional): Number of frames for channel stacking. Defaults to 0.
            n_frames (int, optional): The number of frames to use. Defaults to 1.
            stride (int, optional): The step size between frames. Defaults to 1.
        """
        self.imgs = imgs
        self.transforms = transforms

        assert imgs.dtype == np.uint8
        assert imgs.max() <= 255

        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride

        assert self.n_frames == 1

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Item accessor.
        Refer to _getitem_feature if features are precomputed.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor.
            int: Dummy value.
            int: Dummy value.
        """
        frames = get_frames(
            idx,
            1,
            self.frames_chanel,
            stride=self.stride,
            max_frame=len(self.imgs) - 1,
        )

        # Load
        image = self.imgs[np.array(frames)].transpose(1, 2, 0)
        image = image.astype(np.float32) / 255.0

        # Augment
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        # Reshape
        if self.frames_chanel:
            image = image.view(3, image.size(1), image.size(2))
        else:
            image = image.unsqueeze(0).repeat(3, 1, 1)

        return image, 0, 0


class FeatureInfDataset(Dataset):
    """
    Dataset for training RNN models.
    """
    def __init__(
        self,
        df,
        exp_folders,
        crop_fts,
        resize=None,
        save_folder="../output/tmp",
    ):
        self.df = df.drop_duplicates(keep="first", subset="study_id", ignore_index=True)
        self.resize = resize
        self.exp_folders = exp_folders
        self.save_folder = save_folder

        self.series_dict = self.get_series_dict(df)

        self.sizes = {
            "nfn": resize,
            "scs": resize,
            "ss": resize * 2,
            "ss_aux": resize * 2,
        }

        self.dummies = {}
        self.fts = crop_fts

        study = self.df["study_id"][0]
        series = self.series_dict[study]
        for k in self.exp_folders:
            k_ = k.split("_")[0]
            self.dummies[k] = 0.33 + np.zeros_like(
                np.load(self.save_folder + f'{study}_{series[k_]}_{k_}.npy')
            )

    @staticmethod
    def get_series_dict(df):
        series_dict = defaultdict(dict)
        for study, df_study in df.groupby("study_id"):
            series = df_study.set_index("series_description")["series_id"]
            for k in ["scs", "nfn", "ss"]:
                try:
                    series_dict[study][k] = series[k]
                except KeyError:
                    series_dict[study][k] = 0

            for k in series_dict[study]:
                if not isinstance(series_dict[study][k], int):
                    try:
                        series_dict[study][k] = int(series_dict[study][k])
                    except TypeError:  # Several series, use 1st one
                        series_dict[study][k] = int(series_dict[study][k].values[0])

        return series_dict

    def __len__(self):
        return len(self.df)

    @staticmethod
    def resize_fts(fts, size):
        if len(fts.shape) == 2:  # Add chanel
            fts = fts[:, None]

        fts = fts.transpose(1, 2, 0)  # n x 3 x t -> t x n x 3
        fts = F.interpolate(
            torch.from_numpy(fts).float(), size=size, mode="linear"
        ).numpy()
        fts = fts.transpose(2, 0, 1)
        return fts

    def __getitem__(self, idx):
        study = self.df["study_id"][idx]
        series = self.series_dict[study]

        fts = {}
        for k in self.exp_folders:
            k_ = k.split("_")[0]
            # print(k_)
            if "crop" in k:
                sides = ['Center'] if "scs" in k else ["Left", "Right"]
                ft = []
                for lvl in LEVELS_:
                    for side in sides:
                        # print(f'{study}_{series[k_]}_{lvl}_{side}')
                        ft.append(
                            self.fts[k].get(f'{study}_{series[k_]}_{lvl}_{side}', np.zeros(3))
                        )
                ft = np.vstack(ft)

            else:
                try:
                    ft = np.load(self.save_folder + f'{study}_{series[k_]}_{k_}.npy')
                    # print(k, ft.shape)
                except FileNotFoundError:
                    ft = self.dummies[k]

            # print(k, ft.shape)
            if k in self.sizes:
                ft = self.resize_fts(ft, self.sizes[k])
                ft = ft.reshape(self.sizes[k], -1)
            else:
                ft = ft.flatten()
            fts[k] = torch.from_numpy(ft).float().contiguous()

        return fts, 0, 0
