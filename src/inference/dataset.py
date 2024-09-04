import numpy as np
from torch.utils.data import Dataset

from data.dataset import get_frames, FeatureDataset


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


class FeatureInfDataset(FeatureDataset):
    """
    Dataset for infering feature models.
    Heavily inherits from FeatureDataset.
    """
    def __init__(
        self,
        df,
        exp_folders,
        crop_fts,
        save_folder="../output/tmp",
    ):
        self.df = df
        self.exp_folders = exp_folders

        self.targets = np.zeros((len(df), 1))

        self.series_dict = self.get_series_dict(df)

        self.dummies = {
            "scs_crop": np.zeros(3),
            "nfn_crop": np.zeros(3),
            "crop": np.zeros((5, 3)),
            "dh": np.zeros((25, 3)),
            "ch": np.zeros((25, 3)),
        }

        self.fts = crop_fts
