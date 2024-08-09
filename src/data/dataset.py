import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from collections import Counter, defaultdict
from torch.utils.data import Dataset

from params import LEVELS


def get_frames(frame, n_frames, frames_c, stride=1, max_frame=100):
    """
    Calculate a sequence of frame indices based on the specified parameters.
    If stride is -1, sample n_frames from 0 to max_frame using linear spacing.

    Args:
        frame (int): The central frame index around which the sequence is generated.
        n_frames (int): The number of frames in the sequence.
        frames_c (int): The number of frames to be repeated and offset around each frame.
        stride (int, optional): The step size between frames. Defaults to 1.
        max_frame (int, optional): The maximum frame index allowed. Defaults to 100.

    Returns:
        numpy.ndarray: An array of frame indices representing the calculated sequence.
    """
    frame = int(frame)

    if stride == -1:
        if max_frame <= frames_c * n_frames * 3 + 3:
            frames = np.linspace(
                frames_c, max_frame - frames_c, n_frames, endpoint=True, dtype=int
            )
        else:
            frames = np.linspace(
                frames_c, max_frame - frames_c, n_frames + 2, endpoint=True, dtype=int
            )
            frames = frames[1:-1]

    else:
        frames = np.arange(n_frames) * stride
    frames = frames - frames[n_frames // 2] + frame

    if frames_c:
        offset = np.tile(np.arange(-1, 2) * frames_c, len(frames))
        frames = np.repeat(frames, 3) + offset

    if frames.min() < 0:
        frames -= frames.min()
    elif frames.max() > max_frame:
        frames += max_frame - frames.max()

    frames = np.clip(frames, 0, max_frame)
    # print(frames)
    return frames


class ImageDataset(Dataset):
    """
    Dataset for training 2.5D crop classification models.
    """

    def __init__(
        self,
        df,
        targets="target",
        transforms=None,
        frames_chanel=0,
        n_frames=1,
        stride=1,
        load_in_ram=True,
        train=False,
        **kwargs,
    ):
        """
        Constructor for the CropDataset class.

        Args:
            df (pandas DataFrame): Metadata containing the information.
            transforms (albu transforms, optional): Transforms to apply. Defaults to None.
            frames_chanel (int, optional): Number of frames for channel stacking. Defaults to 0.
            n_frames (int, optional): The number of frames to use. Defaults to 0.
            stride (int, optional): The step size between frames. Defaults to 1.
            train (bool, optional): Whether the dataset is for training. Defaults to False.
        """
        self.df = df
        self.targets = df[targets].values
        self.img_paths = df["img_path"].values

        if "target_aux" in df.columns:
            self.targets_aux = df["target_aux"].values
        else:
            self.targets_aux = np.zeros(len(df))

        self.transforms = transforms
        self.frames_chanel = frames_chanel
        self.n_frames = n_frames
        self.stride = stride

        self.train = train
        self.coords = df["coords"].values

        self.load_in_ram = load_in_ram
        self.load_imgs_in_ram()

    def load_imgs_in_ram(self):
        self.imgs = {}
        if not self.load_in_ram:
            return

        for path in np.unique(self.img_paths):
            img = np.load(path).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            self.imgs[path] = img

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Item accessor. Samples a random frame inside the organ.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [(N,) C, H, W].
            torch.Tensor: Label as a tensor of shape [3].
            int: Dummy value.
        """
        try:
            img = self.imgs[self.img_paths[idx]]
        except KeyError:
            img = np.load(self.img_paths[idx]).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)

        # Pick frame(s)
        try:
            xs = self.coords[idx][:, 0]
        except TypeError:  # No coords
            xs = [len(img) // 2]

        if self.train:
            frame = np.random.choice(xs)
            frame += np.random.choice([-1, 0, 1])
        else:
            frame = Counter(xs).most_common()[0][0]

        frames = get_frames(
            frame,
            self.n_frames,
            self.frames_chanel,
            stride=self.stride,
            max_frame=len(img) - 1,
        )

        # Load
        image = img[np.array(frames)].transpose(1, 2, 0)
        image = image.astype(np.float32) / 255.0

        # image = (image - image.min()) / (image.max() - image.min())

        # Augment
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        y = torch.from_numpy(self.targets[idx])
        y_aux = torch.tensor([self.targets_aux[idx]])

        # Reshape
        if self.frames_chanel:
            image = image.view(self.n_frames, 3, image.size(1), image.size(2))
        else:
            image = image.unsqueeze(1).repeat(1, 3, 1, 1)
        if self.n_frames == 1:
            image = image.squeeze(0)

        return image, y, y_aux


class CropDataset(ImageDataset):
    """
    Dataset for training 2.5D crop classification models.
    """

    def __init__(
        self,
        df,
        targets="target",
        transforms=None,
        frames_chanel=0,
        n_frames=1,
        stride=1,
        use_coords_crop=False,
        load_in_ram=True,
        train=False,
    ):
        """
        Constructor for the CropDataset class.

        Args:
            df (pandas DataFrame): Metadata containing the information.
            transforms (albu transforms, optional): Transforms to apply. Defaults to None.
            frames_chanel (int, optional): Number of frames for channel stacking. Defaults to 0.
            n_frames (int, optional): The number of frames to use. Defaults to 0.
            stride (int, optional): The step size between frames. Defaults to 1.
            train (bool, optional): Whether the dataset is for training. Defaults to False.
        """
        super().__init__(
            df,
            targets=targets,
            transforms=transforms,
            frames_chanel=frames_chanel,
            n_frames=n_frames,
            stride=stride,
            load_in_ram=load_in_ram,
            train=train,
        )

        try:
            self.sides = df["side"].map({"Right": 1, "Center": 4, "Left": 7}).values
        except KeyError:
            self.sides = np.ones(len(df)) * 4  # All Center for scs

        self.use_coords_crop = use_coords_crop
        if "coords_crop" in df.columns:
            self.coords_crop = df["coords_crop"].values

    def __getitem__(self, idx):
        """
        Item accessor. Samples a random frame inside the organ.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [(N,) C, H, W].
            torch.Tensor: Label as a tensor of shape [3].
            int: Dummy value.
        """
        # Load
        try:
            img = self.imgs[self.img_paths[idx]]
        except KeyError:
            img = np.load(self.img_paths[idx]).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)

        # Pick frame(s)
        frame = self.sides[idx] * len(img) // 8

        # print(frame, self.coords_crop[idx][0])

        if self.train:
            if self.use_coords_crop:
                frame_coords = self.coords_crop[idx][0]
                if np.abs(frame_coords - frame) > 3:
                    frame = frame_coords
            frame += np.random.choice([-2, -1, 0, 1, 2])

        frames = get_frames(
            frame,
            self.n_frames,
            self.frames_chanel,
            stride=self.stride,
            max_frame=len(img) - 1,
        )

        image = img[np.array(frames)].transpose(1, 2, 0)
        image = image.astype(np.float32) / 255.0

        # Augment
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        y = torch.zeros(3, dtype=torch.float)
        if self.targets[idx] > -1:
            y[self.targets[idx]] = 1

        # Reshape
        if self.frames_chanel:
            image = image.view(self.n_frames, 3, image.size(1), image.size(2))
        else:
            image = image.unsqueeze(1).repeat(1, 3, 1, 1)
        if self.n_frames == 1:
            image = image[0]

        return image, y, 0


class Seg3dDataset(Dataset):
    """
    Dataset for training 3D segmentation models.

    Attributes:
        df (pandas DataFrame): Metadata containing image and mask information.
        train (bool): Flag indicating whether the dataset is used for training.
        test (bool): Flag indicating whether the dataset is used for testing.
    """

    def __init__(
        self,
        df,
        img_size=(32, 512, 512),
        train=False,
        test=False,
        load_in_ram=False,
    ):
        """
        Constructor.

        Args:
            df (pandas DataFrame): Metadata containing image and mask information.
            train (bool, optional): Whether the dataset is used for training. Defaults to False.
            test (bool, optional): Whether the dataset is used for testing. Defaults to False.
        """
        self.df = df
        self.train = train
        self.test = test
        self.img_size = img_size

        self.img_paths = df["img_path"].values
        self.mask_paths = df["mask_path"].values if "mask_path" in df.columns else []
        # self.class_weights = np.vstack(df['source'].map({
        #     # "spider": [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        #     "spider": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     "spine_seg": [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # L5_S1 not defined
        # }).values)

        self.transforms = None
        if train:
            import monai.transforms as transforms

            # https://docs.monai.io/en/0.3.0/transforms.html
            self.transforms = transforms.Compose(
                [
                    # transforms.RandAffined(
                    #     translate_range=[s // 10 for s in self.img_size],
                    #     # rotate_range=(-0.1, 0.1),
                    #     padding_mode="zeros",
                    #     keys=["image", "mask"],
                    #     mode="nearest",
                    #     prob=0.25,
                    # ),
                    transforms.RandZoomd(
                        min_zoom=0.8,
                        max_zoom=1.2,
                        mode="nearest",
                        keys=["image", "mask"],
                        prob=0.25,
                    ),
                    transforms.RandAdjustContrastd(
                        gamma=(0.5, 2), keys=["image"], prob=0.25
                    ),
                    transforms.RandStdShiftIntensityd(
                        factors=0.1, keys=["image"], prob=0.25
                    ),
                ]
            )

        self.imgs = {}
        self.masks = {}
        if not test and load_in_ram:
            for idx in range(len(self.img_paths)):
                self.imgs[self.img_paths[idx]] = np.load(self.img_paths[idx]).astype(
                    np.float32
                )
                self.masks[self.mask_paths[idx]] = np.load(self.mask_paths[idx]).astype(
                    np.uint8
                )

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_paths)

    def resize(self, x, mode="nearest"):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            x = F.interpolate(
                x.unsqueeze(0),
                self.img_size,
                mode=mode,
            )[0].numpy()
        else:
            x = F.interpolate(
                x.unsqueeze(0),
                self.img_size,
                mode=mode,
            )[0]
        return x

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor.
            torch.Tensor: Mask as a tensor (if not for testing).
            int: Dummy value.
        """
        image = self.imgs.get(
            self.img_paths[idx],
            np.load(self.img_paths[idx]).astype(np.float32),
        )[None]
        image = self.resize(image, "trilinear")

        image = (image - image.min()) / (image.max() - image.min()) * 255

        if not self.test:
            mask = self.masks.get(
                self.mask_paths[idx], np.load(self.mask_paths[idx]).astype(np.uint8)
            )[None]
            mask = self.resize(mask)
        else:
            mask = 0

        if self.transforms is not None:
            res = self.transforms({"image": image, "mask": mask})
            image = res["image"].as_tensor().float() / 255.0
            mask = res["mask"].as_tensor()
        else:
            image = torch.from_numpy(image).float() / 255.0
            if not self.test:
                mask = torch.from_numpy(mask)

        w = 0
        # if not self.test:
        #     w = torch.from_numpy(self.class_weights[idx])
        #     w = torch.isin(mask, torch.where(w)[0])[0]

        return image, mask, w


class FeatureDataset(Dataset):
    """
    Dataset for training RNN models.
    """
    def __init__(
        self,
        df,
        exp_folders,
        targets="target",
        resize=None,
    ):
        self.df = df
        self.targets = df[targets].values

        self.resize = resize
        self.exp_folders = exp_folders

        self.series_dict = self.get_series_dict(df)

        self.sizes = {
            "nfn": resize,
            "scs": resize,
            "ss": resize * 2,
            "ss_aux": resize * 2,
        }

        self.dummies = {}
        self.fts = {}

        study = self.df["study_id"][0]
        series = self.series_dict[study]

        for k in self.exp_folders:
            if "crop" in k:
                self.fts[k] = self.load_fts(self.exp_folders[k])
            else:
                self.dummies[k] = 0.33 + np.zeros_like(
                    np.load(self.exp_folders[k] + f'preds/{study}_{series[k.split("_")[0]]}.npy')
                )

    @staticmethod
    def get_series_dict(df):
        series_dict = defaultdict(dict)
        for study, df_study in df.explode(['series_id', 'series_description']).groupby("study_id"):
            series = df_study.set_index("series_description")["series_id"]
            try:
                series_dict[study]["scs"] = series["Sagittal T2/STIR"]
            except KeyError:
                series_dict[study]["scs"] = 0

            try:
                series_dict[study]["nfn"] = series["Sagittal T1"]
            except KeyError:
                series_dict[study]["nfn"] = 0
                # print(study, "nfn")

            try:
                series_dict[study]["ss"] = series["Axial T2"]
            except KeyError:
                series_dict[study]["ss"] = 0

            for k in series_dict[study]:
                if not isinstance(series_dict[study][k], int):
                    series_dict[study][k] = series_dict[study][k].values[0]

        return series_dict

    def __len__(self):
        return len(self.df)

    @staticmethod
    def load_fts(exp_folder):
        fts = {}
        for fold in range(4):
            preds = np.load(exp_folder + f'pred_inf_{fold}.npy')
            df = pd.read_csv(
                exp_folder + f'df_val_{fold}.csv'
            )[["study_id", "series_id", "level", "side"]].astype(str)
            index = ["_".join(row.tolist()) for row in df.values]
            fts.update(dict(zip(index, preds)))
        return fts

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
                for lvl in LEVELS:
                    for side in sides:
                        # print(f'{study}_{series[k_]}_{lvl}_{side}')
                        ft.append(
                            self.fts[k].get(f'{study}_{series[k_]}_{lvl}_{side}', np.zeros(3))
                        )
                ft = np.vstack(ft)

            else:
                sub_folder = "preds_aux/" if "aux" in k else "preds/"
                try:
                    ft = np.load(self.exp_folders[k] + f'{sub_folder}/{study}_{series[k_]}.npy')
                    # print(k, ft.shape)
                except FileNotFoundError:
                    ft = self.dummies[k]

            if k in self.sizes:
                ft = self.resize_fts(ft, self.sizes[k])
                ft = ft.reshape(self.sizes[k], -1)
            else:
                ft = ft.flatten()
            fts[k] = torch.from_numpy(ft).float().contiguous()

        y = torch.from_numpy(self.targets[idx])

        return fts, y, 0