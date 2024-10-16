import cv2
import torch
import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from torch.utils.data import Dataset

from params import LEVELS, LEVELS_


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
    Base dataset for loading images and their corresponding targets for classification tasks.
    """
    def __init__(
        self,
        df,
        targets="target",
        transforms=None,
        frames_chanel=0,
        n_frames=1,
        stride=1,
        train=False,
        **kwargs,
    ):
        """
        Constructor for the ImageDataset class.

        Args:
            df (pandas DataFrame): Metadata containing image paths and targets.
            targets (str, optional): Column name for the target values. Defaults to "target".
            transforms (callable, optional): Transforms to apply to the images. Defaults to None.
            frames_chanel (int, optional): Number of frames for channel stacking. Defaults to 0.
            n_frames (int, optional): The number of frames to use. Defaults to 1.
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
        self.coords = None

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Item accessor. Loads an image and its corresponding target.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor.
            torch.Tensor: Labels as a tensor.
            torch.Tensor: Auxiliary labels as a tensor.
        """
        img = np.load(self.img_paths[idx]).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)

        # Pick frame(s)
        try:
            xs = self.coords[idx][:, 0]
        except Exception:  # No coords
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
        image = (image - image.min()) / (image.max() - image.min())

        # Augment
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        y_aux = torch.tensor([self.targets_aux[idx]])

        tgt = self.targets[idx]
        if isinstance(self.targets[idx], (int, float, np.int64, np.int32)):
            y = torch.zeros(3, dtype=torch.float)
            if tgt > -1:
                y[tgt] = 1
        else:
            y = torch.zeros((len(tgt), 3), dtype=torch.float)
            for i in range(len(tgt)):
                if tgt[i] > -1:
                    y[i, tgt[i]] = 1

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
        train=False,
        flip=False,
    ):
        """
        Constructor for the CropDataset class.

        Args:
            df (pandas DataFrame): Metadata containing image paths and targets.
            targets (str, optional): Column name for the target values. Defaults to "target".
            transforms (callable, optional): Transforms to apply to the images. Defaults to None.
            frames_chanel (int, optional): Number of frames for channel stacking. Defaults to 0.
            n_frames (int, optional): The number of frames to use. Defaults to 1.
            stride (int, optional): The step size between frames. Defaults to 1.
            train (bool, optional): Whether the dataset is for training. Defaults to False.
            flip (bool, optional): Whether to apply flipping augmentation. Defaults to False.
        """
        super().__init__(
            df,
            targets=targets,
            transforms=transforms,
            frames_chanel=frames_chanel,
            n_frames=n_frames,
            stride=stride,
            train=train,
        )
        try:
            if isinstance(self.targets[0], list):
                self.targets = np.vstack(self.targets)
        except Exception:  # will not work with PL
            pass

        try:
            if "coords_crops" in df["img_path"][0]:
                self.sides = df["side"].map({"Right": 2, "Center": 4, "Left": 6}).values
            else:  # Cropped with seg
                self.sides = df["side"].map({"Right": 1, "Center": 4, "Left": 7}).values
        except KeyError:
            self.sides = np.ones(len(df)) * 4  # All Center for scs

        self.flip = flip

    def __getitem__(self, idx):
        """
        Item accessor. Loads an image and applies any necessary augmentations.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor.
            torch.Tensor: Labels as a tensor.
            int: Dummy value.
        """
        # Load
        try:
            img = np.load(self.img_paths[idx]).astype(np.float32)
            max_, min_ = img.max(), img.min()
            if max_ != min_:
                img = (img - min_) / (max_ - min_) * 255
            else:
                img = img - min_
        except Exception:
            img = np.zeros((1, 64, 64))
        img = img.astype(np.uint8)

        # Pick frame(s)
        frame = self.sides[idx] * len(img) // 8

        if self.train:
            if self.n_frames <= 3:
                frame += np.random.choice([-1, 0, 1])
            else:
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

        tgt = np.array(self.targets[idx])
        if isinstance(self.targets[idx], (int, float, np.int64, np.int32)):
            y = torch.zeros(3, dtype=torch.float)
            if tgt > -1:
                y[tgt] = 1
        elif len(tgt.shape) == 2:  # PL - no need to one-hot encode
            y = torch.from_numpy(tgt.astype(np.float32))
        else:
            y = torch.zeros((len(tgt), 3), dtype=torch.float)
            for i in range(len(tgt)):
                if tgt[i] > -1:
                    y[i, tgt[i]] = 1

        # Reshape
        if self.frames_chanel:
            image = image.view(self.n_frames, 3, image.size(1), image.size(2))
        else:
            image = image.unsqueeze(1).repeat(1, 3, 1, 1)
        if self.n_frames == 1:
            image = image[0]

        if np.random.random() < 0.5 and self.flip:
            if y.size(0) == 5:
                y = y[[0, 2, 1, 4, 3]].contiguous()
            image = torch.flip(image, [0])

        return image, y, 0


class CoordsDataset(Dataset):
    """
    Dataset for training coords models.
    """
    def __init__(
        self,
        df,
        targets="target",
        transforms=None,
        train=False,
        **kwargs,
    ):
        """
        Constructor for the CoordsDataset class.

        Args:
            df (pandas DataFrame): Metadata containing the information.
            targets (str, optional): Column name for the target values. Defaults to "target".
            transforms (callable, optional): Transforms to apply. Defaults to None.
            train (bool, optional): Whether the dataset is for training. Defaults to False.
        """
        self.df = df
        self.targets = np.array(df[targets].values.tolist())
        self.img_paths = df["img_path"].values

        if "target_rel" in df.columns:
            self.targets_rel = np.array(df["target_rel"].values.tolist())
        else:
            self.targets_rel = np.zeros(len(df))
        self.transforms = transforms

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Item accessor. Retrieves an image and its corresponding target coordinates.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor.
            torch.Tensor: Target coordinates as a tensor.
            int: Dummy value.
        """
        image = cv2.imread(self.img_paths[idx]).astype(np.float32) / 255.0

        # Augment
        if self.transforms:
            y = self.targets[idx].copy()
            transformed = self.transforms(image=image, keypoints=y[y.sum(-1) > 0].copy())
            image = transformed["image"]

        y = torch.tensor(y).float()
        y[y.sum(-1) > 0] = torch.tensor(transformed["keypoints"]).float()
        y[:, 0] /= image.size(2)
        y[:, 1] /= image.size(1)
        y = torch.where(y < 0, -1, y)
        y = torch.where(y > 1, -1, y)

        return image, y, 0


class FeatureDataset(Dataset):
    """
    Dataset for training level 2 models.
    """

    def __init__(
        self,
        df,
        exp_folders,
        targets="target",
        resize=None,
    ):
        """
        Constructor for the FeatureDataset class.

        Args:
            df (pandas DataFrame): Metadata containing image paths and targets.
            exp_folders (dict): Dictionary mapping experiment names to folder paths.
            targets (str, optional): Column name for the target values. Defaults to "target".
            resize (tuple, optional): Dimensions to resize the images to. Defaults to None.
        """
        self.df = df
        self.targets = df[targets].values

        self.resize = resize
        self.exp_folders = exp_folders

        self.series_dict = self.get_series_dict(df)

        self.dummies = {
            "scs_crop": np.zeros(3),
            "nfn_crop": np.zeros(3),
            "ss_crop_": np.zeros((2, 3)),
            "crop": np.zeros((5, 3)),
            "crop_bi": np.zeros((5, 3)),
            "crop_2": np.zeros((5, 3)),
            "crop_3": np.zeros((5, 3)),
            "crop_4": np.zeros((5, 3)),
            "dh": np.zeros((25, 3)),
            "ch": np.zeros((25, 3)),
            "spinenet": np.zeros((12)),
        }

        self.fts = {}
        for k in self.exp_folders:
            if "crop" in k:
                self.fts[k] = self.load_fts(self.exp_folders[k])
            elif ("dh" in k) or ("ch" in k):
                file = torch.load(self.exp_folders[k])
                self.fts[k] = dict(zip(
                    file["study_id"].tolist(),
                    file['logits'].float().cpu().numpy(),
                ))
            elif "spinenet" in k:
                df = pd.read_csv(self.exp_folders[k]).set_index("series_id")
                for level in LEVELS_:
                    df[level] = df[level].fillna('()').apply(eval)
                self.fts[k] = df

    @staticmethod
    def get_series_dict(df):
        """
        Constructs a dictionary mapping study IDs to their series descriptions and IDs.

        Args:
            df (pandas DataFrame): DataFrame containing series information.

        Returns:
            dict: Dictionary mapping study IDs to series information.
        """
        series_dict = defaultdict(dict)
        df = df[['series_id', 'series_description', "study_id"]]
        for study, df_study in df.explode(['series_id', 'series_description']).groupby("study_id"):
            series = df_study[
                ["series_id", "series_description"]
            ].groupby("series_description").agg(list)
            series = series['series_id'].to_dict()

            series_dict[study]["scs"] = series.get("Sagittal T2/STIR", [])
            series_dict[study]["nfn"] = series.get("Sagittal T1", [])
            series_dict[study]["ss"] = series.get("Axial T2", [])
        return series_dict

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.df)

    @staticmethod
    def load_fts(exp_folder):
        """
        Loads feature data from the specified experiment folder.

        Args:
            exp_folder (str): Path to the experiment folder.

        Returns:
            dict: Dictionary mapping index keys to feature data.
        """
        fts = {}
        for fold in range(4):
            preds = np.load(exp_folder + f'pred_inf_{fold}.npy')
            df = pd.read_csv(exp_folder + f'df_val_{fold}.csv')
            if "side" not in df.columns:
                df['side'] = 'Center'
            df = df[["study_id", "series_id", "level", "side"]].astype(str)
            index = ["_".join(row.tolist()) for row in df.values]
            fts.update(dict(zip(index, preds)))
        return fts

    def __getitem__(self, idx):
        """
        Retrieves the features and target for a specific index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing features, target, and a dummy value.
        """
        study = self.df["study_id"][idx]
        series = self.series_dict[study]

        fts = {}
        for exp in self.exp_folders:
            series_k = exp.split("_")[0]
            series_k = [series_k] if series_k in ["ss", "nfn", "scs"] else ["nfn", "scs"]  # "ss"

            if "crop" in exp:
                sides = ["Left", "Right"] if "nfn" in exp or "ss" in exp else ['Center']
                ft = defaultdict(list)
                for lvl in LEVELS:
                    for side in sides:
                        for sk in series_k:
                            ft_ = []
                            for s in series[sk]:
                                ft_k = f'{study}_{s}_{lvl}_{side}'
                                try:
                                    ft_.append(self.fts[exp][ft_k])
                                except KeyError:
                                    # print(exp, sk, ft_k, "missing")
                                    pass
                            ft_ = np.mean(ft_, 0) if len(ft_) else self.dummies[exp[:8]]
                            ft[f"{lvl}_{side}"].append(ft_)
                ft_ = []
                for k in ft.keys():
                    try:
                        ft_.append(np.concatenate(ft[k], -1))
                    except ValueError:
                        print([x.shape for x in ft[k]])
                ft = np.vstack(ft_)

                # Put in the right order
                ft = ft.reshape(5, -1, ft.shape[-1]).transpose(1, 0, 2).reshape(-1, ft.shape[-1])

            elif "dh" in exp or "ch" in exp:
                ft = self.fts[exp].get(study, self.dummies[exp[:2]])

            else:
                raise NotImplementedError

            fts[exp] = torch.from_numpy(ft).float().contiguous()

        y = torch.from_numpy(self.targets[idx])

        return fts, y, 0
