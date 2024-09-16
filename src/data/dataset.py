import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

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
        self.coords = df["coords"].values if "coords" in df.columns else None

        self.load_in_ram = load_in_ram
        self.load_imgs_in_ram()

    def load_imgs_in_ram(self):
        self.imgs = {}
        if not self.load_in_ram:
            return

        for path in np.unique(self.img_paths):
            try:
                img = np.load(path).astype(np.float32)
                # img = np.clip(
                #     img, np.percentile(img.flatten(), 0), np.percentile(img.flatten(), 98)
                # )
                img = (img - img.min()) / (img.max() - img.min()) * 255
            except FileNotFoundError:
                img = np.zeros((1, 64, 64))

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

        # image = np.clip(
        #     image, np.percentile(image.flatten(), 0), np.percentile(image.flatten(), 98)
        # )

        image = (image - image.min()) / (image.max() - image.min())

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
        if isinstance(self.targets[0], list):
            self.targets = np.vstack(self.targets)

        try:
            if "coords_crops" in df["img_path"][0]:
                self.sides = df["side"].map({"Right": 2, "Center": 4, "Left": 6}).values
            else:  # Cropped with seg
                self.sides = df["side"].map({"Right": 1, "Center": 4, "Left": 7}).values
        except KeyError:
            self.sides = np.ones(len(df)) * 4  # All Center for scs

        self.use_coords_crop = use_coords_crop
        if self.use_coords_crop:
            col = "coords_crop" if "coords_crop" in df.columns else "coords"
            self.coords_crop = np.array(df[col].values.tolist()).astype(int)

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
        # print(frame, self.coords_crop[idx][0])
        # print(frame, "naive")

        if self.use_coords_crop:
            if self.coords_crop[idx].max() > 0:
                gt_frame = self.coords_crop[idx][0]
                if np.abs(frame - gt_frame) <= 3:  # Noisy
                    frame = gt_frame

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

        # if self.sides[idx] != 4 and self.train and np.random.random() < 0.5:
        #     frames = frames[::-1]  # flip aug

        image = img[np.array(frames)].transpose(1, 2, 0)
        image = image.astype(np.float32) / 255.0

        # Augment
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

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
            image = image[0]

        # print(frames, len(img))

        if np.random.random() < 0.5 and self.flip:
            # print('flip')
            y = torch.flip(y, [0])
            image = torch.flip(image, [1])

        return image, y, 0


class CropSagAxDataset(CropDataset):
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
            use_coords_crop=use_coords_crop,
            load_in_ram=load_in_ram,
            train=train,
        )
        self.img_paths_ax = df['img_path_ax'].values

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
        sag_image, y, _ = super().__getitem__(idx)

        level = self.img_paths_ax[idx][-9:-4]
        level = LEVELS_.index(level)

        try:
            img = np.load(self.img_paths_ax[idx]).astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255
        except FileNotFoundError:
            img = np.zeros((1, 64, 64))
        img = img.astype(np.uint8)

        w = img.shape[2]
        if self.sides[idx] < 4:  # Right
            img = img[:, :, :w // 2]
        elif self.sides[idx] > 4:
            img = img[:, :, w // 2:]

        frame = len(img) // 2

        if self.train:
            frame += np.random.choice([-1, 0, 0, 1])
            # frame += np.random.choice([-2, -1, 0, 1, 2])

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
        # Reshape
        if self.frames_chanel:
            image = image.view(self.n_frames, 3, image.size(1), image.size(2))
        else:
            image = image.unsqueeze(1).repeat(1, 3, 1, 1)
        if self.n_frames == 1:
            image = image[0]

        # return image, y, 0
        return {"sag": sag_image, "ax": image, "level": level}, y, 0


class CoordsDataset(Dataset):
    """
    Dataset for training 2.5D crop classification models.
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
        Item accessor. Samples a random frame inside the organ.

        Args:
            idx (int): Index.

        Returns:
            torch.Tensor: Image as a tensor of shape [(N,) C, H, W].
            torch.Tensor: Label as a tensor of shape [3].
            int: Dummy value.
        """
        image = cv2.imread(self.img_paths[idx]).astype(np.float32) / 255.0

        # Augment
        if self.transforms:
            y = self.targets[idx].copy()
            # print(y)
            transformed = self.transforms(image=image, keypoints=y[y.sum(-1) > 0].copy())
            image = transformed["image"]

        y = torch.tensor(y).float()
        y[y.sum(-1) > 0] = torch.tensor(transformed["keypoints"]).float()
        y[:, 0] /= image.size(2)
        y[:, 1] /= image.size(1)
        y = torch.where(y < 0, -1, y)
        y = torch.where(y > 1, -1, y)

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
    Dataset for training level 2 models.
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

        self.dummies = {
            "scs_crop": np.zeros(3),
            "nfn_crop": np.zeros(3),
            "ss_crop_": np.zeros((2, 3)),
            "crop": np.zeros((5, 3)),
<<<<<<< HEAD
<<<<<<< HEAD
            "crop_bi": np.zeros((5, 3)),
=======
>>>>>>> 7c2b817 (a100)
=======
>>>>>>> 7c2b817be4291a757ac90fb3d10bb0387f572ecc
            "crop_2": np.zeros((5, 3)),
            "dh": np.zeros((25, 3)),
            "ch": np.zeros((25, 3)),
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

    @staticmethod
    def get_series_dict(df):
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

            # if series_dict[study]["nfn"] is None:
            #     series_dict[study]["nfn"] = series_dict[study].get("scs", [])
            # if series_dict[study]["scs"] is None:
            #     series_dict[study]["scs"] = series_dict[study].get("nfn", [])
        return series_dict

    def __len__(self):
        return len(self.df)

    @staticmethod
    def load_fts(exp_folder):
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
                # if not ("scs" in k or "nfn" in k or "ss" in k):
                #     ft[5:15] = np.concatenate([ft[10:15], ft[5:10]], 0)  # Right then left
                #     ft[15:25] = np.concatenate([ft[20:25], ft[15:20]], 0)  # Right then left

            elif "dh" in exp or "ch" in exp:
                ft = self.fts[exp].get(study, self.dummies[exp[:2]])

            fts[exp] = torch.from_numpy(ft).float().contiguous()

        y = torch.from_numpy(self.targets[idx])

        return fts, y, 0
