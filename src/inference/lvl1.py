import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import model_zoo.dsnt as dsnt

from data.transforms import get_transfos
from torch.utils.data import DataLoader

from inference.dataset import ImageInfDataset
from data.dataset import CropDataset, CropSagAxDataset
from model_zoo.models import define_model
from model_zoo.models_bi import define_model_bi

from util.torch import load_model_weights
from util.metrics import disk_auc
from params import LEVELS_


def predict(
    model,
    dataset,
    loss_config,
    batch_size=64,
    device="cuda",
    use_fp16=False,
    num_workers=8,
):
    """
    Perform inference using a single model and generate predictions for the given dataset.

    Args:
        model (torch.nn.Module): Trained model for inference.
        dataset (torch.utils.data.Dataset): Dataset for which to generate predictions.
        loss_config (dict): Configuration for loss function and activation.
        batch_size (int, optional): Batch size for prediction. Defaults to 64.
        device (str, optional): Device for inference, 'cuda' or 'cpu'. Defaults to 'cuda'.
        use_fp16 (bool, optional): Whether to use FP16 inference. Defaults to False.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 8.

    Returns:
        np array [N x C]: Predicted probabilities for each class for each sample.
        list: Empty list, placeholder for the auxiliary task.
    """
    model.eval()
    preds, preds_aux = [], []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for x, _, _ in loader:
            with torch.cuda.amp.autocast(enabled=use_fp16):
                x = {k: x[k].cuda() for k in x} if isinstance(x, dict) else x.cuda()
                y_pred, y_pred_aux = model(x)

            # Get probabilities
            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "series":
                y_pred = y_pred.view(y_pred.size(0), -1, 3)
            elif loss_config["activation"] == "study":
                y_pred = y_pred.view(y_pred.size(0), -1, 3)  # .softmax(-1)
            elif loss_config["activation"] == "dsnt":
                y_pred = dsnt.flat_softmax(y_pred)
                y_pred = (dsnt.dsnt(y_pred) + 1) / 2  # coords in [0, 1]
            # elif loss_config["activation"] == "softmax":
            #     y_pred = y_pred.softmax(-1)
            # else:
            #     pass

            preds.append(y_pred.detach().cpu().numpy())
            preds_aux.append(y_pred_aux.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(preds_aux)


class Config:
    """
    Placeholder to load a config from a saved json
    """
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def kfold_inference(
    df,
    exp_folder,
    debug=False,
    use_fp16=False,
    save=False,
    num_workers=8,
    batch_size=None,
    config=None,
    distributed=False,
    left_right=False,
    use_aux=False,
):
    """
    Perform k-fold inference on a dataset using a trained model.

    Args:
        df_patient (pd.DataFrame): Dataframe containing patient information.
        df_img (pd.DataFrame): Dataframe containing image information.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): If True, enable debug mode. Defaults to False.
        use_fp16 (bool, optional): Whether to use fp16 inference. Defaults to False.
        save (bool, optional): If True, save inference results. Defaults to False.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        batch_size (int, optional): Batch size for inference. Defaults to None.
        config (Config, optional): Configuration object for the experiment. Defaults to None.
    """
    if config is None:
        config = Config(json.load(open(exp_folder + "config.json", "r")))

    if "fold" not in df.columns:
        folds = pd.read_csv(config.folds_file)
        df = df.merge(folds, how="left")

    if save:
        save_folder = exp_folder + "preds/"
        os.makedirs(save_folder, exist_ok=True)
        if use_aux:
            save_folder_aux = exp_folder + "preds_aux/"
            os.makedirs(save_folder_aux, exist_ok=True)

    if debug:
        config.selected_folds = [0]
        df = df.head(100)

    if distributed:
        config.selected_folds = [config.local_rank]

    for fold in config.selected_folds:
        if config.local_rank == 0 and not distributed:
            print(f"\n- Fold {fold + 1}")

        model_fct = define_model_bi if "bi" in config.pipe else define_model
        model = model_fct(
            config.name,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            pooling=config.pooling if hasattr(config, "pooling") else "avg",
            head_3d=config.head_3d,
            n_frames=config.n_frames,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            n_channels=config.n_channels,
            reduce_stride=config.reduce_stride,
            pretrained=False,
        )
        model = model.cuda().eval()

        weights = exp_folder + f"{config.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=config.local_rank == 0)

        df_val = df[df["fold"] == fold].reset_index(drop=True)

        transforms = get_transfos(
            augment=False,
            resize=config.resize,
            crop=config.crop,
        )

        preds = []
        for img_path in tqdm(df_val['img_path'].values, disable=config.local_rank != 0):

            imgs = np.load(img_path)
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min()) * 255
            imgs = imgs.astype(np.uint8)

            dataset = ImageInfDataset(
                imgs,
                transforms=transforms,
                frames_chanel=config.frames_chanel if hasattr(config, "frames_chanel") else 0,
                n_frames=config.n_frames if hasattr(config, "n_frames") else 1,
                stride=config.stride if hasattr(config, "stride") else 1,
            )

            pred, pred_aux = predict(
                model,
                dataset,
                config.loss_config,
                batch_size=config.data_config["val_bs"] if batch_size is None else batch_size,
                use_fp16=use_fp16,
                num_workers=num_workers,
            )

            if left_right:
                preds.append([pred[len(pred) // 2:].max(0), pred[:len(pred) // 2].max(0)])
            elif use_aux:
                preds.append([
                    (pred * pred_aux[:, c][:, None, None]).max(0)
                    for c in range(config.num_classes_aux)
                ])
            else:
                preds.append(pred.max(0))
            if save:
                np.save(save_folder + img_path.split('/')[-1], pred)
                if use_aux:
                    np.save(save_folder_aux + img_path.split('/')[-1], pred_aux)

        preds = np.array(preds)
        # return preds

        aucs = []
        if left_right:
            for s, side in enumerate(["left", "right"]):
                for i, c in enumerate(config.targets):
                    auc = disk_auc(df_val[side + "_" + c].values, preds[:, s, i])
                    aucs.append(auc)

        elif use_aux:
            for i, c in enumerate(config.targets):
                for lv, lvl in enumerate(LEVELS_):
                    try:
                        auc = disk_auc(df_val[c + "_" + lvl].values, preds[:, lv, i])
                        aucs.append(auc)
                    except Exception:
                        aucs.append(0)
        else:
            for i, c in enumerate(config.targets):
                auc = disk_auc(df_val[c].values, preds[:, i])
                aucs.append(auc)
        print(f'\n -> Fold {fold} - Average AUC: {np.mean(aucs):.4f}')


def kfold_inference_crop(
    df,
    exp_folder,
    debug=False,
    use_fp16=False,
    save=False,
    num_workers=8,
    batch_size=None,
    config=None,
    distributed=False,
):
    """
    Perform k-fold inference on a dataset using a trained model.

    Args:
        df_patient (pd.DataFrame): Dataframe containing patient information.
        df_img (pd.DataFrame): Dataframe containing image information.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): If True, enable debug mode. Defaults to False.
        use_fp16 (bool, optional): Whether to use fp16 inference. Defaults to False.
        save (bool, optional): If True, save inference results. Defaults to False.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        batch_size (int, optional): Batch size for inference. Defaults to None.
        config (Config, optional): Configuration object for the experiment. Defaults to None.
    """
    if config is None:
        config = Config(json.load(open(exp_folder + "config.json", "r")))

    if "fold" not in df.columns:
        folds = pd.read_csv(config.folds_file)
        df = df.merge(folds, how="left")

    if debug:
        config.selected_folds = [0]
        # df = df.head(100)

    if distributed:
        config.selected_folds = [config.local_rank]

    for fold in config.selected_folds:
        if config.local_rank == 0 and not distributed:
            print(f"\n- Fold {fold + 1}")

        model_fct = define_model_bi if "bi" in config.pipe else define_model
        model = model_fct(
            config.name,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            pooling=config.pooling,
            head_3d=config.head_3d,
            n_frames=config.n_frames,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            n_channels=config.n_channels,
            reduce_stride=config.reduce_stride,
            pretrained=False,
        )
        model = model.cuda().eval()

        fold_ = 'fullfit_0'
        weights = exp_folder + f"{config.name}_{fold_}.pt"
        model = load_model_weights(model, weights, verbose=config.local_rank == 0)

        df_val = df[df["fold"] == fold].reset_index(drop=True)

        transforms = get_transfos(
            augment=False,
            resize=config.resize,
            crop=config.crop,
        )

        dataset_class = CropSagAxDataset if "bi" in config.pipe else CropDataset
        dataset = dataset_class(
            df_val,
            targets=config.targets,
            transforms=transforms,
            frames_chanel=config.frames_chanel if hasattr(config, "frames_chanel") else 0,
            n_frames=config.n_frames if hasattr(config, "n_frames") else 1,
            stride=config.stride if hasattr(config, "stride") else 1,
            use_coords_crop=config.use_coords_crop,
            train=False,
            load_in_ram=False,
        )

        preds, preds_aux = predict(
            model,
            dataset,
            config.loss_config,
            batch_size=config.data_config["val_bs"] if batch_size is None else batch_size,
            use_fp16=use_fp16,
            num_workers=num_workers,
        )

        if save:
            np.save(exp_folder + f"pred_inf_{fold}.npy", preds)
            np.save(exp_folder + f"pred_inf_aux_{fold}.npy", preds)

        # preds = np.array(preds)

        if isinstance(df_val["target"].values[0], list):
            y = np.vstack(df_val["target"].values)
            auc = np.mean([
                disk_auc(y[:, i], preds[:, i]) for i in range(y.shape[1])
            ])
        else:
            auc = disk_auc(df_val["target"].values, preds)

        print(f'\n -> Fold {fold + 1} - Average AUC: {auc:.4f}')
