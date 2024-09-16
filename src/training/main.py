import gc
import glob
import torch
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from training.train import fit
from model_zoo.models import define_model
from model_zoo.models_bi import define_model_bi

from data.dataset import CropDataset, ImageDataset, CoordsDataset, CropSagAxDataset
from data.transforms import get_transfos

from util.torch import seed_everything, count_parameters, save_model_weights
from params import NOISY_SERIES


def train(config, df_train, df_val, fold, log_folder=None, run=None):
    """
    Train a crop model.

    Args:
        config (Config): Configuration parameters for training.
        df_train (pandas DataFrame): Metadata for training dataset.
        df_val (pandas DataFrame): Metadata for validation dataset.
        fold (int): Fold number for cross-validation.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.

    Returns:
        tuple: A tuple containing predictions and metrics.
    """
    if "crop" in config.pipe:
        if "bi" in config.pipe:
            dataset_class = CropSagAxDataset
        else:
            dataset_class = CropDataset
    elif "coord" in config.pipe:
        dataset_class = CoordsDataset
    else:
        dataset_class = ImageDataset

    transfos = get_transfos(
        strength=config.aug_strength,
        resize=config.resize,
        crop=config.crop,
        use_keypoints="coords" in config.pipe,
    )
    train_dataset = dataset_class(
        df_train,
        targets=config.targets,
        transforms=transfos,
        frames_chanel=config.frames_chanel,
        n_frames=config.n_frames,
        stride=config.stride,
        use_coords_crop=config.use_coords_crop,
        train=True,
        load_in_ram=config.load_in_ram if hasattr(config, "load_in_ram") else False,
    )

    transfos = get_transfos(
        augment=False,
        resize=config.resize,
        crop=config.crop,
        use_keypoints="coords" in config.pipe,
    )
    val_dataset = dataset_class(
        df_val,
        targets=config.targets,
        transforms=transfos,
        frames_chanel=config.frames_chanel,
        n_frames=config.n_frames,
        stride=config.stride,
        use_coords_crop=config.use_coords_crop,
        train=False,
        load_in_ram=config.load_in_ram if hasattr(config, "load_in_ram") else False,
    )

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(
            ".pt"
        ) or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[
                0
            ]
    else:
        pretrained_weights = None

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
        pretrained_weights=pretrained_weights,
        reduce_stride=config.reduce_stride,
        verbose=(config.local_rank == 0),
    ).cuda()

    if config.distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    model.zero_grad(set_to_none=True)
    model.train()

    n_parameters = count_parameters(model)
    if config.local_rank == 0:
        print(f"    -> {len(train_dataset)} training injuries")
        print(f"    -> {len(val_dataset)} validation injuries")
        print(f"    -> {n_parameters} trainable parameters\n")

    preds, metrics = fit(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        epochs=config.epochs,
        verbose_eval=config.verbose_eval,
        use_fp16=config.use_fp16,
        distributed=config.distributed,
        local_rank=config.local_rank,
        world_size=config.world_size,
        log_folder=log_folder,
        run=run,
        fold=fold,
    )

    if (log_folder is not None) and (config.local_rank == 0):
        save_model_weights(
            model.module if config.distributed else model,
            f"{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()

    return preds, metrics


def k_fold(config, df, log_folder=None, run=None):
    """
    Perform k-fold cross-validation training for a crop model.

    Args:
        config (dict): Configuration parameters for training.
        df (pandas DataFrame): Metadata.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.
    """
    folds = pd.read_csv(config.folds_file)
    df = df.merge(folds, how="left")
    df["fold"] = df["fold"].fillna(-1)

    # from params import DATA_PATH
    # from data.preparation import prepare_data_scs
    # df2 = prepare_data_scs(DATA_PATH, crop_folder=config.crop_folder)
    # df2 = df2.merge(folds, how="left")

    all_metrics = []
    for fold in range(config.k):
        if fold in config.selected_folds:
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n"
                )
            seed_everything(config.seed + fold)

            df_train = df[df["fold"] != fold].reset_index(drop=True)
            df_val = df[df["fold"] == fold].reset_index(drop=True)

            if hasattr(config, "remove_noisy"):
                if config.remove_noisy:
                    df_train = df_train[~df_train['series_id'].isin(NOISY_SERIES)]
                    df_train = df_train.reset_index(drop=True)

            # df_train = pd.concat([df_train, df2[df2["fold"] != fold]], ignore_index=True)

            # df_train = df_val.copy()
            # if len(df) <= 1000:
            #     df_train, df_val = df, df

            if config.pipe in ["crop_nfn", "crop_scs"]:
                if config.use_coords_crop:
                    if config.local_rank == 0:
                        print('- Overriding coords with seg_sag_coords\n')

                    df_preds_coords = pd.read_csv('../output/seg_sag_coords.csv')
                    df_val = df_val.merge(df_preds_coords, how="left")
                    df_val['coords'] = df_val.apply(
                        lambda x: [{"Left": x.left, "Right": x.right, "Center": x.center}[x.side]],
                        axis=1
                    )
                    # df_train = df_train.merge(df_preds_coords, how="left")
                    # df_train['coords'] = df_train.apply(
                    #     lambda x: [{"Left": x.left, "Right": x.right, "Center": x.center}[x.side]],
                    #     axis=1
                    # )

            preds, metrics = train(
                config,
                df_train,
                df_val,
                fold,
                log_folder=log_folder,
                run=run,
            )
            all_metrics.append(metrics)

            if log_folder is None:
                return

            if config.local_rank == 0:
                np.save(log_folder + f"pred_val_{fold}", preds)
                df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)

    if config.local_rank == 0 and len(config.selected_folds):
        print("\n-------------   CV Scores  -------------\n")

        for k in all_metrics[0].keys():
            avg = np.mean([m[k] for m in all_metrics])
            print(f"- {k.split('_')[0][:7]} score\t: {avg:.3f}")
            if run is not None:
                run[f"global/{k}"] = avg

        if run is not None:
            run["global/logs"].upload(log_folder + "logs.txt")

        np.save(log_folder + f"pred_val_{fold}", preds)
        df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)

    if config.fullfit and config.selected_folds != [0]:
        for ff in range(config.n_fullfit):
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                )
            seed_everything(config.seed + ff)

            train(
                config,
                df,
                df[df["fold"] == 0].reset_index(drop=True),
                f"fullfit_{ff}",
                log_folder=log_folder,
                run=run,
            )

    if run is not None:
        print()
        run.stop()
