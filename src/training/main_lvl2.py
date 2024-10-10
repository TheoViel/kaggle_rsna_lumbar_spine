import gc
import torch
import numpy as np
import pandas as pd

from training.train import fit
from model_zoo.models_lvl2 import define_model

from data.dataset import FeatureDataset
from util.metrics import rsna_loss
from util.torch import seed_everything, count_parameters, save_model_weights
from params import NOISY_STUDIES


def train(
    config, df_train, df_val, fold, log_folder=None, run=None
):
    """
    Train a level 2 model.

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
    train_dataset = FeatureDataset(
        df_train,
        config.exp_folders,
        targets=config.targets,
        resize=config.resize,
    )

    val_dataset = FeatureDataset(
        df_val,
        config.exp_folders,
        targets=config.targets,
        resize=config.resize,
    )

    model = define_model(
        config.name,
        ft_dim=config.ft_dim,
        layer_dim=config.layer_dim,
        dense_dim=config.dense_dim,
        resize=config.resize,
        p=config.p,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_fts=config.n_fts,
    ).cuda()

    model.zero_grad(set_to_none=True)
    model.train()

    n_parameters = count_parameters(model)
    if config.local_rank == 0:
        print(f"    -> {len(train_dataset)} training studies")
        print(f"    -> {len(val_dataset)} validation studies")
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

    if log_folder is not None:
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
    Perform k-fold cross-validation training for a level 2 model.

    Args:
        config (dict): Configuration parameters for training.
        df (pandas DataFrame): Main dataset metadata.
        df_img (pandas DataFrame): Metadata containing image information.
        log_folder (str, optional): Folder for saving logs. Defaults to None.
        run: Neptune run. Defaults to None.
    """
    if "fold" not in df.columns:
        folds = pd.read_csv(config.folds_file)
        df = df.merge(folds, how="left")

    pred_oof = np.zeros((len(df), 25, 3))
    for fold in range(config.k):
        if fold in config.selected_folds:
            print(
                f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n"
            )
            seed_everything(config.seed + fold)

            # preds = np.load('../logs/2024-10-07/0/pred_oof.npy')
            # t = torch.from_numpy(df[config.targets].values).contiguous().long().cuda()
            # p = torch.from_numpy(preds).contiguous().float().softmax(-1).cuda()

            # from tqdm import tqdm
            # for _ in tqdm(range(500000)):
            #     ids = np.random.choice(np.arange(len(df)), 400, replace=False)
            #     avg_loss, losses = rsna_loss(t[ids], torch.ones_like(p)[ids] / 3, verbose=0)
            
            #     if avg_loss >= 1.02:
            #         break

            # if avg_loss < 1.02:
            #     continue

            # df_train = df[df.index.isin(ids)].reset_index(drop=True)
            # df_val = df[df.index.isin(ids)].reset_index(drop=True)

            df_train = df[df["fold"] != fold].reset_index(drop=True)
            df_val = df[df["fold"] == fold].reset_index(drop=True)

            if hasattr(config, "remove_noisy"):
                if config.remove_noisy:
                    df_train = df_train[~df_train['study_id'].isin(NOISY_STUDIES)]
                    df_train = df_train.reset_index(drop=True)

            preds, metrics = train(
                config,
                df_train,
                df_val,
                fold,
                log_folder=log_folder,
                run=run,
            )

            pred_oof[df[df["fold"] == fold].index.values] = preds
            # if log_folder is None:
            #     return preds, preds_aux

            if log_folder is not None:
                np.save(log_folder + f"pred_val_{fold}", preds)
                df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)

    if len(config.selected_folds) == config.k:
        if config.fullfit:
            for ff in range(config.n_fullfit):
                if config.local_rank == 0:
                    print(
                        f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                    )
                seed_everything(config.seed + ff)
                train(
                    config,
                    df,
                    df,
                    f"fullfit_{ff}",
                    log_folder=log_folder,
                    run=run,
                )

        avg_loss, losses = rsna_loss(df[config.targets].values, pred_oof)
        print()
        for k, v in losses.items():
            print(f"- {k}_loss\t: {v:.3f}")
        print(f"\n -> CV Score : {avg_loss :.4f}")

        if log_folder is not None:
            np.save(log_folder + "pred_oof.npy", pred_oof)            

        if run is not None:
            run["global/logs"].upload(log_folder + "logs.txt")
            run["global/cv"] = avg_loss
            for k, v in losses.items():
                run[f"global/{k}"] = v

            print()
            run.stop()

    return pred_oof
