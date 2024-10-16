import os
import time
import torch
import warnings
import argparse

from data.preparation import prepare_data_crop
from util.torch import init_distributed
from util.logger import (
    create_logger,
    save_config,
    prepare_log_folder,
    init_neptune,
    get_last_log_folder,
)

from params import DATA_PATH


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Fold number",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device number",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="Folder to log results to",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        help="learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay",
    )
    return parser.parse_args()


class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1

    pipe = "crop"
    targets = "target"

    # Data
    crop_folder = "../input/coords_crops_0.1_2/"

    resize = (224, 224)
    frames_chanel = 1
    n_frames = 13
    stride = 1
    aug_strength = 5
    crop = False

    fix_train_crops = True
    flip = False

    # k-fold
    k = 4
    # folds_file = f"../input/folds_{k}.csv"
    folds_file = "../input/train_folded_v1.csv"
    selected_folds = [0, 1, 2, 3]

    # Model
    name = "coatnet_1_rw_224"
    pretrained_weights = None  # "../logs/2024-09-19/17/"

    num_classes = 15
    num_classes_aux = 0
    drop_rate = 0.
    drop_path_rate = 0.
    n_channels = 3
    reduce_stride = False
    pooling = "avg"
    head_3d = "lstm_side" if n_frames > 1 else ""
    delta = 2

    # Training
    loss_config = {
        "name": "series",
        "weighted": False,
        "use_any": False,
        "smoothing": 0.0,
        "activation": "series",
        "aux_loss_weight": 0.0,
        "name_aux": "patient",
        "smoothing_aux": 0.0,
        "activation_aux": "",
        "ousm_k": 0,
    }

    data_config = {
        "batch_size": 16,  # 8
        "val_bs": 32,
        "mix": "mixup",
        "mix_proba": 1.0,  # 1.0
        "sched": False,
        "mix_alpha": 0.4,
        "additive_mix": False,
        "num_classes": 3,
        "num_workers": 8,
    }

    optimizer_config = {
        "name": "Ranger",
        "lr": 1e-3,
        "warmup_prop": 0.0,
        "betas": (0.9, 0.999),
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
    }

    epochs = 10

    use_fp16 = True
    verbose = 1
    verbose_eval = 50 if data_config["batch_size"] >= 16 else 100

    fullfit = True
    n_fullfit = 1


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)

    config = Config
    init_distributed(config)

    if config.local_rank == 0:
        print("\nStarting !")
    args = parse_args()

    if not config.distributed:
        device = args.fold if args.fold > -1 else args.device
        time.sleep(device)
        print("Using GPU ", device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        assert torch.cuda.device_count() == 1

    log_folder = args.log_folder
    if not log_folder:
        from params import LOG_PATH

        if config.local_rank == 0:
            log_folder = prepare_log_folder(LOG_PATH)
            print(f"\n -> Logging results to {log_folder}\n")
        else:
            time.sleep(2)
            log_folder = get_last_log_folder(LOG_PATH)
    #             print(log_folder)

    if args.model:
        config.name = args.model

    if args.epochs:
        config.epochs = args.epochs

    if args.lr:
        config.optimizer_config["lr"] = args.lr

    if args.weight_decay:
        config.optimizer_config["weight_decay"] = args.weight_decay

    if args.batch_size:
        config.data_config["batch_size"] = args.batch_size
        config.data_config["val_bs"] = args.batch_size

    run = None
    if config.local_rank == 0:
        run = init_neptune(config, log_folder)

        if args.fold > -1:
            config.selected_folds = [args.fold]
            create_logger(directory=log_folder, name=f"logs_{args.fold}.txt")
        else:
            create_logger(directory=log_folder, name="logs.txt")

        save_config(config, log_folder + "config.json")
        if run is not None:
            run["global/config"].upload(log_folder + "config.json")

    if config.local_rank == 0:
        print("Device :", torch.cuda.get_device_name(0), "\n")

        print(f"- Model {config.name}")
        print(f"- Epochs {config.epochs}")
        print(
            f"- Learning rate {config.optimizer_config['lr']:.1e}   (n_gpus={config.world_size})"
        )
        print("\n -> Training\n")

    df = prepare_data_crop(DATA_PATH, crop_folder=config.crop_folder)

    from training.main import k_fold
    k_fold(config, df, log_folder=log_folder, run=run)

    if len(config.selected_folds) == 4:
        if config.local_rank == 0:
            print("\n -> Inference\n")

        # log_folder = "../logs/2024-10-04/9/"

        from inference.lvl1 import kfold_inference_crop
        kfold_inference_crop(
            df,
            log_folder,
            use_fp16=config.use_fp16,
            save=True,
            distributed=True,
            config=config,
        )

    if config.local_rank == 0:
        print("\nDone !")
