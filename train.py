import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import yaml
import numpy as np

from lightning_modules import PharPocketDDPM

import os
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config

    # Initial merge to determine output directory
    ckpt_path = None if args.resume is None else Path(args.resume)
    args = merge_args_and_yaml(args, config)

    out_dir = Path(args.logdir, args.run_name)
    ckpt_dir = Path(out_dir, 'checkpoints')

    # Automatically resume from last checkpoint if training stopped or disconnected
    if ckpt_path is None and ckpt_dir.exists():
        last_ckpt = ckpt_dir / 'last.ckpt'
        if last_ckpt.exists():
            print(f"==> [Auto-Resume] Found existing last checkpoint at: {last_ckpt}")
            print("==> Automatically resuming training from where it stopped!")
            ckpt_path = last_ckpt
        else:
            existing_ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
            if existing_ckpts:
                ckpt_path = existing_ckpts[-1]
                print(f"==> [Auto-Resume] Found existing checkpoint at: {ckpt_path}")
                print("==> Automatically resuming training from where it stopped!")

    # Merge saved hyperparameters if resuming
    if ckpt_path is not None:
        try:
            resume_config = torch.load(
                ckpt_path, map_location=torch.device('cpu'), weights_only=False).get('hyper_parameters', {})
            if resume_config:
                config = merge_configs(config, resume_config)
                args = merge_args_and_yaml(args, config)
        except Exception as e:
            print(f"==> [Notice] Reading hyper_parameters from checkpoint: {e}")

    # Resolve datadir if configured path doesn't exist (e.g. Drive folder name variations)
    datadir_path = Path(args.datadir)
    if not datadir_path.exists():
        possible_variations = [
            Path("/content/drive/MyDrive/DRUG DISVOVERY DATASETS/G1591 DATASET/DiffPhar_data/processed_crossdock_noH_full_temp"),
            Path("/content/drive/MyDrive/DRUG DISCOVERY DATASET/G1591 DATASET/DiffPhar_data/processed_crossdock_noH_full_temp"),
            Path("/content/drive/MyDrive/DRUG DISCOVERY DATASETS/G1591 DATASET/DiffPhar_data/processed_crossdock_noH_full_temp"),
            Path("/content/drive/MyDrive/DRUG DISVOVERY DATASETS/G1591/DiffPhar_data/processed_crossdock_noH_full_temp"),
            Path("/content/drive/MyDrive/DRUG DISCOVERY DATASET/G1591/DiffPhar_data/processed_crossdock_noH_full_temp"),
            Path("/content/drive/MyDrive/DRUG DISVOVERY DATASETS/G1591 DATASET/DiffPhar_data/processed_crossdock_noH_ca_only_temp"),
            Path("/content/drive/MyDrive/DRUG DISCOVERY DATASET/G1591 DATASET/DiffPhar_data/processed_crossdock_noH_ca_only_temp"),
        ]
        matched = next((p for p in possible_variations if p.exists()), None)
        if matched is None and Path("/content/drive/MyDrive").exists():
            search_matches = list(Path("/content/drive/MyDrive").glob("**/processed_crossdock_noH_*"))
            if search_matches:
                matched = search_matches[0]
        if matched is not None:
            print(f"==> Notice: '{args.datadir}' not found. Auto-resolved datadir to: {matched}")
            args.datadir = str(matched)

    out_dir = Path(args.logdir, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    pl_module = PharPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        model_type=args.model_type,
        oracle_class_training=getattr(args, 'oracle_class_training', False)
    )

    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project='ligand-pocket-ddpm',
        group=args.dataset,
        name=args.run_name,
        id=args.run_name,
        resume='allow' if ckpt_path is not None else False,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        mode="min",
    )

    # Automatically detect accelerator and devices
    use_gpu = torch.cuda.is_available() and args.gpus > 0
    accelerator = 'gpu' if use_gpu else 'cpu'
    devices = args.gpus if use_gpu else 1
    strategy = 'auto' if (not use_gpu or args.gpus == 1) else 'ddp'

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy
    )

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)

    # # run test set
    # result = trainer.test(ckpt_path='best')


