#!/usr/bin/env python3
"""
major actions here for training VTAB datasets: use val200 to find best lr/wd, and retrain on train800val200, report results on test
"""
import glob
import numpy as np
import os
import torch
import warnings
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
from collections import defaultdict
# torch.serialization.add_safe_globals([defaultdict, dict])
warnings.filterwarnings("ignore")

def setup(args, lr, wd, k, lamb, final_runs, run_idx=None, seed=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.SEED = seed

    cfg.RUN_N_TIMES = 1
    cfg.MODEL.SAVE_CKPT = False
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_val"
    lr = lr / 256 * cfg.DATA.BATCH_SIZE  # update lr based on the batchsize
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.WEIGHT_DECAY = wd
    cfg.MODEL.PROMPT.K = k
    cfg.MODEL.PROMPT.LAMB = lamb

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}_k{k}_lamb{lamb}"
    )

    # train cfg.RUN_N_TIMES times
    if run_idx is None:
        count = 1
        while count <= cfg.RUN_N_TIMES:
            output_path = os.path.join(output_dir, output_folder, f"run{count}")
            # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
            sleep(randint(1, 5))
            if not PathManager.exists(output_path):
                PathManager.mkdirs(output_path)
                cfg.OUTPUT_DIR = output_path
                break
            else:
                count += 1
        if count > cfg.RUN_N_TIMES:
            raise ValueError(
                f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")
    else:
        output_path = os.path.join(output_dir, output_folder, f"run{run_idx}")
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
        else:
            raise ValueError(
                f"Already run run-{run_idx} for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger, final_runs=False):
    # support two training paradims:
    # 1) train / val / test, using val to tune
    # 2) train / val: for imagenet

    logger.info("Loading training data...")
    train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    val_loader = data_loader.construct_val_loader(cfg)
    # not really nessecary to check the results of test set.
    test_loader = None

    return train_loader, val_loader, test_loader


def train(cfg, args, final_runs):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(
        cfg, logger, final_runs)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
        # save the evaluation results
        torch.save(
            evaluator.results,
            os.path.join(cfg.OUTPUT_DIR, "eval_results.pth")
        )
    else:
        print("No train loader presented. Exit")


def main(args):
    """main function to call from workflow"""
    # tuning lr and wd first:
    try:
        cfg = setup(args, args.lrr, args.wdd, args.kk, args.lambdaa, final_runs=False)
    except ValueError:
        # already ran
        return
    train(cfg, args, final_runs=False)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)