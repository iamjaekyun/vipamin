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
# from collections import defaultdict
# torch.serialization.add_safe_globals([defaultdict, dict])
warnings.filterwarnings("ignore")

def find_best_lrwd(files, data_name):
    t_name = "val_" + data_name
    best_lr = None
    best_wd = None
    best_k = None
    best_lamb = None
    best_val_acc = -1
    for f in files:
        try:
            results_dict = torch.load(f, "cpu")
            epoch = len(results_dict) - 1
            val_result = results_dict[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            val_result = float(val_result)
        except Exception as e:
            print(f"Encounter issue: {e} for file {f}")
            continue

        if val_result == best_val_acc:
            frag_txt = f.split("/run")[0]
            cur_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            cur_wd = float(frag_txt.split("_wd")[-1].split("_k")[0])
            cur_k = int(frag_txt.split("_k")[-1].split("_lamb")[0])
            cur_lamb = float(frag_txt.split("_lamb")[-1])
            if best_lr is not None and cur_lr < best_lr:
                # get the smallest lr to break tie for stability
                best_lr = cur_lr
                best_wd = cur_wd
                best_k = cur_k
                best_lamb = cur_lamb
                best_val_acc = val_result

        elif val_result > best_val_acc:
            best_val_acc = val_result
            frag_txt = f.split("/run")[0]
            best_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
            best_wd = float(frag_txt.split("_wd")[-1].split("_k")[0])
            best_k = int(frag_txt.split("_k")[-1].split("_lamb")[0])
            best_lamb = float(frag_txt.split("_lamb")[-1])

    return best_lr, best_wd, best_k, best_lamb


def setup(args, lr, wd, k, lamb, final_runs, run_idx=None, seed=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.SEED = seed

    cfg.RUN_N_TIMES = 3
    cfg.MODEL.SAVE_CKPT = True
    cfg.MODEL.SAVE_CKPT_FINALRUNS = True # set here to True to enable final saving 
    
    # find the best lr and best wd
    files = glob.glob(f"{cfg.OUTPUT_DIR}_val/{cfg.DATA.NAME}/{cfg.DATA.FEATURE}/*/run1/eval_results.pth")
    lr, wd, k, lamb = find_best_lrwd(files, cfg.DATA.NAME)
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "_finalfinal"
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

    logger.info("Loading training data...")
    if cfg.DATA.NAME.startswith('vtab'):
        train_loader = data_loader.construct_trainval_loader(cfg)    
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    # not really nessecary to check the results of val set, but the trainer class does not support no-validation loader yet  # noqa
    logger.info("Loading validation data...")
    val_loader = data_loader.construct_val_loader(cfg)

    logger.info("Loading test data...")
    test_loader = data_loader.construct_test_loader(cfg)

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
    # final run 3 times with fixed seed
    random_seeds = [42, 44, 82]
    for run_idx, seed in enumerate(random_seeds):
        try:
            cfg = setup(
                args, 0.1, 0.1, 2, 0.0, final_runs=True, run_idx=run_idx+1, seed=seed)
        except ValueError:
            # already ran
            continue
        train(cfg, args, final_runs=True)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)