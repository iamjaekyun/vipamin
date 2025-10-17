#!/usr/bin/env python3
"""
major actions here for training VTAB datasets: use val200 to find best lr/wd, and retrain on train800val200, report results on test
"""
import glob
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")
def convert_np_floats_to_python_floats(data):
    """
    Recursively traverse nested dicts/lists and convert any NumPy float types 
    (e.g., np.float64) to native Python float.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_np_floats_to_python_floats(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = convert_np_floats_to_python_floats(data[i])
    elif isinstance(data, np.floating):  # catches np.float32, np.float64, etc.
        data = float(data)
    return data
def find_best_lrwd(files, data_name):
    t_name = "val_" + data_name
    best_lr = None
    best_wd = None
    best_k = None
    best_lamb = None
    best_val_acc = -1
    for f in files:
        try:
            results_dict = torch.load(f, "cpu", weights_only=False)
            epoch = len(results_dict) - 1
            val_result = results_dict[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            val_result = float(val_result)
        except Exception as e:
            print(f"Encounter issue: {e} for file {f}")
            continue

        if val_result == best_val_acc:
            frag_txt = f.split("/run")[0]
            cond_num = len(frag_txt.split('/')[-1].split('_'))
            if cond_num == 2:
                cur_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
                cur_wd = float(frag_txt.split("_wd")[-1])
                cur_k = None
                cur_lamb = None
            elif cond_num == 3:
                cur_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
                cur_wd = float(frag_txt.split("_wd")[-1].split("_k")[0])
                cur_k = int(frag_txt.split("_k")[-1])
                cur_lamb = None
            elif cond_num == 4:
                cur_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
                cur_wd = float(frag_txt.split("_wd")[-1].split("_k")[0])
                cur_k = int(frag_txt.split("_k")[-1].split("_lamb")[0])
                cur_lamb = float(frag_txt.split("_lamb")[-1])
            else:
                print(f"Encounter issue: {e} for file {f}")
                continue

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
            cond_num = len(frag_txt.split('/')[-1].split('_'))
            if cond_num == 2:
                best_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
                best_wd = float(frag_txt.split("_wd")[-1])
                best_k = None
                best_lamb = None
            elif cond_num == 3:
                best_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
                best_wd = float(frag_txt.split("_wd")[-1].split("_k")[0])
                best_k = int(frag_txt.split("_k")[-1])
                best_lamb = None
            elif cond_num == 4:
                best_lr = float(frag_txt.split("/lr")[-1].split("_wd")[0])
                best_wd = float(frag_txt.split("_wd")[-1].split("_k")[0])
                best_k = int(frag_txt.split("_k")[-1].split("_lamb")[0])
                best_lamb = float(frag_txt.split("_lamb")[-1])
    return best_lr, best_wd, best_k, best_lamb, best_val_acc


xx,which = ["caltech101","cifar","clevr_count","clevr_dist","diabetic_retinopathy","dmlab","dsprites_loc","dsprites_ori","dtd","eurosat","kitti","oxford_flowers102","oxford_iiit_pet","patch_camelyon","resisc45","smallnorb_azi","smallnorb_ele","sun397","svhn"], 'vtab'

ret = []
for ds in xx:
    files = glob.glob(f'output_val/{which}-{ds}/mae_vitb16/*/run1/eval_results.pth')
    if not files:
        continue
    try:
        print(ds, find_best_lrwd(files, f'{which}-{ds}'))
        # ret.append(lamb)
    except Exception as e:
        continue