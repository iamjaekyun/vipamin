# VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion

This repository contains the official PyTorch implementation for NeurIPS 2025 VIPAMIN: Visual Prompt Initialization via Embedding Selection and Subspace Expansion. Our work is based on [Visual Prompt Tuning](https://github.com/KMnP/vpt), [E2VPT: An Effective and Efficient Approach for Visual Prompt Tuning](https://github.com/ChengHan111/E2VPT), and [Gated Prompt Tuning](https://github.com/ryongithub/GatedPromptTuning). We thank the great work of them. 

## Quickstart

1. Prepare datasets (Refer to [Datasets Preparation](#datasets-preparation))
2. Generate config files (`gen_yaml_*.py`)
3. **Generate VIPAMIN-initialized prompts** (`load_match.py` and `load_orth.py`)
4. Hyperparameter sweep (`tune_vipamin_indiv.py`)
5. Final run (`tune_vipamin_finalfinal.py`)
6. Parse the results (`parse_sweep.py`)

## Environment settings

See `env_setup.sh`

## Structure of the this repo (Many thanks to VPT, key files are marked with 👉)

- `src/configs`: handles config parameters for the experiments.
  
  * 👉 `gen_yaml_*.py`: <u>Generates config file for VIPAMIN experiment</u> 

- `src/data`: loading and setup input datasets. The `src/data/vtab_datasets` are borrowed from 
  [VTAB github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data).


- `src/engine`: main training and eval actions here.

- `src/models`: handles backbone archs and heads for different fine-tuning protocols 

    * 👉`src/models/vit_prompt`: <u>a folder contains the same backbones in `vit_backbones` folder,</u> specified for VPT. This folder should contain the same file names as those in  `vit_backbones`

    * 👉 `src/models/vit_models.py`: <u>main model for transformer-based models</u> ❗️Note❗️: Current version only support ViT with mae, moco-v3

    * `src/models/build_model.py`: main action here to utilize the config and build the model to train / eval.

- `src/solver`: optimization, losses and learning rate schedules.  
- `src/utils`: helper functions for io, loggings, training, visualizations. 
- 👉`train.py`: call this one for training and eval a model with a specified transfer type.
- 👉`tune_vipamin_indiv.py`: call this one for tuning hyperparameters for a model with a specified transfer type. 
- 👉`tune_vipamin_finalfinal.py`: call this one for final training & evaluation. The best hyperparameter set from the result of `tune_vipamin_indiv.py` is automatically selected.
- `launch.py`: contains functions used to launch the job.

## Experiments

### Key configs:

- VIPAMIN related:
  - MODEL.PROMPT.K: Controls the locality of the matching prompt
  - MODEL.PROMPT.LAMB: Controls the strength of orthogonal prompt
  
- Vision backbones:
  - DATA.FEATURE: specify which representation to use
  - MODEL.TYPE: the general backbone type, e.g., "ssl-vit"
  - MODEL.MODEL_ROOT: folder with pre-trained model checkpoints
- Optimization related: 
  - SOLVER.BASE_LR: learning rate for the experiment
  - SOLVER.WEIGHT_DECAY: weight decay value for the experiment
  - DATA.BATCH_SIZE
- Datasets related:
  - DATA.NAME
  - DATA.DATAPATH: where you put the datasets
  - DATA.NUMBER_CLASSES
- Others:
  - OUTPUT_DIR: output dir of the final model and logs

### Datasets preparation

Follow the instructions in [VPT](https://github.com/KMnP/vpt) for more details. We follow the same datasets setup as VPT and E2VPT. For FGVC, refer to this [link](https://github.com/KMnP/vpt/issues/74#issuecomment-2744912188)

### Pre-trained model preperation

Download and place the pre-trained Transformer-based backbones to `MODEL.MODEL_ROOT`.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
<th valign="bottom">Link</th>
<th valign="bottom">md5sum</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">MoCo v3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar">link</a></td>
<td align="center"><tt>8f39ce</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MAE</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">link</a></td>
<td align="center"><tt>8cad7c</tt></td>
</tr>
<tr><td align="left">ViT-Tiny/16</td>
<td align="center">MAE</td>
<td align="center"><a href="https://drive.google.com/file/d/1ZQYlvCPLZrJDqn2lp4GCIVL246WPqgEf/view">link</a></td>
<td align="center"><tt>1cb2a4</tt></td>
</tr>
<tr><td align="left">ViT-Small/16</td>
<td align="center">DINO</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth">link</a></td>
<td align="center"><tt>cf0f22</tt></td>
</tr>
</tbody></table>

<!-- ### Examples for training and aggregating results

See [`demo.ipynb`](https://github.com/KMnP/vpt/blob/main/demo.ipynb) for how to use this repo. -->

## Results

For VTAB-1k experiment, VIPAMIN achieves the following performance

### MoCo-v3 pretrained ViT-B/16

| **Method**   | *Natural* | *Specialized* | *Structured* | **Mean** |
|--------------|-----------|----------------|---------------|----------|
| **VIPAMIN**  | 76.75 | 84.14        | 56.68     | 69.86 |

### MAE pretrained ViT-B/16

| **Method**   | *Natural* | *Specialized* | *Structured* | **Mean** |
|--------------|-----------|----------------|---------------|----------|
| **VIPAMIN**  | 62.60 | 79.96        | 57.47     | 64.09 |


## License

The majority of VPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) and [DINO] (https://github.com/facebookresearch/dino) and [MAE-Lite] (https://github.com/wangsr126/MAE-Lite) are licensed under the Attribution-NonCommercial 4.0 International license.
