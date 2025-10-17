DATA2CLS = {
    'caltech101': 102,
    'cifar': 100,
    'dtd': 47,
    'oxford_flowers102': 102,
    'oxford_iiit_pet': 37,
    'patch_camelyon': 2,
    'sun397': 397,
    'svhn': 10,
    'resisc45': 45,
    'eurosat': 10,
    'dmlab': 6,
    'kitti(task="closest_vehicle_distance")': 4,
    'smallnorb(predicted_attribute="label_azimuth")': 18,
    'smallnorb(predicted_attribute="label_elevation")': 9,
    'dsprites(predicted_attribute="label_x_position",num_classes=16)': 16,
    'dsprites(predicted_attribute="label_orientation",num_classes=16)': 16,
    'clevr(task="closest_object_distance")': 6,
    'clevr(task="count_all")': 8,
    'diabetic_retinopathy(config="btgraham-300")': 5
}


TRANSLATOR = {
    'caltech101': 'caltech101',
    'cifar': 'cifar',
    'dtd': 'dtd',
    'oxford_flowers102': 'oxford_flowers102',
    'oxford_iiit_pet': 'oxford_iiit_pet',
    'patch_camelyon': 'patch_camelyon',
    'sun397': 'sun397',
    'svhn': 'svhn',
    'resisc45': 'resisc45',
    'eurosat': 'eurosat',
    'dmlab': 'dmlab',
    'kitti(task="closest_vehicle_distance")': 'kitti',
    'smallnorb(predicted_attribute="label_azimuth")': 'smallnorb_azi',
    'smallnorb(predicted_attribute="label_elevation")': 'smallnorb_ele',
    'dsprites(predicted_attribute="label_x_position",num_classes=16)': 'dsprites_loc',
    'dsprites(predicted_attribute="label_orientation",num_classes=16)': 'dsprites_ori',
    'clevr(task="closest_object_distance")': 'clevr_dist',
    'clevr(task="count_all")': 'clevr_count',
    'diabetic_retinopathy(config="btgraham-300")': 'diabetic_retinopathy',
}

from pathlib import Path
import yaml

Path(f'cfg_for_vpt').mkdir(exist_ok=True)
Path(f'cfg_for_full').mkdir(exist_ok=True)

for mn in ['mae', 'moco']:
    Path(f'my_{mn}').mkdir(exist_ok=True)
    for k in DATA2CLS:
        ret = {}
        ret['_BASE_'] = f'../configs/base-{mn}.yaml'
        ret['RUN_N_TIMES'] = 1
        ret['DATA'] = {
            'NAME': f'vtab-{TRANSLATOR[k]}',
            'DATAPATH': '/home/data/vtab-1k_tfds/',
            'NUMBER_CLASSES': DATA2CLS[k],
            'MULTILABEL': False,
            'BATCH_SIZE': 256,
        }
        ret['MODEL'] = {
            # 'PROMPT': {
                # 'ID': TRANSLATOR[k],
                # 'INITIATION': 'random',
                # 'NUM_TOKENS': 100,
                # 'DEEP': True,
            # }
            'TRANSFER_TYPE': 'end2end',
        }
        ret['SOLVER'] = {
            'OPTIMIZER': 'adamw'
        }
        with open(f'cfg_for_full/{mn}_{TRANSLATOR[k]}.yaml', 'w') as f:
            yaml.dump(ret, f)
    Path(f'my_{mn}').mkdir(exist_ok=True)
    for k in DATA2CLS:
        ret = {}
        ret['_BASE_'] = f'../configs/base-{mn}.yaml'
        ret['RUN_N_TIMES'] = 1
        ret['DATA'] = {
            'NAME': f'vtab-{TRANSLATOR[k]}',
            'DATAPATH': '/home/data/vtab-1k_tfds/',
            'NUMBER_CLASSES': DATA2CLS[k],
            'MULTILABEL': False,
            'BATCH_SIZE': 256,
        }
        ret['MODEL'] = {
            'PROMPT': {
                # 'ID': TRANSLATOR[k],
                'INITIATION': 'random',
                'NUM_TOKENS': 100,
                # 'DEEP': True,
            }
            # 'TRANSFER_TYPE': 'end2end',
        }
        ret['SOLVER'] = {
            'OPTIMIZER': 'adamw'
        }
        with open(f'cfg_for_vpt/{mn}_{TRANSLATOR[k]}.yaml', 'w') as f:
            yaml.dump(ret, f)
    Path(f'my_{mn}').mkdir(exist_ok=True)
    for k in DATA2CLS:
        ret = {}
        ret['_BASE_'] = f'../configs/base-{mn}.yaml'
        ret['RUN_N_TIMES'] = 1
        ret['DATA'] = {
            'NAME': f'vtab-{TRANSLATOR[k]}',
            'DATAPATH': '/home/data/vtab-1k_tfds/',
            'NUMBER_CLASSES': DATA2CLS[k],
            'MULTILABEL': False,
            'BATCH_SIZE': 32,
        }
        ret['MODEL'] = {
            'PROMPT': {
                'ID': TRANSLATOR[k],
                'INITIATION': 'vipamin',
                'NUM_TOKENS': 100,
                # 'DEEP': True,
            }
            # 'TRANSFER_TYPE': 'end2end',
        }
        ret['SOLVER'] = {
            'OPTIMIZER': 'adamw'
        }
        with open(f'my_{mn}/{TRANSLATOR[k]}.yaml', 'w') as f:
            yaml.dump(ret, f)
    Path(f'my_{mn}_deep').mkdir(exist_ok=True)
    for k in DATA2CLS:
        ret = {}
        ret['_BASE_'] = f'../configs/base-{mn}.yaml'
        ret['RUN_N_TIMES'] = 1
        ret['DATA'] = {
            'NAME': f'vtab-{TRANSLATOR[k]}',
            'DATAPATH': '/home/data/vtab-1k_tfds/',
            'NUMBER_CLASSES': DATA2CLS[k],
            'MULTILABEL': False,
            'BATCH_SIZE': 32,
        }
        ret['MODEL'] = {
            'PROMPT': {
                'ID': TRANSLATOR[k],
                'INITIATION': 'vipamin',
                'NUM_TOKENS': 20,
                'DEEP': True,
            }
            # 'TRANSFER_TYPE': 'end2end',
        }
        ret['SOLVER'] = {
            'OPTIMIZER': 'adamw'
        }
        with open(f'my_{mn}_deep/{TRANSLATOR[k]}.yaml', 'w') as f:
            yaml.dump(ret, f)