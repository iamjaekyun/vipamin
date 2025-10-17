DATA2CLS = {
    'CUB': 200,
    'nabirds': 555,
    'OxfordFlowers': 102,
    'StanfordDogs': 120,
    'StanfordCars': 196,
}

DATA2PATH = {
    'CUB': 'CUB_200_2011',
    'nabirds': 'nabirds',
    'OxfordFlowers': 'OxfordFlower',
    'StanfordDogs': 'Stanford-dogs',
    'StanfordCars': 'Stanford-cars',
}

TRANSLATOR = {
    'CUB': 'cub',
    'nabirds': 'birds',
    'OxfordFlowers': 'flowers',
    'StanfordDogs': 'dogs',
    'StanfordCars': 'cars',
}


import os
import yaml
from pathlib import Path
mns = ['mae', 'moco']
Path(f'cfg_for_full').mkdir(exist_ok=True)
Path(f'cfg_for_vpt').mkdir(exist_ok=True)
for K_SHOT in [1,2,4,8]:
    for mn in mns:
        for k in DATA2CLS:
            ret = {}
            ret['_BASE_'] = f'../configs/base-{mn}.yaml'
            ret['RUN_N_TIMES'] = 1
            ret['DATA'] = {
                'NAME': f'fgvc-{k}',
                'DATAPATH': os.path.join('/home/data', 'fgvc', DATA2PATH[k]),
                'NUMBER_CLASSES': DATA2CLS[k],
                'MULTILABEL': False,
                'BATCH_SIZE': 256,
            }
            ret['MODEL'] = {
                # 'PROMPT': {
                #     'ID': TRANSLATOR[k],
                #     'INITIATION': 'random',
                #     'NUM_TOKENS': 100,
                # }
                'TRANSFER_TYPE': 'end2end'
            }
            ret['SOLVER'] = {
                'OPTIMIZER': 'adamw'
            }
            with open(f'cfg_for_full/{mn}_{TRANSLATOR[k]}.yaml', 'w') as f:
                yaml.dump(ret, f)
        for k in DATA2CLS:
            ret = {}
            ret['_BASE_'] = f'../configs/base-{mn}.yaml'
            ret['RUN_N_TIMES'] = 1
            ret['DATA'] = {
                'NAME': f'fgvc-{k}',
                'DATAPATH': os.path.join('/home/data', 'fgvc', DATA2PATH[k]),
                'NUMBER_CLASSES': DATA2CLS[k],
                'MULTILABEL': False,
                'BATCH_SIZE': 256,
            }
            ret['MODEL'] = {
                'PROMPT': {
                #     'ID': TRANSLATOR[k],
                    'INITIATION': 'random',
                    'NUM_TOKENS': 100,
                }
                # 'TRANSFER_TYPE': 'tinytl-bias'
            }
            ret['SOLVER'] = {
                'OPTIMIZER': 'sgd'
            }
            with open(f'cfg_for_vpt/{mn}_{TRANSLATOR[k]}.yaml', 'w') as f:
                yaml.dump(ret, f)
        Path(f'fgvc_kshot/my_{mn}_{K_SHOT}').mkdir(exist_ok=True, parents=True)
        for k in DATA2CLS:
            ret = {}
            ret['_BASE_'] = f'../../configs/base-{mn}.yaml'
            ret['RUN_N_TIMES'] = 1
            ret['OUTPUT_DIR'] = f'output_{TRANSLATOR[k]}_{K_SHOT}'
            ret['DATA'] = {
                'NAME': f'fgvc-{k}',
                'DATAPATH': os.path.join('/home/data', 'fgvc', DATA2PATH[k]),
                'NUMBER_CLASSES': DATA2CLS[k],
                'MULTILABEL': False,
                'BATCH_SIZE': 32,
                'K_SHOT': K_SHOT
            }
            ret['MODEL'] = {
                'PROMPT': {
                    'ID': TRANSLATOR[k],
                    'INITIATION': 'vipamin',
                    'NUM_TOKENS': 100,
                }
            }
            ret['SOLVER'] = {
                'OPTIMIZER': 'adamw'
            }
            with open(f'fgvc_kshot/my_{mn}_{K_SHOT}/{TRANSLATOR[k]}.yaml', 'w') as f:
                yaml.dump(ret, f)
