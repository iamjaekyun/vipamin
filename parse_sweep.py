import glob
import pandas as pd

from src.utils.vis_utils import get_df, average_df

# from src.utils.vis_utils_pvk import get_df, average_df

LOG_NAME = "logs.txt"
MODEL_NAME = "mocov3_vitb"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

root = "output_finalfinal"

df_list=[]

for idx, seed in enumerate([42,44,82]):
    run = idx + 1
    files = glob.glob(f"{root}/*/{MODEL_NAME}/*/run{run}/{LOG_NAME}")
    for f in files:
        ds_name = f.split('/')[1][5:]
        df = get_df(files, root, ds_name, MODEL_NAME, is_best=False, is_last=True)
        # print(df)
        if df is None:
            continue
        df["seed"] = seed
    df_list.append(df)

df= pd.concat(df_list)
df["type"] = "VIPAMIN"

f_df = average_df(df, metric_names=["l-test_top1"], take_average=True)
print(f_df)