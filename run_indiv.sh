declare -a vtabs=(
    # "caltech101"
    "cifar"
    "clevr_count"
    "clevr_dist"
    "diabetic_retinopathy"
    "dmlab"
    "dsprites_loc"
    "dsprites_ori"
    "dtd"
    "eurosat"
    "kitti"
    "oxford_flowers102"
    "oxford_iiit_pet"
    "patch_camelyon"
    "resisc45"
    "smallnorb_azi"
    "smallnorb_ele"
    "sun397"
    "svhn"
)

declare -a lrrs=(
    0.05
    0.1
    0.25
    0.5
    1.0
    2.5
    5.0
)

for i in "${vtabs[@]}"
do
    for j in "${lrrs[@]}"
    do
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 2 --lambdaa 0.0")
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 8 --lambdaa 0.0")
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 32 --lambdaa 0.0")
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 128 --lambdaa 0.0")
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 2 --lambdaa 0.5")
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 8 --lambdaa 0.5")
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 32 --lambdaa 0.5")
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 128 --lambdaa 0.5")
        StringArray+=("python tune_vtab_hyper_indiv.py --config-file my_mae/$i.yaml --train-type prompt --lrr $j --wdd 0.01 --kk 2 --lambdaa 1.0")
    done
done

gpuqueue "${StringArray[@]}" --available_gpus 0 1 2 3 4 5 6 7