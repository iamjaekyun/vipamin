declare -a vtabs=(
    "cub"
    "dogs"
    "cars"
    "birds"
    "flowers"
)

for i in "${vtabs[@]}"
do
    # StringArray+=("python load_ora_cand.py --data-set $i")
    # StringArray+=("python load_antiora.py --data-set $i")
    StringArray+=("python load_ora_cand_h.py --data-set $i")
    # StringArray+=("python load_antiora_h.py --data-set $i")
    # StringArray+=("python load_ora_cand_l.py --data-set $i")
    # StringArray+=("python load_antiora_l.py --data-set $i")
done


gpuqueue "${StringArray[@]}" --available_gpus 0 1 2 3