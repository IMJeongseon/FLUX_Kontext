#!/bin/bash
# Move experiments, batch 2 (7 regenerated sources), sequential on 2 GPUs.
# Prompt rule: neutral garment words ("outfit"), marker mentioned explicitly,
# explicit motion verb (displacement is a co-requirement of vacate).
set -e
cd /home/jeongseon39/MLLAB/FLUX
PY=~/anaconda3/envs/flowedit/bin/python
export CUDA_VISIBLE_DEVICES=${GPUS:-2,3}

run() {  # name object prompt
  $PY src/pipeline.py \
    --input "data/sources/$1.png" \
    --mask "data/masks/mask_$1_rough.png" \
    --object "$2" \
    --prompt "$3" \
    --output "outputs/bench/$1_ours.png"
}

run rabbit_vest rabbit \
  "The white rabbit wearing an orange vest is jumping high in the air over the grass."
run penguin_scarf penguin \
  "The penguin wearing a green and yellow striped scarf is sliding on its belly across the snow."
run goat_blanket goat \
  "The goat wearing a purple outfit with yellow stars is rearing up on its hind legs."
run duck_ribbon duck \
  "The duck with a red and white striped ribbon is flapping its wings and leaping into the air."
run sheep_tag sheep \
  "The sheep with a blue number 3 marked on its side is jumping high in the air in the meadow."
run horse_blanket horse \
  "The horse wearing a checkered outfit with the red number 7 is rearing up on its hind legs."
run dalmatian_collar dalmatian \
  "The dalmatian wearing a yellow collar is jumping high in the air to catch a red frisbee."
