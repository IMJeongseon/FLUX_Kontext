#!/bin/bash
# Wait until two GPUs are free (<1GiB used), then run the 3 OT-arm
# experiments sequentially on that pair. Polls every 60s, gives up after 4h.
cd /home/jeongseon39/MLLAB/FLUX
PY=~/anaconda3/envs/flowedit/bin/python

free_pair() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' '$2 < 1024 {print $1}' | head -2 | paste -sd, -
}

for i in $(seq 1 240); do
  PAIR=$(free_pair)
  if [ "$(echo "$PAIR" | tr ',' '\n' | wc -l)" -ge 2 ]; then
    echo "[wait] GPUs $PAIR free after $((i-1)) min, starting"
    break
  fi
  sleep 60
done
PAIR=$(free_pair)
[ "$(echo "$PAIR" | tr ',' '\n' | wc -l)" -ge 2 ] || { echo "timeout"; exit 1; }

export CUDA_VISIBLE_DEVICES=$PAIR

$PY src/pipeline.py \
  --input data/sources/pug_shirt.png --mask data/masks/mask_pug_shirt_rough.png --object pug \
  --prompt "The pug wearing a green outfit with the white letter K is jumping high in the air to catch a tennis ball." \
  --seed 123 --identity-ot \
  --output outputs/bench/pug_jump_ours_s123_ot.png

$PY src/pipeline.py \
  --input data/sources/kitten_sweater.png --mask data/masks/mask_kitten_sweater_rough.png --object kitten \
  --prompt "The kitten wearing a red and white patterned outfit is leaping high in the air to catch a feather toy." \
  --identity-ot \
  --output outputs/bench/kitten_leap_ours_ot.png

$PY src/pipeline.py \
  --input data/sources/dalmatian_collar.png --mask data/masks/mask_dalmatian_collar_rough.png --object dalmatian \
  --prompt "The dalmatian wearing a yellow collar is jumping high in the air to catch a red frisbee." \
  --identity-ot \
  --output outputs/bench/dalmatian_collar_ours_ot.png
