#!/bin/bash
# Reliability stream A: seed sweep on motion-weak cases (variance vs systematic)
set -e
cd /home/jeongseon39/MLLAB/FLUX
PY=~/anaconda3/envs/flowedit/bin/python
export CUDA_VISIBLE_DEVICES=0,1

PUG="The pug wearing a green outfit with the white letter K is jumping high in the air to catch a tennis ball."
PEN="The penguin wearing a green and yellow striped scarf is sliding on its belly across the snow."
HOR="The horse wearing a checkered outfit with the red number 7 is rearing up on its hind legs."

for S in 123 777; do
  $PY src/pipeline.py --input data/sources/pug_shirt.png \
    --mask data/masks/mask_pug_shirt_rough.png --object pug \
    --prompt "$PUG" --seed $S --output outputs/bench/pug_jump_ours_s$S.png
  $PY src/pipeline.py --input data/sources/penguin_scarf.png \
    --mask data/masks/mask_penguin_scarf_rough.png --object penguin \
    --prompt "$PEN" --seed $S --output outputs/bench/penguin_scarf_ours_s$S.png
done
$PY src/pipeline.py --input data/sources/horse_blanket.png \
  --mask data/masks/mask_horse_blanket_rough.png --object horse \
  --prompt "$HOR" --seed 123 --output outputs/bench/horse_blanket_ours_s123.png
