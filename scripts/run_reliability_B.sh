#!/bin/bash
# Reliability stream B: horse seed 777 + prompt-fix arms + guidance lever
set -e
cd /home/jeongseon39/MLLAB/FLUX
PY=~/anaconda3/envs/flowedit/bin/python
export CUDA_VISIBLE_DEVICES=2,3

$PY src/pipeline.py --input data/sources/horse_blanket.png \
  --mask data/masks/mask_horse_blanket_rough.png --object horse \
  --prompt "The horse wearing a checkered outfit with the red number 7 is rearing up on its hind legs." \
  --seed 777 --output outputs/bench/horse_blanket_ours_s777.png

# goat: quadruped-anchored garment word ("cover on its back", not "outfit")
$PY src/pipeline.py --input data/sources/goat_blanket.png \
  --mask data/masks/mask_goat_blanket_rough.png --object goat \
  --prompt "The goat with a purple cover with yellow stars on its back is rearing up on its hind legs." \
  --output outputs/bench/goat_blanket_ours_coverfix.png

# duck: milder deformation, marker kept in prompt
$PY src/pipeline.py --input data/sources/duck_ribbon.png \
  --mask data/masks/mask_duck_ribbon_rough.png --object duck \
  --prompt "The duck with a red and white striped ribbon around its neck is spreading its wings wide." \
  --output outputs/bench/duck_ribbon_ours_mild.png

# pug: guidance as the prompt-adherence lever
$PY src/pipeline.py --input data/sources/pug_shirt.png \
  --mask data/masks/mask_pug_shirt_rough.png --object pug \
  --prompt "The pug wearing a green outfit with the white letter K is jumping high in the air to catch a tennis ball." \
  --guidance-scale 5.0 --output outputs/bench/pug_jump_ours_g5.png
