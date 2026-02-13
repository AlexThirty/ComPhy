#!/bin/bash
conda activate MLgeneric

device="$1"
weight_mode="$2"

echo "Using device: $device"

python run.py --device cuda:1 --weight_mode static --model dualpinnncl --alignment_mode DERL &
python run.py --device cuda:2 --weight_mode static --model triplepinnncl --alignment_mode DERL
python run.py --device cuda:1 --weight_mode static --model dualpinnncl --alignment_mode SOB &
python run.py --device cuda:2 --weight_mode static --model triplepinnncl --alignment_mode SOB
