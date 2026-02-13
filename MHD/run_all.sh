#!/bin/bash
conda activate MLgeneric

weight_mode="$1"
device="$2"
echo "Weight mode: $weight_mode"
echo "Device: $device"

#python run.py --device cuda:0 --weight_mode static --model triplepinn --alignment_mode SOB &
python run.py --device cuda:1 --weight_mode grad --model ncl --alignment_mode none & 
python run.py --device cuda:0 --weight_mode grad --model pinn --alignment_mode none

python run.py --device cuda:1 --weight_mode grad --model dualpinn --alignment_mode SOB &
python run.py --device cuda:0 --weight_mode grad --model dualpinn --alignment_mode DERL 
python run.py --device cuda:1 --weight_mode grad --model triplepinn --alignment_mode SOB &
python run.py --device cuda:0 --weight_mode grad --model triplepinn --alignment_mode DERL 
python run.py --device cuda:1 --weight_mode grad --model quadpinn --alignment_mode SOB &
python run.py --device cuda:0 --weight_mode grad --model quadpinn --alignment_mode DERL 
#python run.py --device $device --weight_mode grad --model pinn --alignment_mode none


#python run.py --device cuda:1 --weight_mode static --model quadpinn --alignment_mode SOB
#python run.py --device $device --weight_mode $weight_mode --model triplepinn --alignment_mode SOB
#python run.py --device $device --weight_mode $weight_mode --model pinn --alignment_mode none
#python run.py --device $device --weight_mode $weight_mode --model ncl --alignment_mode none
