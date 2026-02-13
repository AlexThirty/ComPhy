#!/bin/bash
python train_pinn.py --restart --device cuda:1 --weight_type grad & 
python train_dualncl.py --restart --device cuda:0 --weight_type grad --mode DERL &
python train_dualncl.py --restart --device cuda:1 --weight_type grad --mode SOB
python train_dualpinn.py --restart --device cuda:0 --weight_type grad --mode DERL &
python train_dualpinn.py --restart --device cuda:1 --weight_type grad --mode SOB
python train_ncl.py --restart --device cuda:0 --weight_type grad


python train_triplepinn.py --restart --device cuda:0 --weight_type grad --mode DERL &
python train_triplepinn.py --restart --device cuda:1 --weight_type grad --mode SOB
python train_pinnncl.py --restart --device cuda:0 --weight_type grad --mode DERL &
python train_pinnncl.py --restart --device cuda:1 --weight_type grad --mode SOB
