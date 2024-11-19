#!/bin/bash
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=24:00:00
source ~/.bashrc
conda activate pytch
cd /gs/fs/tga-i/zhou.y.ak/workspace/A2J-Transformer_backbone3

python train.py --gpu 0 --wandb
python test.py --test_epoch 99 --gpu 0
python test.py --test_epoch 98 --gpu 0
python test.py --test_epoch 97 --gpu 0
python test.py --test_epoch 96 --gpu 0
python test.py --test_epoch 95 --gpu 0
python test.py --test_epoch 94 --gpu 0
python test.py --test_epoch 93 --gpu 0
python test.py --test_epoch 92 --gpu 0
python test.py --test_epoch 91 --gpu 0
python test.py --test_epoch 90 --gpu 0
python test.py --test_epoch 89 --gpu 0
python test.py --test_epoch 88 --gpu 0
python test.py --test_epoch 87 --gpu 0
python test.py --test_epoch 86 --gpu 0
python test.py --test_epoch 85 --gpu 0
python test.py --test_epoch 84 --gpu 0
python test.py --test_epoch 83 --gpu 0
python test.py --test_epoch 82 --gpu 0
python test.py --test_epoch 81 --gpu 0
python test.py --test_epoch 80 --gpu 0

