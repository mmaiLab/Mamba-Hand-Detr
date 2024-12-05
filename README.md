# Mamba-Hand-Detr
## Environment configration
- Follow the instruction at A2J-Transformer's repo: [a2j](https://github.com/ChanglongJiangGit/A2J-Transformer/tree/main).
Notice that the recommanded version of python and torch are 3.10 and 2.2.2 (not 3.7 and 1.7.1 as the a2j repo says).

- Follow the instruction at Mamba's repo: [mamba-ssm](https://github.com/state-spaces/mamba).
- There are still some packages that are needed, so when you run train.py, install them when you see the "module not found" error.
## Mamba vision backbone
- [paper](https://arxiv.org/pdf/2407.08083)  
- [repository](https://github.com/NVlabs/MambaVision)
## Config setting
- dataset: choose the dataset.py file from the "dataset" folder(interhand, RHD and STB is available) and put it into the main direction. 
- output_dir: '/direction/of/output'
- cur_dir: '/direction/of/this/project'
- backbone_type: 'swin_transformer'/'resnet'/'mamba_vision'
- backbone_size: for swin_transformer: B/L,  
for mamba_vision: T/T2/S/B/L/L2
- depthlayer: the number of depth layer of anchor points,default is 3
- lr_dec_epoch: [50, 90, 95] for small backbones,  
[60, 90, 95] for large backbones
## Wandb setting
Set in train.py
## Run
### Train with wandb
```python train.py --wandb```
### Continue training
```python train.py --continue```
### Job
Change **node type**, **run time**, **file direction**, and run  

```qsub -g tga-i run.sh```
## Result so far
- 10000 images on RHP dataset

|Method|lr_dec_epoch|MPJPE↓|
|-|-|-|
|a2j|[24, 45, 90]|22.31|
|swinB_AdamW|[50, 90, 95]|21.56|
|swinB_AdamW|[80, 90, 95]|20.67|
|mamba_visionL|[50, 90, 95]|20.85|
|mamba_visionL|[80, 90, 95]|20.36|
|depth layer = 1|[50, 90, 95]|**19.71**|
|depth layer = 1, mamba_visionS|[50, 90, 95]|20.68|
- full RHP dataset

|Method|lr_dec_epoch|MPJPE↓|
|-|-|-|
|a2j||17.75|
|swinB|[60, 90, 95]|17.49|
|swinL|[60, 90, 95]|17.42|
|mamba_visionT|[50, 90, 95]|17.87|
|mamba_visionT2|[50, 90, 95]|17.48|
|mamba_visionS|[50, 90, 95]|17.37|
|mamba_visionB|[60, 90, 95]|17.55|
|mamba_visionL|[60, 90, 95]|17.34|
|mamba_visionL2|[60, 90, 95]|17.34|
|depth layer = 1|[50, 90, 95]|**17.19**|
|depth layer = 1, mamba_visionS|[50, 90, 95]|17.44|
|depth layer = 1, mamba_visionB|[55, 90, 95]|17.41|
|depth layer = 1, mamba_visionL|[60, 90, 95]|17.21|
|depth layer = 1, mamba_visionL2|[50, 90, 95]|17.46|
