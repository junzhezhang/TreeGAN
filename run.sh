#!/bin/bash

partition=${1}
model_name=${2}
gpus=${3:-1}

root_dir=/mnt/lustre/$(whoami)/McDALF
config_dir=${root_dir}/mt_models/${model_name}
g=$((${gpus}<8?${gpus}:8))

export PYTHONPATH=${root_path}:$PYTHONPATH

srun -u --partition=${partition} --job-name=${model_name} \
    -n1 --gres=gpu:${gpus} --ntasks-per-node=1 \
    -x SG-IDC1-10-51-0-34 \
    python demo.py --name "bird_net" --num_train_epoch 500 --img_path "misc/demo_data/img1.jpg"



rm -r /mnt/lustre/zhangjunzhe/SinGAN/TrainedModels/birds  && 
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python train.py --class_choice None

srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python train.py --class_choice Airplane --batch_size 1000
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python train.py --class_choice Airplane

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        python train.py \
        --class_choice Chair --epochs 80 \
        --load_ckpt /mnt/lustre/zhangjunzhe/TreeGAN/model/checkpoints/tree_ckpt_19_Airplane.pt


srun -u --partition=Sensetime -n1 --job-name=allcls2  --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python train.py --class_choice None --epochs 2000


# preprocess for all class, but not uniform
srun -u --partition=Sensetime -n1 --job-name=allcls2  --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python dataset_preparation.py --class_choice None