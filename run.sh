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

# tg_big
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -w SG-IDC1-10-51-0-35 \
        --job-name=tg_big \
        python train.py \
        --class_choice None --epochs 2000   

srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-35 \
        --job-name=tg_big \
        python train.py \
        --class_choice None --epochs 2000 \   

# eval sampple 
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=eval_g \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --class_choice None \
        --save_num_generated 500 \
        --save_generated_dir ./gen_all \
        --model_pathname ./model/checkpoints18/tree_ckpt_100_None.pt \
        --num_samples 100
        

        /model/checkpoints/tree_ckpt_1340_None.pt 
# eval chair
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=eval_g \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --class_choice Chair \
        --save_num_generated 500 \
        --save_generated_dir ./gen_to_delete \
        --model_pathname ./model/checkpoints18/tree_ckpt_1660_Chair.pt \
        --num_samples 100


# tg_chair script, on 430pm May 14 (batch 64)

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg18chair \
        python train.py \
        --class_choice Chair --epochs 2000 \
        --FPD_path ./evaluation/pre_statistics_chair.npz

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg18plane \
        python train.py \
        --class_choice Airplane --epochs 2000 \
        --FPD_path ./evaluation/pre_statistics_plane.npz

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg18all \
        python train.py \
        --class_choice None --epochs 2000 \
        --FPD_path ./evaluation/pre_statistics_all.npz



srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_chair \
        python train.py \
        --class_choice Chair --epochs 2000 --batch_size 20

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_chair \
        python train.py \
        --class_choice Chair --epochs 2000  \
        --ckpt_load tree_ckpt_420_Chair.pt

# tg_plane script, on 11pm May 14 (IP but not shown in squeue)
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_plane \
        python train.py \
        --class_choice Airplane --epochs 2000 --batch_size 20        

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_plane \
        python train.py \
        --class_choice Airplane --epochs 2000   \
        --ckpt_load tree_ckpt_530_Chair.pt
# tg_all 
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_all \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 20                

# tg_sofa ( actually it is table)
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_table \
        python train.py \
        --class_choice Table --epochs 2000    --batch_size 20  


srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_cap \
        python train.py \
        --class_choice Cap --epochs 2000    --batch_size 20  
       

srun -u --partition=Sensetime -n1 --job-name=allcls2  --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python train.py --class_choice None --epochs 2000


# preprocess for all class, but not uniform
srun -u --partition=Sensetime -n1 --job-name=allcls2  --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 python dataset_preparation.py --class_choice None