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


# test input data
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=eval_g \
        python test_input_data.py 

srun -u --partition=Sensetime -n1 --gres=gpu:2 --ntasks-per-node=4 -w SG-IDC1-10-51-0-42 \
        --job-name=tg2_big \
        python train.py \
        --class_choice Chair --epochs 1 \
        --FPD_path ./evaluation/pre_statistics_chair.npz 
        # --ckpt_path ./model/checkpoints2/


# eval sampple 
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=eval_g \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_all.npz \
        --class_choice None \
        --save_num_generated 500 \
        --save_generated_dir ./gen_all \
        --model_pathname ./model/checkpoints18/tree_ckpt_1180_None.pt 
        \
        --num_samples 100
        /model/checkpoints/tree_ckpt_1340_None.pt 
# eval_cgan
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=eval_g \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --class_choice None \
        --save_num_generated 500 \
        --gen_path ./gen_cgan_v0 \
        --model_pathname ./model/checkpoints_cgan_v0/tree_ckpt_240_None.pt \
        --num_samples 100 \
        --conditional True \
        --batch_size 50

# eval chair
srun -u --partition=Sensetime -n1 --gres=gpu:1 --ntasks-per-node=1 -x SG-IDC1-10-51-0-30 \
        --job-name=eval_g \
        python eval_GAN.py \
        --FPD_path ./evaluation/pre_statistics_chair.npz \
        --class_choice Chair \
        --save_num_generated 500 \
        --save_generated_dir ./gen_to_delete \
        --model_pathname ./model/checkpoints18/tree_ckpt_1660_Chair.pt \
        --num_samples 5000

# May 31, train cgan, 4.5
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=1 -x SG-IDC1-10-51-0-34 \
        --job-name=cgv0_4.5 \
        python train_cgan.py \
        --class_choice None --epochs 2000   --batch_size 64 \
        --ckpt_path gen_cgan_v0_4.5 \
        --FPD_path ./evaluation/pre_statistics_all.npz

        \
        --ckpt_load ./model/checkpoints_cgan_v0/tree_ckpt_200_None.pt

# May 31, train gan, 4 class 500, 
srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_4.5 \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64  \
        --ratio_base 500 \
        --dataset ShapeNet_v0  \
        --ckpt_path gen_gan_4.5 \
        --FPD_path ./evaluation/pre_statistics_all.npz

srun -u --partition=Sensetime -n1 --gres=gpu:4 --ntasks-per-node=4 -x SG-IDC1-10-51-0-34 \
        --job-name=tg_4.1 \
        python train.py \
        --class_choice None --epochs 2000   --batch_size 64  \
        --ratio_base 1000 \
        --dataset ShapeNet_v0  \
        --ckpt_path gen_gan_4.1 \
        --FPD_path ./evaluation/pre_statistics_all.npz



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
        --class_choice None --epochs 2000   --batch_size 64                

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