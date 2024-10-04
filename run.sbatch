#!/bin/bash

#SBATCH --account=gts-mtao8
#SBATCH -N4 --gres=gpu:A100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -C A100-40GB
#SBATCH -t24:00:00
#SBATCH -q inferno
#SBATCH -o logs_coco4.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=krojas6@gatech.edu

cd /storage/home/hcoda1/7/krojas6/p-mtao8-0/repos/Multimodal-Diffusion/storage/home/hcoda1/7/krojas6/p-mtao8-0/repos/variational-sb
module load anaconda3
conda activate variational

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
    --nnodes 4 \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    training.py --dir experiments/cifar-unet-3/ --load_from_ckpt experiments/cifar-unet-3/itr_230000 --batch_size 256 --disable_wandb