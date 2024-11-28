#!/bin/bash

#SBATCH --account=gts-mtao8
#SBATCH -N 16 --gres=gpu:V100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -t8:00:00
#SBATCH -q embers
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

srun torchrun --nnodes 16 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
    training.py --dir experiments/sparta/ --load_from_ckpt experiments/sparta/itr_195000 --batch_size 256 --disable_wandb --damp_coef .9


# srun torchrun --nnodes 8 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 \
#     sampling.py --dir samples_dir --load_from_ckpt experiments/cifar-ema/itr_350000 --batch_size 1000 --num_samples 30000

# srun torchrun --nnodes 16 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoin $head_node_ip:29500     sampling.py --dir samples_445 \
#  --load_from_ckpt experiments/damp85/itr_445000 --batch_size 2000 --num_samples 50000 --num_steps 125 --damp_coef .85

# torchrun sampling.py --sde vp --dir samples_vp --load_from_ckpt vp-song.pt --batch_size 32 --num_samples 32