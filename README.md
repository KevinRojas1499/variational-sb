# Variational SB

Some commands that can be runned:
```{bash}
python3 training.py --dataset spiral --model_forward linear --model_backward mlp --sde linear-momentum-sb --loss_routine variational --forward_opt_steps 5 --backward_opt_steps 495 --dir checkpoints/spiral-skewed --num_iters 30000

python3 training.py --dataset exchange_rate --model_backward time-series  --num_iters 2000 --dir checkpoints/exc 
python3 sampling.py --dataset electricity_nips --model_backward time-series --batch_size 9 --load_from_ckpt checkpoints/elec/itr_2000/

python3 train_mnist.py --dir checkpoints/mnist-cld --sde cld 

python3 training.py --dataset checkerboard --model_forward linear  --model_backward mlp  --sde linear-momentum-sb  --dir checkpoints/board-momentum --num_iters 2500 --forward_opt_steps 2 --backward_opt_steps 500
```

# Time Series
```{bash}
 python time_series/main.py --data electricity_nips --seed 1 
--batch_size 32 --hidden_dim 64 --epochs 20 --forward_opt_steps 0 --backward_opt_steps 200 --t0 0.01 --T 1 --beta_min 0.1 --beta_max 10 --beta_r 1.7 --st
eps 100 --device 0
```
