# Variational SB

Some commands that can be runned:

python3 train_mnist.py --dir checkpoints/mnist-cld --sde cld 

python3 training.py --dataset checkerboard --model_forward linear  --model_backward mlp  --sde linear-momentum-sb  --dir checkpoints/board-momentum --num_iters 2500 --forward_opt_steps 2 --backward_opt_steps 500