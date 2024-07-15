# Variational SB

Some commands that can be runned:

python3 training.py --dataset spiral --model_forward linear --model_backward mlp --sde linear-momentum-sb --loss_routine variational --forward_opt_steps 5 --backward_opt_steps 495 --dir checkpoints/spiral-skewed --num_iters 30000

python3 train_mnist.py --dir checkpoints/mnist-cld --sde cld 

python3 training.py --dataset checkerboard --model_forward linear  --model_backward mlp  --sde linear-momentum-sb  --dir checkpoints/board-momentum --num_iters 2500 --forward_opt_steps 2 --backward_opt_steps 500