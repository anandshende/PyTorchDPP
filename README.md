# PyTorchDPP

torchrun --standalone --nproc_per_node=1 server.py


# Export env
1. conda env export | grep -v "^prefix: " > environment.yml

# Import env
1. conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name
