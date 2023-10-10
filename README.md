# PyTorchDPP

torchrun --standalone --nproc_per_node=1 server.py
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=192.168.1.18:29400 fine-tuning-bert-small-topic-classification.py

# Export env
1. conda env export | grep -v "^prefix: " > environment.yml

# Import env
1. conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name
