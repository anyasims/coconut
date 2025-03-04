## For running:
### GSM8K
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml # stage 0
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml # coconut
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml # eval
### ProsQA
# MASTER_PORT=29501 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml


# python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); print(s.connect_ex(('127.0.0.1', 29501)))" # "111" means error i.e. not in use, i.e. free. "0" means in use. Torch's default port is 29500.

CUDA_VISIBLE_DEVICES=1 python run_full1.py use_wandb=False