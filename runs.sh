## For running:
### GSM8K
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml # stage 0
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml # coconut
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml # eval
### ProsQA
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29501 --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml


# python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); print(s.connect_ex(('127.0.0.1', 29501)))" # "111" means error i.e. not in use, i.e. free. "0" means in use. Torch's default port is 29500.

CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nnodes 1 --nproc_per_node 4 run_full_ddp.py use_wandb=False
CUDA_VISIBLE_DEVICES=2 torchrun --master_port=29501 --nnodes=1 --nproc_per_node=1 run_full_ddp.py use_wandb=False generation.generations_per_prompt=1 loss.loss_type=logp total_batch_size=32 per_device_batch_size=1 generation.as_full_distribution=True generation.max_length=1024 
CUDA_VISIBLE_DEVICES=2 python run_full_single.py use_wandb=False generation.generations_per_prompt=1 loss.loss_type=logp total_batch_size=32 per_device_batch_size=1 generation.as_full_distribution=True generation.step_for_answer=10 generation.max_length=1024
CUDA_VISIBLE_DEVICES=2 python run_full_single.py use_wandb=False total_batch_size=16 per_device_batch_size=2 generation.max_length=300 generation.generations_per_prompt=1 loss.loss_type=logp

MASTER_PORT=29502 CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nnodes 1 --nproc_per_node 4 run_full_ddp.py generation.generations_per_prompt=1 loss.loss_type=logp total_batch_size=6 per_device_batch_size=1 generation.as_full_distribution=True generation.step_for_answer=10 generation.max_length=1024

CUDA_VISIBLE_DEVICES=2 python run_full_single.py generation.generations_per_prompt=1 loss.loss_type=logp total_batch_size=3 per_device_batch_size=1 generation.as_full_distribution=True generation.step_for_answer=80 generation.max_length=1024 generation.compute_everything=True

CUDA_VISIBLE_DEVICES=2 python run_full_single.py generation.generations_per_prompt=1 loss.loss_type=logp total_batch_size=3 per_device_batch_size=1 generation.as_full_distribution=True generation.step_for_answer=80 generation.max_length=1024 generation.compute_everything=True

############################################################################################################
# 15/03/2025

CUDA_VISIBLE_DEVICES=0 python run_full3.py
CUDA_VISIBLE_DEVICES=0 python run_full3.py cot_normalization_type=rloo ans_normalization_type=rloo
CUDA_VISIBLE_DEVICES=1 python run_full3.py cot_normalization_type=null ans_normalization_type=null
CUDA_VISIBLE_DEVICES=1 python run_full3.py cot_reward_type=prob
CUDA_VISIBLE_DEVICES=2 python run_full3.py cot_reward_type=prob cot_normalization_type=rloo ans_normalization_type=rloo
CUDA_VISIBLE_DEVICES=2 python run_full3.py cot_reward_type=prob cot_normalization_type=null ans_normalization_type=null

CUDA_VISIBLE_DEVICES=3 python run_full3.py cot_reward_type=prob patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=3 python run_full3.py cot_reward_type=prob cot_normalization_type=rloo ans_normalization_type=rloo patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=4 python run_full3.py cot_reward_type=prob cot_normalization_type=null ans_normalization_type=null patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=4 python run_full3.py cot_reward_type=prob ans_reward_type=prob
CUDA_VISIBLE_DEVICES=5 python run_full3.py cot_reward_type=prob ans_reward_type=prob cot_normalization_type=rloo ans_normalization_type=rloo
CUDA_VISIBLE_DEVICES=5 python run_full3.py cot_reward_type=prob ans_reward_type=prob cot_normalization_type=null ans_normalization_type=null

CUDA_VISIBLE_DEVICES=6 python run_full3.py cot_reward_type=prob ans_reward_type=prob patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=6 python run_full3.py cot_reward_type=prob ans_reward_type=prob cot_normalization_type=rloo ans_normalization_type=rloo patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=7 python run_full3.py cot_reward_type=prob ans_reward_type=prob cot_normalization_type=null ans_normalization_type=null patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=7 python run_full3.py generations_per_prompt=1
CUDA_VISIBLE_DEVICES=2 python run_full3.py generations_per_prompt=1 cot_reward_type=prob
CUDA_VISIBLE_DEVICES=2 python run_full3.py generations_per_prompt=1 cot_reward_type=prob patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=7 python run_full3.py generations_per_prompt=1 cot_reward_type=prob ans_reward_type=prob
CUDA_VISIBLE_DEVICES=7 python run_full3.py generations_per_prompt=1 cot_reward_type=prob ans_reward_type=prob patch_in_answer_prompt=true


CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 run_full3.py max_new_tokens=256
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 run_full3.py max_new_tokens=256 cot_reward_type=prob ans_reward_type=prob patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 run_full3.py max_new_tokens=256 cot_reward_type=prob ans_reward_type=prob
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 run_full3.py max_new_tokens=256 cot_reward_type=prob  patch_in_answer_prompt=true
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes 1 --nproc_per_node 2 run_full3.py max_new_tokens=256 cot_reward_type=prob