## For running:
### GSM8K
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml # stage 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml # coconut
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml # eval
### ProsQA
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml