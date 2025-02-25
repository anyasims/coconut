#!/bin/bash
## First:
# docker run --name asims1-c1 --shm-size=10.24gb --user $(id -u) -v $(pwd):/home/anya -v /scratch/local/homes/80/anya:/scratch/anya -it --gpus '"device=0,1,2,3,4,5,6,7"' asims1
# docker exec -it asims1-c1 /bin/bash
# Or if container not running:
# docker start asims1-c1

## First time:
# conda create --name coconut-1 python=3.12
# conda activate coconut-1
# cd /home/anya/Documents/llm_tiny_ideas/coconut-outer/coconut
# pip install -r requirements.txt

## git
# git remote rename origin upstream
# git remote add origin https://github.com/anyasims/coconut.git
# git branch -M main
# git remote set-url origin git@github.com:anyasims/coconut.git
# git push origin main

## other
# conda install -c anaconda ipykernel
# python -m ipykernel install --user --name=coconut-1
# often need to go to settings (click on remote settings) and set condaPath to "~/anaconda3/bin/conda"

## data
# bash preprocessing/gsm_icot.bash

# Then:
conda activate coconut-1
cd /home/anya/Documents/llm_tiny_ideas/coconut-outer/coconut
export PYTHONPATH=$PYTHONPATH:/home/anya/Documents/llm_tiny_ideas/coconut-outer/coconut
# ^Call with:
# source /home/anya/Documents/llm_tiny_ideas/coconut-outer/coconut/setup.sh


## For running:
### GSM8K
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml # stage 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml # coconut
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml # eval
### ProsQA
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml