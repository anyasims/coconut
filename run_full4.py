import os
import socket
import random
import time
import tqdm
import re
import hydra
import wandb
import gc
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import DatasetDict
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def get_model_param_stats(model, ref_model):
    with torch.no_grad():
        model_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
        ref_model_params = torch.cat([p.view(-1) for p in ref_model.parameters()])
        assert model_params.shape == ref_model_params.shape, f"{model_params.shape=} {ref_model_params.shape=}"
        metrics = {}
        metrics["params_with_grads_mean"] = model_params.mean().item()
        metrics["params_with_grads_std"] = model_params.std().item()
        metrics["distance_to_ref"] = torch.nn.functional.mse_loss(model_params, ref_model_params)
        metrics = {f"params/{k}": v for k, v in metrics.items()}
    return metrics

def careful_repeat(tensor, num_repeats):
    assert isinstance(tensor, torch.Tensor)
    batch_size = tensor.shape[0]
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1).repeat(1, num_repeats).reshape(batch_size*num_repeats, *tensor.shape[1:])
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(1).repeat(1, num_repeats, 1).reshape(batch_size*num_repeats, *tensor.shape[1:])
    else:
        raise ValueError(f"Invalid ndim: {tensor.ndim}")
    return tensor

def careful_repeat_dict(data, num_repeats):
    assert isinstance(data, dict)
    for k, v in data.items():
        data[k] = careful_repeat(v, num_repeats)
    return data

class MyDataloader:
    def __init__(self, dataset, per_device_prompt_batch_size, rank, world_size, shuffle=True):
        self.dataset = dataset
        self.per_device_prompt_batch_size = per_device_prompt_batch_size
        self.rank = rank
        self.world_size = world_size
        self.dataset_size = len(dataset)
        self.indices = list(range(self.dataset_size))
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.indices)
        self.subbatch_start = 0
        self.epoch = 0

    def reset(self):
        self.subbatch_start = 0
        self.epoch = 0
        if self.shuffle:
            random.shuffle(self.indices)

    def __next__(self):
        if self.subbatch_start + self.per_device_prompt_batch_size * self.world_size >= self.dataset_size:
            self.epoch += 1
            if self.shuffle:
                random.shuffle(self.indices)
            self.subbatch_start = 0
        start = self.subbatch_start + self.rank * self.per_device_prompt_batch_size 
        end = start + self.per_device_prompt_batch_size
        subbatch_indices = self.indices[start:end]
        self.subbatch_start += self.per_device_prompt_batch_size * self.world_size
        return self.dataset.select(subbatch_indices), self.epoch
    

class Trainer:
    def __init__(self, cfg) -> None:
        assert torch.cuda.is_available(), "CUDA not available"
        self.using_ddp = dist.is_initialized()
        if self.using_ddp:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.world_size = torch.cuda.device_count()
            assert self.world_size == 1, f"Expected single GPU, {self.world_size=}"
            self.rank = 0
            self.local_rank = None
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # config
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        self.set_config(cfg)

        # model
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model).to(self.device)
        ref_model = AutoModelForCausalLM.from_pretrained(cfg.base_model).to(self.device)
        self.model = DDP(model, device_ids=[self.local_rank]) if self.using_ddp else model
        self.ref_model = DDP(ref_model, device_ids=[self.local_rank]) if self.using_ddp else ref_model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.vocab_size = model.config.vocab_size

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.lr),
            betas=cfg.get('betas', (0.9, 0.999)),
            weight_decay=cfg.get('weight_decay', 1e-2))

        # dataset
        dataset = DatasetDict.load_from_disk(f"./data/my_data/{cfg.dataset}")
        if cfg.dataset.startswith("gsm8k"):
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
            self.do_eval = True
        elif cfg.dataset.startswith("svamp"):
            train_dataset = dataset["test"]
            self.do_eval = False
        else:
            raise ValueError(f"{cfg.dataset=}")

        # dataloader
        self.train_loader = MyDataloader(train_dataset, self.per_device_prompt_batch_size, self.rank, self.world_size)
        self.iters_per_epoch = len(train_dataset) // self.total_batch_size
        if self.do_eval:
            self.eval_loader = MyDataloader(eval_dataset, self.eval_per_device_batch_size, self.rank, self.world_size, shuffle=False)
            assert len(eval_dataset) >= self.eval_num_samples, f"{len(eval_dataset)=} {self.eval_num_samples=}"

        if self.rank == 0:
            print(f"Dataset: {cfg.dataset}")
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Iters per epoch: {self.iters_per_epoch}")
            print(f"Eval dataset size: {len(eval_dataset)}\n\n" if eval_dataset is not None else "\n\n")
        assert cfg.dataset.endswith("_hash"), f"{cfg.dataset=}"

        # answer prompt
        answer_prompt_id = self.tokenizer.encode("####")
        assert len(answer_prompt_id) == 1, f"{answer_prompt_id=}"
        self.answer_prompt_id = answer_prompt_id[0]
        space_id = self.tokenizer.encode(" ")
        assert len(space_id) == 1, f"{space_id=}"
        self.space_id = space_id[0]
        dot_dot_dot_id = self.tokenizer.encode("...")
        assert len(dot_dot_dot_id) == 1, f"{dot_dot_dot_id=}"
        self.dot_dot_dot_id = dot_dot_dot_id[0]

        self.ctx = self._setup_ctx()

    def set_config(self, cfg):
        if self.rank == 0:
            print(f"\nTrainer::-----------------------------------")
        self.cfg = cfg
        self.use_wandb = cfg.use_wandb
        self.seed = cfg.seed

        ### training
        self.max_iters = cfg.max_iters
        self.total_batch_size = cfg.total_batch_size
        self.per_device_batch_size = cfg.per_device_batch_size
        self.gradient_accumulation_steps = self.total_batch_size // (self.per_device_batch_size * self.world_size)
        self.generations_per_prompt = cfg.generations_per_prompt
        self.total_prompt_batch_size = self.total_batch_size // self.generations_per_prompt
        self.per_device_prompt_batch_size = self.per_device_batch_size // self.generations_per_prompt
        assert self.per_device_batch_size * self.world_size * self.gradient_accumulation_steps == self.total_batch_size
        assert self.per_device_prompt_batch_size * self.world_size * self.gradient_accumulation_steps == self.total_prompt_batch_size
        self.dataset_size = cfg.dataset_size

        ### eval
        self.eval_freq = cfg.eval_freq
        self.eval_num_samples = cfg.eval_num_samples
        self.eval_per_device_batch_size = self.per_device_batch_size # * 2
        # self.eval_per_device_batch_size = 1
        self.eval_num_subbatches = self.eval_num_samples // (self.eval_per_device_batch_size * self.world_size) # like accumulation steps
        assert self.eval_per_device_batch_size * self.world_size * self.eval_num_subbatches == self.eval_num_samples

        # generation
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.patch_in_answer_prompt = cfg.patch_in_answer_prompt

        # reward
        self.cot_reward_type = cfg.cot_reward_type
        self.ans_reward_type = cfg.ans_reward_type
        self.cot_normalization_type = cfg.cot_normalization_type if self.generations_per_prompt != 1 and self.cot_reward_type is not None else None
        self.ans_normalization_type = cfg.ans_normalization_type if self.generations_per_prompt != 1 and self.ans_reward_type is not None else None
        assert self.cot_reward_type in [None, "binary", "prob"], f"{self.cot_reward_type=}"
        assert self.ans_reward_type in [None, "binary", "prob"], f"{self.ans_reward_type=}"
        assert self.cot_normalization_type in [None, "grpo", "rloo"], f"{self.cot_normalization_type=}"
        assert self.ans_normalization_type in [None, "grpo", "rloo"], f"{self.ans_normalization_type=}"
        
        # loss
        self.kl_type = cfg.kl_type
        self.kl_coef = cfg.kl_coef
        self.entropy_coef = cfg.entropy_coef

        if self.rank == 0:
            
            # run name
            run_name = f"{cfg.run_name_prefix}-" if cfg.run_name_prefix != "" else ""
            
            # reward
            run_name += f"-COT{self.cot_reward_type}" if self.cot_reward_type is not None else "-noCOT"
            run_name += f"_{self.cot_normalization_type}" if self.cot_normalization_type is not None else ""
            run_name += f"-ANS{self.ans_reward_type}" if self.ans_reward_type is not None else ""
            run_name += f"_{self.ans_normalization_type}" if self.ans_normalization_type is not None else ""
            run_name += f"-g{self.generations_per_prompt}"
            # generation
            run_name += f"-PatchAP" if self.patch_in_answer_prompt else ""
            run_name += f"-L{self.max_new_tokens}"
            run_name += f"-T{self.temperature}" if self.temperature != 1.0 else ""
            # training
            run_name += f"--B{self.total_batch_size}"
            run_name += f"-D{self.dataset_size}" if self.dataset_size is not None else ""
            run_name += f"-lr{cfg.lr:.0e}"
            # loss
            run_name += f"-kl_{self.kl_type}{self.kl_coef}" if self.kl_coef != 0.0 else ""
            run_name += f"-ent{self.entropy_coef}" if self.entropy_coef != 0.0 else ""
            # model
            short_model_name = f"Qw{cfg.base_model.split("/")[-1].split("-")[1]}" if "Qwen" in cfg.base_model else cfg.base_model.split("/")[-1]
            run_name += f"-{short_model_name}"
            run_name += f"--seed{self.seed}" if self.seed != 0 else ""
            # other
            run_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
            run_name += f"--{run_id}"
            self.run_name = run_name

            if self.use_wandb:
                node_name = socket.gethostname()
                cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                wandb_cfg = OmegaConf.to_container(cfg)
                wandb_cfg["node_name"] = node_name
                wandb_cfg["cuda_visible_devices"] = cuda_visible_devices

                wandb.init(
                    project=cfg.wandb_project,
                    config=wandb_cfg,
                    name=run_name,
                )
                print("WandB logging initialized")
                print(OmegaConf.to_yaml(cfg), "\n")

            print(f"---SETUP:")
            print(f"Using DDP: {self.using_ddp}")
            print(f"Device: {self.device}")
            print(f"World size: {self.world_size}")
            print(f"Rank: {self.rank}")
            print(f"Local rank: {self.local_rank}")
            print(f"Device: {self.device}")
            print(f"Run name: {self.run_name}")
            print(f"-----------------------------------\n")
            print(f"---TRAINING CONFIG:")
            print(f"Max iters: {self.max_iters}")
            print(f"Total batch size: {self.total_batch_size}")
            print(f"Total prompt batch size: {self.total_prompt_batch_size}")
            print(f"Per device batch size: {self.per_device_batch_size}")
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"Generations per prompt: {self.generations_per_prompt}")
            print(f"Per device prompt batch size: {self.per_device_prompt_batch_size}")
            print(f"Dataset size: {self.dataset_size} (smaller for debugging)")
            print(f"Lr: {cfg.lr}")
            print(f"-----------------------------------\n")
            print(f"---EVAL CONFIG:")
            print(f"Eval freq: {self.eval_freq}")
            print(f"Eval total batch size: {self.eval_num_samples}")
            print(f"Eval per device batch size: {self.eval_per_device_batch_size}")
            print(f"Eval num subbatches: {self.eval_num_subbatches}")
            print(f"-----------------------------------\n")
            print(f"---GENERATION CONFIG:")
            print(f"Max new tokens: {self.max_new_tokens}")
            print(f"Temperature: {self.temperature}")
            print(f"-----------------------------------\n")
            print(f"---REWARD CONFIG:")
            print(f"Generations per prompt: {self.generations_per_prompt}")
            print(f"Cot reward type: {self.cot_reward_type}")
            print(f"Ans reward type: {self.ans_reward_type}")
            print(f"Cot normalization type: {self.cot_normalization_type}")
            print(f"Ans normalization type: {self.ans_normalization_type}")
            print(f"-----------------------------------\n")
            print(f"---LOSS CONFIG:")
            print(f"KL type: {self.kl_type}")
            print(f"KL coef: {self.kl_coef}")
            print(f"Entropy coef: {self.entropy_coef}")
            print(f"-----------------------------------\n")


    def _setup_ctx(self):
        """Get the context manager"""
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self._setup_scaler(dtype)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        return ctx

    def _setup_scaler(self, dtype=torch.float16):
        """Setup the scaler"""
        self.scaler = torch.amp.GradScaler(enabled=dtype == torch.float16, device='cuda')

    def apply_update(self):
        grad_clip = 1.0
        # once gradients are accumulated, step 
        if grad_clip > 0:
            # Unscale the gradients of the optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Clip the gradients with normalization
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        # Perform a single optimization step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()  # Reset gradients after update

    def get_loss(self, x):
        cot_logp, ans_logp, kl, entropy, cot_normalized_reward, ans_normalized_reward = x
        assert cot_logp.shape == ans_logp.shape == kl.shape == entropy.shape == cot_normalized_reward.shape == ans_normalized_reward.shape == (self.per_device_batch_size,), f"{cot_logp.shape=}, {ans_logp.shape=}, {kl.shape=}, {entropy.shape=}, {cot_normalized_reward.shape=}, {ans_normalized_reward.shape=}"

        cot_pg_loss = - cot_logp * cot_normalized_reward
        ans_pg_loss = - ans_logp * ans_normalized_reward

        loss = cot_pg_loss + ans_pg_loss + self.kl_coef * kl + self.entropy_coef * entropy

        with torch.no_grad():
            metrics = {}
            metrics["loss"] = loss.mean()
            metrics["cot_pg_loss"] = cot_pg_loss.mean()
            metrics["ans_pg_loss"] = ans_pg_loss.mean()
            metrics["kl"] = kl.mean()
            metrics["entropy"] = entropy.mean()
            metrics = {f"loss/{k}": v for k, v in metrics.items()}

        return loss.mean() / self.gradient_accumulation_steps, metrics
    
    @torch.no_grad()
    def normalize_reward(self, reward, normalization_type):
        if self.generations_per_prompt > 1:
            reward = reward.reshape(self.per_device_prompt_batch_size, self.generations_per_prompt)
            if normalization_type == "grpo":
                normalized_reward = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
            elif normalization_type == "rloo":
                group_sum = reward.sum(1, keepdim=True)
                normalized_reward = (group_sum - reward) / (self.generations_per_prompt - 1)
            elif normalization_type is None:
                normalized_reward = reward
            else:
                raise ValueError(f"{normalization_type=}")
            reward = reward.reshape(-1)
            normalized_reward = normalized_reward.reshape(-1)
        else:
            assert normalization_type is None, f"{normalization_type=}"
            normalized_reward = reward
        return normalized_reward

    def generate(
        self,
        dataset_batch,
        generations_per_prompt,
        is_eval=False,
    ):
        model = self.model.module if self.using_ddp else self.model
        questions_text = dataset_batch["question"]
        answer_text = dataset_batch["answer"]
        questions_inputs = self.tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
        questions_inputs = {k: v.to(self.model.device) for k, v in questions_inputs.items()}
        tokenized_answers = self.tokenizer(answer_text).input_ids
        # tokenized_answers = [[self.space_id] + a + [self.tokenizer.eos_token_id] for a in tokenized_answers]
        tokenized_answers = [[self.space_id] + a + [self.space_id] for a in tokenized_answers]

        # shapes
        batch_size = questions_inputs["input_ids"].shape[0]
        full_batch_size = batch_size * generations_per_prompt
        question_length = questions_inputs["input_ids"].shape[1]
        # repeat
        questions_inputs = careful_repeat_dict(questions_inputs, generations_per_prompt)
        tokenized_answers = [a for a in tokenized_answers for _ in range(generations_per_prompt)]
        # generate responses
        with torch.no_grad():
            q_responses_ids = model.generate(
                    input_ids=questions_inputs["input_ids"],
                    attention_mask=questions_inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=None,
                    do_sample=True,
                    pad_token_id=model.config.eos_token_id,
                    )
        
        # # check
        # check_texts = tokenizer.batch_decode(q_responses_ids, skip_special_tokens=False)
        # check_contains_ap = ["####" in t for t in check_texts]
        # check_contains_ap_id = [self.answer_prompt_id in t for t in q_responses_ids.tolist()]
        # for i in range(full_batch_size):
        #     if check_contains_ap[i] != check_contains_ap_id[i]:
        #         print(f"{check_contains_ap[i]=}, {check_contains_ap_id[i]=}")
        #         print(f"{q_responses_ids[i]=}")
        #         print(f"{[tokenizer.decode([t]) for t in q_responses_ids[i]]}")

        # print(self.tokenizer.decode(q_responses_ids[0].tolist(), skip_special_tokens=False), "\n", "-"*50)
        # make patched tensors
        patched_q_responses_ids = q_responses_ids.clone()
        reponse_mask = torch.zeros_like(q_responses_ids, dtype=torch.float)
        patched_cot_mask = torch.zeros_like(patched_q_responses_ids, dtype=torch.float)
        patched_answer_mask = torch.zeros_like(patched_q_responses_ids, dtype=torch.float)
        reponse_mask[:, question_length:] = 1
        patched_cot_mask[:, question_length:] = 1
        # make contains tensors
        contains_eot = torch.zeros(full_batch_size, device=self.device)
        contains_answer_prompt = torch.zeros(full_batch_size, device=self.device)
        contains_answer = torch.zeros(full_batch_size, device=self.device)
        # fill in patched tensors and contains
        responses_ids = q_responses_ids[:, question_length:]
        for i, (response_ids, answer_ids) in enumerate(zip(responses_ids, tokenized_answers)):
            answer_length = len(answer_ids)
            contains_eot[i] = (response_ids == self.tokenizer.eos_token_id).any()
            if contains_eot[i] == 1:
                response_eot_idx = (response_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                eot_idx = response_eot_idx + question_length
                reponse_mask[i, eot_idx+1:] = 0
                patched_cot_mask[i, eot_idx+1:] = 0
                response_ids[response_eot_idx+1:] = self.tokenizer.pad_token_id
            contains_answer_prompt[i] = (response_ids[:-answer_length-1] == self.answer_prompt_id).any()
            if contains_answer_prompt[i] == 1:
                response_ap_idx = (response_ids == self.answer_prompt_id).nonzero(as_tuple=True)[0][0]
                ap_idx = question_length + response_ap_idx
                assert response_ap_idx + 1 + answer_length < len(response_ids), f"{response_ap_idx=} {answer_length=} {len(response_ids)=}"
                answer_ids = torch.tensor(answer_ids, device=self.device)
                contains_answer_i = (response_ids[response_ap_idx+1:response_ap_idx+1+answer_length] == answer_ids).all()
                contains_answer[i] = contains_answer_i
                patched_q_responses_ids[i, ap_idx+1:ap_idx+1+answer_length] = answer_ids
                patched_answer_mask[i, ap_idx+1:ap_idx+1+answer_length] = 1
                patched_cot_mask[i, ap_idx+1:] = 0
            elif self.patch_in_answer_prompt:
                if contains_eot[i] == 1:
                    patch_ids = [self.space_id, self.answer_prompt_id] + answer_ids
                else:
                    patch_ids = [self.space_id, self.dot_dot_dot_id, self.space_id, self.answer_prompt_id] + answer_ids
                patch_start_idx = patched_q_responses_ids.shape[1] - len(patch_ids)
                if contains_eot[i] == 1:
                    patch_start_idx = min(patch_start_idx, eot_idx)
                    answer_start_idx = eot_idx + 2 # for [space, answer_prompt]
                else:
                    answer_start_idx = patch_start_idx + 4 # for [space, dot_dot_dot, space, answer_prompt]
                patch_length = len(patch_ids)
                answer_length = len(answer_ids)
                patch_ids = torch.tensor(patch_ids, device=self.device)
                patched_q_responses_ids[i, patch_start_idx:patch_start_idx+patch_length] = patch_ids
                patched_answer_mask[i, answer_start_idx:answer_start_idx+answer_length] = 1
                patched_cot_mask[i, answer_start_idx:] = 0

        # print decoded
        # print(f"{tokenizer.decode(patched_q_responses_ids[0].tolist(), skip_special_tokens=False)}")

        # check
        patched_mask_check = patched_cot_mask + patched_answer_mask
        assert patched_mask_check[:, :question_length].sum() == 0
        assert torch.all((patched_mask_check == 0) | (patched_mask_check == 1)), f"Should be only zeros or ones. {patched_mask_check=}, {patched_cot_mask=}, {patched_answer_mask=}"
        assert torch.all(patched_mask_check[:, question_length] == 1), f"Should start with 1. {patched_mask_check=}, {patched_cot_mask=}, {patched_answer_mask=}"

        # prepare inputs
        forward_with_patched = self.cot_reward_type == "prob" or self.ans_reward_type == "prob"
        if forward_with_patched:
            input_ids = patched_q_responses_ids
            mask = torch.logical_or(patched_cot_mask, patched_answer_mask).float()[:, 1:]
        else:
            input_ids = q_responses_ids
            mask = reponse_mask[:, 1:]
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, :question_length] = questions_inputs["attention_mask"]
        # forward pass
        all_logits = self.model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask,
            ).logits
        # ref forward pass
        with torch.no_grad():
            ref_all_logits = self.ref_model(
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask,
                ).logits
        assert not torch.isnan(all_logits).any(), f"{all_logits=}"
        assert not torch.isinf(all_logits).any(), f"{all_logits=}"
        assert not torch.isnan(ref_all_logits).any(), f"{ref_all_logits=}"
        assert not torch.isinf(ref_all_logits).any(), f"{ref_all_logits=}"

        # logps
        all_logps = torch.nn.functional.log_softmax(all_logits, dim=-1)
        ref_all_logps = torch.nn.functional.log_softmax(ref_all_logits, dim=-1)
        per_token_logps = torch.gather(all_logps, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        ref_per_token_logps = torch.gather(ref_all_logps, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        assert not torch.isnan(per_token_logps).any(), f"{per_token_logps=}"
        assert not torch.isinf(per_token_logps).any(), f"{per_token_logps=}"
        assert not torch.isnan(ref_per_token_logps).any(), f"{ref_per_token_logps=}"
        assert not torch.isinf(ref_per_token_logps).any(), f"{ref_per_token_logps=}"
        # entropy
        entropy = torch.logsumexp(all_logits, dim=-1) - torch.sum(torch.nn.functional.softmax(all_logits, dim=-1) * all_logits, dim=-1)
        ref_entropy = torch.logsumexp(ref_all_logits, dim=-1) - torch.sum(torch.nn.functional.softmax(ref_all_logits, dim=-1) * ref_all_logits, dim=-1)
        # kl
        sample_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        full_kl = (all_logps.exp() * (all_logps - ref_all_logps)).sum(-1)
        # mask and mean over length
        entropy = (entropy * mask).sum(dim=-1)
        ref_entropy = (ref_entropy * mask).sum(dim=-1)
        sample_kl = (sample_kl * mask).sum(dim=-1)
        full_kl = (full_kl * mask).sum(dim=-1)
        # logp
        cot_logp = (per_token_logps * patched_cot_mask[:, 1:]).sum(dim=-1)
        ans_logp = (per_token_logps * patched_answer_mask[:, 1:]).sum(dim=-1)
        cot_ref_logp = (ref_per_token_logps * patched_cot_mask[:, 1:]).sum(dim=-1)
        ans_ref_logp = (ref_per_token_logps * patched_answer_mask[:, 1:]).sum(dim=-1)
        # kl
        if self.kl_type == "sample":
            kl = sample_kl
        elif self.kl_type == "full":
            kl = full_kl
        else:
            raise ValueError(f"{self.kl_type=}")
        # reward
        with torch.no_grad():
            answer_mask_lengths = patched_answer_mask[:, 1:].sum(dim=-1)
            answer_mask_lengths[answer_mask_lengths == 0] = 1 # avoid division by zero
            answer_p_length_normalized = torch.exp(ans_logp / answer_mask_lengths.float())
            answer_p_length_normalized *= (answer_mask_lengths > 1).float()
            ref_answer_p_length_normalized = torch.exp(ans_ref_logp / answer_mask_lengths.float()) # for logging
            # cot reward
            if self.cot_reward_type == "binary":
                cot_reward = contains_answer.float()
            elif self.cot_reward_type == "prob":
                cot_reward = answer_p_length_normalized
            elif self.cot_reward_type is None:
                cot_reward = torch.zeros(full_batch_size, device=self.device, dtype=torch.float)
            else:
                raise ValueError(f"{self.cot_reward_type=}")
            # ans reward
            if self.ans_reward_type == "binary":
                ans_reward = torch.ones_like(cot_reward) if self.cot_reward_type == "prob" else contains_answer.float()
            elif self.ans_reward_type == "prob":
                ans_reward = answer_p_length_normalized
            elif self.ans_reward_type is None:
                ans_reward = torch.zeros(full_batch_size, device=self.device, dtype=torch.float)
            else:
                raise ValueError(f"{self.ans_reward_type=}")
            assert not torch.isnan(cot_reward).any(), f"{cot_reward=}"
            assert not torch.isinf(cot_reward).any(), f"{cot_reward=}"
            assert not torch.isnan(ans_reward).any(), f"{ans_reward=}"
            assert not torch.isinf(ans_reward).any(), f"{ans_reward=}"
            # normalize reward
            if not is_eval:
                cot_normalized_reward = self.normalize_reward(cot_reward, self.cot_normalization_type)
                ans_normalized_reward = self.normalize_reward(ans_reward, self.ans_normalization_type)
                assert not torch.isnan(cot_normalized_reward).any(), f"{cot_normalized_reward=}"
                assert not torch.isinf(cot_normalized_reward).any(), f"{cot_normalized_reward=}"
                assert not torch.isnan(ans_normalized_reward).any(), f"{ans_normalized_reward=}"
                assert not torch.isinf(ans_normalized_reward).any(), f"{ans_normalized_reward=}"

        # collect decoded generations
        decoded_generations = []
        for i in range(full_batch_size):
            reponse_start = question_length
            reponse_end = (reponse_mask[i] == 1).nonzero(as_tuple=True)[0][-1] + 1
            response = self.tokenizer.decode(q_responses_ids[i, reponse_start:reponse_end+1].tolist(), skip_special_tokens=False)
            num_padding = (q_responses_ids.shape[1] - (reponse_end+1)).item()
            response += f"*{str(num_padding)}" if num_padding > 0 else ""
            if contains_answer_prompt[i]:
                ap_idx = (patched_cot_mask[i] == 1).nonzero(as_tuple=True)[0][-1]
                generated_answer = [self.tokenizer.decode(q_responses_ids[i, ap_idx+1:ap_idx+1+len(tokenized_answers[i])].tolist(), skip_special_tokens=False)]
                logits_ap_idx = ap_idx - 1
                all_answer_logps = all_logps[i, logits_ap_idx+1:logits_ap_idx+1+len(tokenized_answers[i])]
                argmax_answer = [self.tokenizer.decode(all_answer_logps.argmax(-1).tolist(), skip_special_tokens=False)]
            else:
                generated_answer = None
                argmax_answer = None
            decoded_generations.append((response,
                                generated_answer,
                                argmax_answer,
                                contains_answer_prompt[i].item() == 1,
                                contains_answer[i].item() == 1,
                                answer_p_length_normalized[i].item() if forward_with_patched else None,
                                patched_cot_mask.sum(dim=-1)[i].item(), # length of cot
                                cot_normalized_reward[i].item() if not is_eval else None,
                                ans_normalized_reward[i].item() if not is_eval else None,
            ))

        with torch.no_grad():
            metrics = {}
            metrics["contains_answer_prompt"] = contains_answer_prompt.mean()
            metrics["contains_answer"] = contains_answer.mean()
            metrics["length_to_eot"] = reponse_mask.sum(dim=-1).mean()
            metrics["length_of_cot"] = patched_cot_mask.sum(dim=-1).mean()
            metrics["entropy"] = entropy.mean()
            metrics["ref_entropy"] = ref_entropy.mean()
            metrics["sample_kl"] = sample_kl.mean()
            metrics["full_kl"] = full_kl.mean()
            metrics["kl"] = kl.mean()
            metrics["cot_reward"] = cot_reward.mean()
            metrics["ans_reward"] = ans_reward.mean()
            if not is_eval:
                metrics["cot_normalized_reward_max"] = cot_normalized_reward.max()
                metrics["cot_normalized_reward_min"] = cot_normalized_reward.min()
                metrics["ans_normalized_reward_max"] = ans_normalized_reward.max()
                metrics["ans_normalized_reward_min"] = ans_normalized_reward.min()
            metrics["ans_logp"] = ans_logp.mean()
            metrics["cot_logp"] = cot_logp.mean()
            metrics["ans_ref_logp"] = ans_ref_logp.mean()
            metrics["cot_ref_logp"] = cot_ref_logp.mean()
            metrics["answer_p_length_normalized"] = answer_p_length_normalized.mean()
            metrics["ref_answer_p_length_normalized"] = ref_answer_p_length_normalized.mean()
            metrics = {f"gen/{k}": v for k, v in metrics.items()}
        
        if is_eval:
            x = None
        else:
            x = (cot_logp, ans_logp, kl, entropy, cot_normalized_reward, ans_normalized_reward)
            assert cot_logp.requires_grad == True, f"{cot_logp.requires_grad=}"
            assert ans_logp.requires_grad == True, f"{ans_logp.requires_grad=}"
            assert entropy.requires_grad == True, f"{entropy.requires_grad=}"
            assert kl.requires_grad == True, f"{kl.requires_grad=}"
            assert cot_normalized_reward.requires_grad == False, f"{cot_normalized_reward.requires_grad=}"
            assert ans_normalized_reward.requires_grad == False, f"{ans_normalized_reward.requires_grad=}"
        return x, decoded_generations, metrics
    

    def print_metrics(self,
                      decoded_generations,
                      questions_text,
                      answers_text,
                      gen_time=None,
                      loss_time=None,
                      prefix=""
                      ):
        response, ans_gen, ans_argmax, contains_ap, contains_ans, ans_prob, length, cot_r, ans_r = decoded_generations[0]
        peak_mem_allocated = torch.cuda.max_memory_allocated() // 1024 // 1024
        peak_mem_reserved = torch.cuda.max_memory_reserved() // 1024 // 1024
        # print
        print("-"*50)
        if gen_time is not None and loss_time is not None:
            print(f"Generation time: {gen_time:.1f}s")
            print(f"Loss time: {loss_time:.1f}s")
        print(f"peak memory allocated: {peak_mem_allocated} MiB, reserved: {peak_mem_reserved} MiB")
        print("-"*50)
        print(f"              QUESTION: {questions_text[0]}")
        print(f"              RESPONSE: {response}")
        print(f"CONTAINS ANSWER PROMPT: {contains_ap}")
        print(f"       CONTAINS ANSWER: {contains_ans}")
        print(f"      GENERATED ANSWER: {ans_gen}")
        print(f"         ARGMAX ANSWER: {ans_argmax}")
        print(f"        CORRECT ANSWER: {answers_text[0]}")
        if self.ans_reward_type == "prob" or self.cot_reward_type == "prob":
            print(f"           ANSWER PROB: {ans_prob:.2e}")
        print(f"                LENGTH: {length}")
        if cot_r is not None and ans_r is not None:
            print(f" COT NORMALIZED REWARD: {cot_r:.2e}")
            print(f" ANS NORMALIZED REWARD: {ans_r:.2e}")
        print("-"*50)
        print(f"{prefix}")
        print("\n")
        del response, ans_gen, ans_argmax, contains_ap, contains_ans, ans_prob, length
        gc.collect()
        torch.cuda.empty_cache()
    

    def run_training_loop(self, num_iters=None):
        """Run the training loop"""
        train_start_time = time.time()
        num_iters = self.max_iters if num_iters is None else num_iters
        
        for i in tqdm.tqdm(range(num_iters+1), desc="Train time", total=num_iters):

            # EVAL
            if (i == 0 or i == num_iters or i % self.eval_freq == 0) and self.do_eval:
                self.eval_loader.reset()
                for j in tqdm.tqdm(range(self.eval_num_subbatches), desc="Eval subbatches", disable=True):
                    with torch.no_grad():
                        # GENERATE
                        dataset_batch, epoch = next(self.eval_loader)
                        _, decoded_generations, generation_metrics = self.generate(dataset_batch, generations_per_prompt=1, is_eval=True)

                    # METRICS
                    if j == 0:
                        metrics_s = {**generation_metrics}
                    else:
                        metrics_s = {k: v + generation_metrics[k] for k, v in metrics_s.items()}

                    # PRINT
                    if self.rank == 0 and j % (self.eval_num_subbatches // 2) == 0:
                        prefix = f"EVAL: Iter {i+1}/{num_iters}, subbatch {j+1}/{self.eval_num_subbatches}, epoch {epoch+1}, "
                        prefix += f"mean contains answer: {metrics_s['gen/contains_answer']/(j+1):.2f}"
                        self.print_metrics(decoded_generations, dataset_batch["question"], dataset_batch["answer"], prefix=prefix)

                # LOG
                metrics_s = {k: v / self.eval_num_subbatches for k, v in metrics_s.items()}
                metrics_s = {f"eval_{k}": v for k, v in metrics_s.items()}
                if self.using_ddp:
                    for k, v in metrics_s.items():
                        dist.all_reduce(v, op=dist.ReduceOp.AVG)
                if self.use_wandb and self.rank == 0:
                    wandb.log({**metrics_s, "iter": i})

                # CLEANUP
                del dataset_batch, generation_metrics, decoded_generations
                gc.collect()
                torch.cuda.empty_cache()

            # TRAIN
            if i < num_iters:
                for j in tqdm.tqdm(range(self.gradient_accumulation_steps), desc="Gradient accumulation", disable=True):
                    # GENERATE AND FORWARD
                    start_time = time.time()
                    dataset_batch, epoch = next(self.train_loader)
                    x, decoded_generations, generation_metrics = self.generate(dataset_batch, generations_per_prompt=self.generations_per_prompt, is_eval=False)
                    gen_time = time.time()-start_time

                    # LOSS AND UPDATE
                    start_time = time.time()
                    with self.ctx: 
                        loss, loss_metrics = self.get_loss(x)
                        self.scaler.scale(loss).backward()
                    loss_time = time.time()-start_time

                    # METRICS
                    generation_metrics["time/gen_time"] = torch.tensor(gen_time, device=self.device)
                    loss_metrics["time/loss_time"] = torch.tensor(loss_time, device=self.device)
                    if j == 0:
                        metrics_s = {**generation_metrics, **loss_metrics}
                    else:
                        metrics = {**generation_metrics, **loss_metrics}
                        metrics_s = {k: v + metrics[k] for k, v in metrics_s.items()}

                    # PRINT
                    if self.rank == 0 and j % (self.gradient_accumulation_steps // 2) == 0:
                        prefix = f"TRAIN: Iter {i+1}/{num_iters}, acc step {j+1}/{self.gradient_accumulation_steps}, epoch {epoch+1}, "
                        prefix += f"mean contains answer: {metrics_s['gen/contains_answer']/(j+1):.2f}"
                        self.print_metrics(decoded_generations, dataset_batch["question"], dataset_batch["answer"], gen_time, loss_time, prefix=prefix)

                    # CLEANUP
                    del dataset_batch, generation_metrics, decoded_generations
                    gc.collect()
                    torch.cuda.empty_cache()

                # UPDATE MODEL
                self.apply_update()

                # LOG
                metrics_s = {k: v / self.gradient_accumulation_steps for k, v in metrics_s.items()}
                if self.using_ddp:
                    for k, v in metrics_s.items():
                        dist.all_reduce(v, op=dist.ReduceOp.AVG)
                param_metrics = get_model_param_stats(self.model, self.ref_model)
                metrics_s.update({"iter": i, "lr": self.optimizer.param_groups[0]["lr"]})
                if self.use_wandb and self.rank == 0:
                    wandb.log(metrics_s)
                del metrics_s, param_metrics


        if self.rank == 0:
            print(f"Training time: {time.time()-train_start_time:.1f}s")
            print("Training complete")
            wandb.finish()


    def train(self, num_iters=None):
        """Train the model"""
        # set seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.run_training_loop(num_iters=num_iters)


@hydra.main(config_path="args", config_name="full_test3", version_base=None)
def main(cfg):

    if "LOCAL_RANK" in os.environ:
        print("Running with torchrun (distributed mode)")
        dist.init_process_group(backend="nccl")
    trainer = Trainer(cfg)
    trainer.train()
    if "LOCAL_RANK" in os.environ:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=0 python run_full_single2.py use_wandb=False