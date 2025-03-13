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

def get_model_param_stats(model, ref_model):
    model_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
    ref_model_params = torch.cat([p.view(-1) for p in ref_model.parameters()])
    assert model_params.shape == ref_model_params.shape, f"{model_params.shape=} {ref_model_params.shape=}"
    return {
        "params_with_grads_mean": model_params.mean().item(),
        "params_with_grads_std": model_params.std().item(),
        "distance_to_ref": torch.nn.functional.mse_loss(model_params, ref_model_params),
    }

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

class Trainer:
    def __init__(self, cfg) -> None:
        self.world_size = torch.cuda.device_count()
        self.rank = 0 # dummy for now
        self.local_rank = None
        assert torch.cuda.is_available(), "CUDA not available"
        assert self.world_size == 1, "Only single GPU training is supported"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # config
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        self.set_config(cfg)

        # model
        model = AutoModelForCausalLM.from_pretrained(cfg.base_model).to(self.device)
        ref_model = AutoModelForCausalLM.from_pretrained(cfg.base_model).to(self.device)
        self.model=model
        self.ref_model=ref_model
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
        split_name = "train" if "train" in dataset.keys() else list(dataset.keys())[0]
        train_dataset = dataset[split_name]
        if self.dataset_size is not None:
            train_dataset = train_dataset.select(range(self.dataset_size))
        if self.rank == 0:
            print(f"Dataset: {cfg.dataset}")
            print(f"Split: {split_name}")
            print(f"Dataset size: {len(train_dataset)}")
        self.train_dataset = train_dataset
        assert cfg.dataset.endswith("_hash"), f"{cfg.dataset=}"
        answer_prompt_id = self.tokenizer.encode("####")
        assert len(answer_prompt_id) == 1, f"{answer_prompt_id=}"
        self.answer_prompt_id = answer_prompt_id[0]

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
        assert self.total_batch_size % (self.per_device_batch_size * self.world_size) == 0, f"{self.total_batch_size=} {self.per_device_batch_size=}, {self.world_size=}"
        self.gradient_accumulation_steps = self.total_batch_size // (self.per_device_batch_size * self.world_size)
        assert self.per_device_batch_size * self.world_size * self.gradient_accumulation_steps == self.total_batch_size, f"{self.per_device_batch_size=} {self.world_size=} {self.gradient_accumulation_steps=} {self.total_batch_size=}"
        self.generations_per_prompt = cfg.generations_per_prompt
        assert self.per_device_batch_size % self.generations_per_prompt == 0, f"{self.per_device_batch_size=} {self.generations_per_prompt=}"
        self.per_device_prompt_batch_size = self.per_device_batch_size // self.generations_per_prompt
        self.dataset_size = cfg.dataset_size

        # generation
        self.max_new_tokens = cfg.max_new_tokens
        self.normalization_type = cfg.normalization_type if self.generations_per_prompt != 1 else None
        self.temperature = cfg.temperature
        # reward
        self.reward_type = cfg.reward_type
        assert self.normalization_type in [None, "grpo", "rloo"], f"{self.normalization_type=}"
        assert self.reward_type in ["binary", "prob"], f"{self.reward_type=}"
        # loss
        self.kl_type = cfg.kl_type
        self.kl_coef = cfg.kl_coef
        self.entropy_coef = cfg.entropy_coef

        if self.rank == 0:
            
            # run name
            run_name = f"{cfg.run_name_prefix}-" if cfg.run_name_prefix != "" else ""
            
            # generation
            run_name += f"-L{self.max_new_tokens}"
            run_name += f"-g{self.generations_per_prompt}" if self.generations_per_prompt != 1 else ""
            run_name += f"-{self.normalization_type}" if self.normalization_type != None else ""
            run_name += f"-T{self.temperature}" if self.temperature != 1.0 else ""
            # reward
            run_name += f"--R_{self.reward_type}"
            # training
            run_name += f"--B{self.total_batch_size}"
            run_name += f"-D{self.dataset_size}" if self.dataset_size is not None else ""
            run_name += f"-lr{cfg.lr:.0e}"
            # loss
            if self.kl_coef > 0:
                run_name += f"-kl_full" if self.kl_type == "full" else ""
                run_name += f"{self.kl_coef}"
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

            print(f"World size: {self.world_size}")
            print(f"Rank: {self.rank}")
            print(f"Local rank: {self.local_rank}")
            print(f"Device: {self.device}")
            print(f"Run name: {self.run_name}")
            print(f"-----------------------------------\n")
            print(f"---TRAINING CONFIG:")
            print(f"Max iters: {self.max_iters}")
            print(f"Total batch size: {self.total_batch_size}")
            print(f"Per device batch size: {self.per_device_batch_size}")
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            print(f"Generations per prompt: {self.generations_per_prompt}")
            print(f"Per device prompt batch size: {self.per_device_prompt_batch_size}")
            print(f"Dataset size: {self.dataset_size} (smaller for debugging)")
            print(f"Lr: {cfg.lr}")
            print(f"-----------------------------------\n")
            print(f"---GENERATION CONFIG:")
            print(f"Max new tokens: {self.max_new_tokens}")
            print(f"Temperature: {self.temperature}")
            print(f"-----------------------------------\n")
            print(f"---REWARD CONFIG:")
            print(f"Reward type cot: {self.reward_type}")
            print(f"Generations per prompt: {self.generations_per_prompt}")
            print(f"Normalization type: {self.normalization_type}")
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
        logp, kl, entropy, normalized_reward = x
        assert logp.shape == kl.shape == entropy.shape == normalized_reward.shape == (self.per_device_batch_size,), f"{logp.shape=}, {kl.shape=}, {entropy.shape=}, {normalized_reward.shape=}"

        pg_loss = - logp * normalized_reward
        loss = pg_loss + self.kl_coef * kl + self.entropy_coef * entropy

        metrics = {}
        metrics["loss"] = loss.mean()
        metrics["pg_loss"] = pg_loss.mean()
        metrics["kl"] = kl.mean()
        metrics["entropy"] = entropy.mean()
        
        return loss.mean() / self.gradient_accumulation_steps, metrics
    
    @torch.no_grad()
    def normalize_reward(self, reward):
        if self.generations_per_prompt > 1:
            reward = reward.reshape(self.per_device_prompt_batch_size, self.generations_per_prompt)
            if self.normalization_type == "grpo":
                normalized_reward = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
            elif self.normalization_type == "rloo":
                group_sum = reward.sum(1, keepdim=True)
                normalized_reward = (group_sum - reward) / (self.generations_per_prompt - 1)
            elif self.normalization_type is None:
                normalized_reward = reward
            else:
                raise ValueError(f"{self.normalization_type=}")
            reward = reward.reshape(-1)
            normalized_reward = normalized_reward.reshape(-1)
        else:
            assert self.normalization_type is None, f"{self.normalization_type=}"
            normalized_reward = reward
        return normalized_reward

    def generate(
        self,
        questions_text,
        answer_text,
        tokenizer
    ):
        questions_inputs = self.tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
        questions_inputs = {k: v.to(self.model.device) for k, v in questions_inputs.items()}
        tokenized_answers = tokenizer(answer_text).input_ids
        # shapes
        batch_size = questions_inputs["input_ids"].shape[0]
        full_batch_size = batch_size * self.generations_per_prompt
        question_length = questions_inputs["input_ids"].shape[1]
        # repeat
        questions_inputs = careful_repeat_dict(questions_inputs, self.generations_per_prompt)
        tokenized_answers = [a for a in tokenized_answers for _ in range(self.generations_per_prompt)]
        # generate responses
        with torch.no_grad():
            q_responses_ids = self.model.generate(
                    input_ids=questions_inputs["input_ids"],
                    attention_mask=questions_inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_k=None,
                    do_sample=True,
                    pad_token_id=self.model.config.eos_token_id,
                    )
        # print(self.tokenizer.decode(q_responses_ids[0].tolist(), skip_special_tokens=False), "\n", "-"*50)
        # make patched tensors
        patched_q_responses_ids = q_responses_ids.clone()
        patched_q_responses_ids = torch.cat((q_responses_ids, torch.full((full_batch_size, 1), self.tokenizer.eos_token_id, dtype=torch.long, device=self.device)), dim=1) # extend by one in case eot doesn't fit
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
        for i, (reponse_ids, answer_ids) in enumerate(zip(responses_ids, tokenized_answers)):
            contains_eot[i] = (reponse_ids == self.tokenizer.eos_token_id).any()
            if contains_eot[i] == 1:
                idx = (reponse_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                reponse_mask[i, question_length+idx+1:] = 0
                patched_cot_mask[i, question_length+idx+1:] = 0
                reponse_ids[idx+1:] = self.tokenizer.pad_token_id
            contains_answer_prompt[i] = (reponse_ids == self.answer_prompt_id).any()
            if contains_answer_prompt[i] == 1:
                idx = (reponse_ids == self.answer_prompt_id).nonzero(as_tuple=True)[0][0]
                answer_start = question_length + idx + 1
                answer_length = len(answer_ids)
                if answer_start + answer_length < len(reponse_ids):
                    answer_ids = torch.tensor(answer_ids, device=self.device)
                    print(f"{reponse_ids[idx:idx+answer_length]=}")
                    print(f"{answer_ids=}")
                    contains_answer_i = (reponse_ids[idx:idx+answer_length] == answer_ids).all()
                    contains_answer[i] = contains_answer_i
                    patched_q_responses_ids[i, answer_start:answer_start+answer_length] = answer_ids
                    patched_q_responses_ids[i, answer_start+answer_length] = self.tokenizer.eos_token_id
                    patched_answer_mask[i, answer_start:answer_start+answer_length+1] = 1
                    patched_cot_mask[i, answer_start:] = 0
        if self.reward_type == "binary":
            input_ids = q_responses_ids
            mask = reponse_mask[:, 1:]
        elif self.reward_type == "prob":
            input_ids = patched_q_responses_ids
            mask = torch.logical_or(patched_cot_mask, patched_answer_mask).float()[:, 1:]
        else:
            raise ValueError(f"{self.reward_type=}")
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
        full_kl = (all_logps.exp() * (all_logps - ref_all_logps)).mean(-1)
        # mask and mean over length
        entropy = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1).float()
        ref_entropy = (ref_entropy * mask).sum(dim=-1) / mask.sum(dim=-1).float()
        sample_kl = (sample_kl * mask).sum(dim=-1) / mask.sum(dim=-1).float()
        full_kl = (full_kl * mask).sum(dim=-1) / mask.sum(dim=-1).float()
        logp = (per_token_logps * mask).sum(dim=-1) / mask.sum(dim=-1).float()
        ref_logp = (ref_per_token_logps * mask).sum(dim=-1) / mask.sum(dim=-1).float()
        # kl
        if self.kl_type == "sample":
            kl = sample_kl
        elif self.kl_type == "full":
            kl = full_kl
        else:
            raise ValueError(f"{self.kl_type=}")
        # reward
        with torch.no_grad():
            if self.reward_type == "binary":
                # contains answer
                reward = contains_answer.float()
            elif self.reward_type == "prob":
                # patched answer prob if contains prompt else 0
                answer_logp = torch.zeros(full_batch_size, device=self.device, dtype=torch.float)
                ref_answer_logp = torch.zeros(full_batch_size, device=self.device, dtype=torch.float)
                answer_p_length_normalized = torch.zeros(full_batch_size, device=self.device, dtype=torch.float)
                ref_answer_p_length_normalized = torch.zeros(full_batch_size, device=self.device, dtype=torch.float)
                for i in range(full_batch_size):
                    if patched_answer_mask[i].sum() > 0:
                        answer_logp_i = (per_token_logps * patched_answer_mask[i, 1:]).sum()
                        ref_answer_logp_i = (ref_per_token_logps * patched_answer_mask[i, 1:]).sum()
                        answer_p_length_normalized_i = torch.exp(answer_logp_i / patched_answer_mask[i, 1:].sum().float())
                        ref_answer_p_length_normalized_i = torch.exp(ref_answer_logp_i / patched_answer_mask[i, 1:].sum().float())
                        answer_logp[i] = answer_logp_i
                        ref_answer_logp[i] = ref_answer_logp_i
                        answer_p_length_normalized[i] = answer_p_length_normalized_i
                        ref_answer_p_length_normalized[i] = ref_answer_p_length_normalized_i
                reward = answer_p_length_normalized
            assert not torch.isnan(reward).any(), f"{reward=}"
            assert not torch.isinf(reward).any(), f"{reward=}"
            # normalize reward
            normalized_reward = self.normalize_reward(reward)
            assert not torch.isnan(normalized_reward).any(), f"{normalized_reward=}"
            assert not torch.isinf(normalized_reward).any(), f"{normalized_reward=}"
            print(f"{reward=}, {normalized_reward=}")

        # collect decoded generations
        decoded_generations = []
        for i in range(full_batch_size):
            reponse_start = question_length
            reponse_end = (reponse_mask[i] == 1).nonzero(as_tuple=True)[0][-1] + 1
            response = self.tokenizer.decode(q_responses_ids[i, reponse_start:reponse_end+1].tolist(), skip_special_tokens=False)
            if contains_answer_prompt[i]:
                answer_start = (patched_cot_mask[i] == 1).nonzero(as_tuple=True)[0][-1]
                generated_answer = self.tokenizer.decode(q_responses_ids[i, answer_start+1:answer_start+1+len(tokenized_answers[i])].tolist(), skip_special_tokens=False)
                all_answer_logps = all_logps[i, answer_start+1:answer_start+1+len(tokenized_answers[i])]
                argmax_answer = self.tokenizer.decode(all_answer_logps.argmax(-1).tolist(), skip_special_tokens=False)
            else:
                generated_answer = None
                argmax_answer = None
            decoded_generations.append((response,
                                generated_answer,
                                argmax_answer,
                                contains_answer_prompt[i].item() == 1,
                                contains_answer[i].item() == 1,
                                answer_p_length_normalized[i].item() if self.reward_type == "prob" else None,
                                patched_cot_mask.sum(dim=-1)[i].item(), # length of cot
                                normalized_reward[i].item(),
            ))

        metrics = {}
        metrics["contains_answer_prompt"] = contains_answer_prompt.mean()
        metrics["contains_answer"] = contains_answer.mean()
        metrics["length_to_eot"] = reponse_mask.sum(dim=-1).mean()
        metrics["length_of_cot"] = patched_cot_mask.sum(dim=-1).mean()
        metrics["logp"] = logp.mean()
        metrics["ref_logp"] = ref_logp.mean()
        metrics["entropy"] = entropy.mean()
        metrics["ref_entropy"] = ref_entropy.mean()
        metrics["sample_kl"] = sample_kl.mean()
        metrics["full_kl"] = full_kl.mean()
        metrics["kl"] = kl.mean()
        metrics["reward"] = reward.mean()
        metrics["normalized_reward_max"] = normalized_reward.max()
        metrics["normalized_reward_min"] = normalized_reward.min()
        if self.reward_type == "prob":
            metrics["answer_logp"] = answer_logp.mean()
            metrics["ref_answer_logp"] = ref_answer_logp.mean()
            metrics["answer_p_length_normalized"] = answer_p_length_normalized.mean()
            metrics["ref_answer_p_length_normalized"] = ref_answer_p_length_normalized.mean()

        assert logp.requires_grad == True, f"{logp.requires_grad=}"
        assert entropy.requires_grad == True, f"{entropy.requires_grad=}"
        assert kl.requires_grad == True, f"{kl.requires_grad=}"
        assert reward.requires_grad == False, f"{reward.requires_grad=}"
        return (logp, kl, entropy, normalized_reward), decoded_generations, metrics

    def run_training_loop(self, num_iters=None):
        """Run the training loop"""
        train_start_time = time.time()
        num_iters = self.max_iters if num_iters is None else num_iters
        for i in tqdm.tqdm(range(num_iters), desc="Training"):
            # GRADIENT ACCUMULATION
            for j in tqdm.tqdm(range(self.gradient_accumulation_steps), desc="Gradient accumulation", disable=True):
            
                # GENERATE ROLLOUTS
                start_time = time.time()
                self.model.eval()
                # sample batch randomly from the dataset
                indices = random.sample(range(len(self.train_dataset)), self.per_device_prompt_batch_size)
                dataset_batch = self.train_dataset.select(indices)
                questions_text = dataset_batch["question"]
                answers_text = dataset_batch["answer"]
                questions_inputs = self.tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
                answer_inputs = self.tokenizer(answers_text, return_tensors="pt", padding=True, padding_side="right")
                questions_inputs = {k: v.to(self.model.device) for k, v in questions_inputs.items()}
                answer_inputs = {k: v.to(self.model.device) for k, v in answer_inputs.items()}

                # GENERATE
                start_time = time.time()
                x, decoded_generations, generation_metrics = self.generate(questions_text, answers_text, self.tokenizer)
                generation_metrics = {f"gen/{k}": v for k, v in generation_metrics.items()}
                gen_time = time.time()-start_time
                generation_metrics["gen/time"] = torch.tensor(gen_time, device=self.device)

                # COMPUTE LOSS
                start_time = time.time()
                with self.ctx: 
                    loss, loss_metrics = self.get_loss(x)
                    self.scaler.scale(loss).backward()
                loss_metrics = {f"loss/{k}": v for k, v in loss_metrics.items()}
                loss_time = time.time()-start_time
                loss_metrics["loss/time"] = torch.tensor(loss_time, device=self.device)

                # UPDATE METRICS
                if j == 0:
                    metrics_s = {**generation_metrics, **loss_metrics}
                else:
                    metrics = {**generation_metrics, **loss_metrics}
                    metrics_s = {k: v + metrics[k] for k, v in metrics_s.items()}

                if j % 1 == 0 and self.rank == 0:
                    response, ans_gen, ans_argmax, contains_ap, contains_ans, ans_prob, length, norm_r = decoded_generations[0]
                    peak_mem_allocated = torch.cuda.max_memory_allocated() // 1024 // 1024
                    peak_mem_reserved = torch.cuda.max_memory_reserved() // 1024 // 1024
                    # print
                    print("-"*50)
                    print(f"Generation time: {gen_time:.1f}s")
                    print(f"Loss time: {loss_time:.1f}s")
                    print(f"peak memory allocated: {peak_mem_allocated} MiB, reserved: {peak_mem_reserved} MiB")
                    print("-"*50)
                    print(f"              QUESTION: {questions_text[0]}")
                    print(f"              RESPONSE: {response}")
                    print(f"      GENERATED ANSWER: {ans_gen}")
                    print(f"         ARGMAX ANSWER: {ans_argmax}")
                    print(f"        CORRECT ANSWER: {answers_text[0]}")
                    print(f"CONTAINS ANSWER PROMPT: {contains_ap}")
                    print(f"       CONTAINS ANSWER: {contains_ans}")
                    if self.reward_type == "prob":
                        print(f"           ANSWER PROB: {ans_prob:.2e}")
                    print(f"                LENGTH: {length}")
                    print(f"     NORMALIZED REWARD: {norm_r:.2e}")
                    print("-"*50)
                    print(f"Iter {i+1}/{self.max_iters}, Accumulation step {j+1}/{self.gradient_accumulation_steps}, Mean contains answer: {metrics_s['gen/contains_answer']}")
                    print("\n")
                    del response, ans_gen, ans_argmax, contains_ap, contains_ans, ans_prob, length
                    gc.collect()
                    torch.cuda.empty_cache()

                # cleanup
                del x, questions_inputs, dataset_batch, generation_metrics, loss_metrics, questions_text, answers_text, decoded_generations
                gc.collect()
                torch.cuda.empty_cache()

            # UPDATE MODEL
            self.apply_update()

            # LOG
            metrics_s = {k: v / self.gradient_accumulation_steps for k, v in metrics_s.items()}
            metrics_s = {k: v.item() / self.world_size for k, v in metrics_s.items()}
            param_metrics = get_model_param_stats(self.model, self.ref_model)
            metrics_s.update({f"params/{k}": v for k, v in param_metrics.items()})
            metrics_s.update({"iter": i, "lr": self.optimizer.param_groups[0]["lr"]})
            if self.use_wandb and self.rank == 0:
                wandb.log(metrics_s)
            del metrics_s, loss, param_metrics
            gc.collect()
            torch.cuda.empty_cache()

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
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=0 python run_full_single2.py use_wandb=False