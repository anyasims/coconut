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

from my_utils import careful_repeat, batch_generate_rnn, get_model_param_stats

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
        self.model = AutoModelForCausalLM.from_pretrained(cfg.base_model).to(self.device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(cfg.base_model).to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.lr),
            betas=cfg.get('betas', (0.9, 0.999)),
            weight_decay=cfg.get('weight_decay', 1e-2))

        # dataset
        if cfg.dataset == "gsm8k":
            train_dataset = DatasetDict.load_from_disk(f"./data/my_data/gsm8k")["train"]
            if self.dataset_size is not None:
                train_dataset = train_dataset.select(range(self.dataset_size))
        else:
            raise NotImplementedError(f"Dataset {cfg.dataset} not implemented.")
        self.train_dataset = train_dataset

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
        self.generations_per_prompt = cfg.generation.generations_per_prompt
        assert self.per_device_batch_size % self.generations_per_prompt == 0, f"{self.per_device_batch_size=} {self.generations_per_prompt=}"
        self.per_device_prompt_batch_size = self.per_device_batch_size // self.generations_per_prompt
        self.dataset_size = cfg.dataset_size

        # generation
        self.loss_type = cfg.loss.loss_type
        if self.loss_type in ["pg", "logp"]:
            self.temperature = cfg.generation.temperature
            self.top_k = cfg.generation.top_k
            self.max_steps = cfg.generation.max_steps
            self.step_for_answer = cfg.generation.step_for_answer if self.loss_type == "logp" or cfg.generation.compute_everything else None
            if self.step_for_answer is not None:
                assert self.step_for_answer < self.max_steps, f"{self.step_for_answer=}, {self.max_steps=}"
            self.inject_answer_prompt = cfg.generation.inject_answer_prompt
            self.as_full_distribution = cfg.generation.as_full_distribution
            self.dot_by_dot = cfg.generation.dot_by_dot
            assert not (self.as_full_distribution and self.dot_by_dot), f"{self.as_full_distribution=}, {self.dot_by_dot=}"
            self.answer_prompt_text = " .... Answer:"
            self.answer_prompt_ids = self.tokenizer.encode(self.answer_prompt_text)
            assert len(self.tokenizer.encode("....")) == 1
            self.dot_by_dot_id = self.tokenizer.encode("....")[0]
            self.predict_answer_prompt = cfg.generation.predict_answer_prompt
            assert not (self.predict_answer_prompt and self.inject_answer_prompt), f"{self.predict_answer_prompt=}, {self.inject_answer_prompt=}"
        self.compute_everything = cfg.generation.compute_everything

        # loss
        if self.loss_type == "sft":
            self.sft_include_cot = cfg.loss.sft_include_cot
            self.sft_predict_cot = cfg.loss.sft_predict_cot
            raise NotImplementedError("SFT not yet implemented.")
        elif self.loss_type == "pg":
            self.pg_normalization_type = "none" if self.generations_per_prompt == 1 else cfg.loss.pg_normalization_type
            self.answer_prompt_coef = cfg.loss.answer_prompt_coef
            assert self.answer_prompt_coef == 0.0, f"Have removed answer prompt reward for now but have: {self.answer_prompt_coef=}"
        self.entropy_coef = cfg.loss.entropy_coef
        self.kl_loss_coef = cfg.loss.kl_loss_coef

        if self.rank == 0:
            short_model_name = f"Qw{cfg.base_model.split("/")[-1].split("-")[1]}" if "Qwen" in cfg.base_model else cfg.base_model.split("/")[-1]
            # run name
            run_name = f"{cfg.run_name_prefix}-" if cfg.run_name_prefix != "" else ""
            run_name += f"-{short_model_name}"
            # method
            run_name += f"-{self.loss_type}"
            run_name += f"-FULLDIST" if self.as_full_distribution else ""
            run_name += f"-DOT" if self.dot_by_dot else ""
            run_name += f"-INJ" if self.inject_answer_prompt else ""
            run_name += f"-AP" if self.predict_answer_prompt and self.loss_type == "logp" else ""
            run_name += f"-steps{self.max_steps}"
            run_name += f"_{self.step_for_answer}" if self.loss_type == "logp" else ""
            run_name += f"-{self.pg_normalization_type}" if self.loss_type == "pg" and self.pg_normalization_type is not None else ""
            # training
            run_name += "-"
            run_name += f"-B{self.total_batch_size}"
            run_name += f"-G{self.generations_per_prompt}" if self.generations_per_prompt != 1 else ""
            run_name += f"-D{self.dataset_size}" if self.dataset_size is not None else ""
            run_name += f"-lr{cfg.lr:.0e}"
            # generation
            run_name += "-"
            run_name += f"-T{self.temperature}" if self.temperature != 1.0 else ""
            run_name += f"-topK{self.top_k}" if self.top_k is not None else ""
            # loss
            run_name += f"-a{self.answer_prompt_coef}" if self.loss_type == "pg" else ""
            run_name += f"-kl{self.kl_loss_coef}" if self.kl_loss_coef != 0.0 else ""
            run_name += f"-ent{self.entropy_coef}" if self.entropy_coef != 0.0 else ""
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
            print(f"Dataset size: {self.dataset_size} (saller for debugging)")
            print(f"-----------------------------------\n")
            print(f"---GENERATION CONFIG:")
            print(f"Temperature: {self.temperature}")
            print(f"Top k: {self.top_k}")
            print(f"Max length: {self.max_steps}")
            print(f"Step for answer: {self.step_for_answer}")
            print(f"Inject answer prompt: {self.inject_answer_prompt}")
            print(f"As full distribution: {self.as_full_distribution}")
            print(f"Answer prompt text: {self.answer_prompt_text}, ids: {self.answer_prompt_ids}")
            print(f"Dot by dot: {self.dot_by_dot}, id: {self.dot_by_dot_id}")
            print(f"Predict answer prompt: {self.predict_answer_prompt}")
            print(f"Compute everything: {self.compute_everything}")
            print(f"-----------------------------------\n")
            print(f"---LOSS CONFIG:")
            print(f"Loss type: {self.loss_type}")
            if self.loss_type == "sft":
                print(f"SFT include cot: {self.sft_include_cot}")
                print(f"SFT predict cot: {self.sft_predict_cot}")
            elif self.loss_type == "pg":
                print(f"PG normalization type: {self.pg_normalization_type}")
            print(f"Entropy coef: {self.entropy_coef}")
            print(f"KL loss coef: {self.kl_loss_coef}")
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

    def get_rewards(self, generations, answers_text):
        """Get rewards"""
        ### check for answer
        contains_answer_prompt = torch.zeros((self.per_device_prompt_batch_size, self.generations_per_prompt), device=self.model.device)
        contains_answer = torch.zeros((self.per_device_prompt_batch_size, self.generations_per_prompt), device=self.model.device)
        generations = generations.reshape(self.per_device_prompt_batch_size, self.generations_per_prompt, -1)
        decoded_generations = []
        extracted_answers = []
        for i in range(self.per_device_prompt_batch_size):
            answer_text = answers_text[i]
            decoded_batch = self.tokenizer.batch_decode(generations[i])
            for j in range(self.generations_per_prompt):
                decoded = decoded_batch[j].split(self.tokenizer.eos_token)[0]
                contains_answer_prompt_ij = self.answer_prompt_text in decoded
                numbers = re.findall(r'\d+', decoded)
                extracted_answer = numbers[-1] if numbers else None
                contains_answer_ij = extracted_answer == answer_text
                contains_answer_prompt[i, j] = contains_answer_prompt_ij
                contains_answer[i, j] = contains_answer_ij
                decoded_generations.append(decoded)
                extracted_answers.append(extracted_answer)
        
        ### caluclate rewards
        rewards = contains_answer.float()
        if self.loss_type == "pg":
            if self.generations_per_prompt == 1:
                assert self.pg_normalization_type == "none", f"{self.pg_normalization_type=}"
            if self.pg_normalization_type == "grpo":
                normalized_rewards = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)
            elif self.pg_normalization_type == "rloo":
                group_sum = rewards.sum(1, keepdim=True)
                normalized_rewards = (group_sum - rewards) / (self.generations_per_prompt - 1)
            elif self.pg_normalization_type == "none":
                normalized_rewards = rewards
            else:
                raise ValueError(f"{self.pg_normalization_type=}")
        else:
            normalized_rewards = None
        
        metrics = {
            "REWARD": rewards.mean(),
            "reward/reward_std": rewards.std() if self.per_device_batch_size > 1 else torch.tensor(0, device=self.device),
            "reward/reward_std_within_q": rewards.std(1).mean() if self.generations_per_prompt > 1 else torch.tensor(0, device=self.device),
            "reward/reward_std_between_q": rewards.mean(1).std() if self.per_device_prompt_batch_size > 1 else torch.tensor(0, device=self.device),
        }

        rewards = rewards.reshape(self.per_device_batch_size)
        normalized_rewards = normalized_rewards.reshape(self.per_device_batch_size) if normalized_rewards is not None else None
        return rewards, normalized_rewards, decoded_generations, extracted_answers, metrics
    
    def get_loss(self, x, rewards):
        metrics = {}
        pg_loss, logp_loss = torch.tensor(0.0, device=self.model.device), torch.tensor(0.0, device=self.model.device)
        if self.loss_type == "sft":
            raise NotImplementedError
        else:
            answer_logps, _, gen_per_token_logps, ref_per_token_logps, entropy = x
            if self.loss_type == "pg":
                pg_loss = - torch.exp(gen_per_token_logps - gen_per_token_logps.detach()).mean(-1) * rewards
            elif self.loss_type == "logp":
                logp_loss = - answer_logps
            else:
                raise ValueError(f"{self.loss_type=}")
            kl = (torch.exp(ref_per_token_logps - gen_per_token_logps) - (ref_per_token_logps - gen_per_token_logps) - 1).mean(-1)
            loss = pg_loss + logp_loss + self.kl_loss_coef * kl - self.entropy_coef * entropy
            loss = loss.mean()

            metrics["loss"] = loss
            metrics["pg_loss"] = pg_loss.mean()
            metrics["logp_loss"] = logp_loss.mean()
            metrics["kl"] = kl.mean()
            metrics["entropy"] = entropy.mean()
        return loss / self.gradient_accumulation_steps, metrics


    def run_training_loop(self, num_iters=None):
        """Run the training loop"""
        start_time = time.time()
        num_iters = self.max_iters if num_iters is None else num_iters
        for i in tqdm.tqdm(range(num_iters), desc="Training"):
            # GRADIENT ACCUMULATION
            for j in tqdm.tqdm(range(self.gradient_accumulation_steps), desc="Gradient accumulation", disable=False):
            
                # GENERATE ROLLOUTS
                start_time = time.time()
                self.model.eval()
                # with torch.no_grad():
                # get questions (and answers)
                # sample batch randomly from the dataset
                indices = random.sample(range(len(self.train_dataset)), self.per_device_prompt_batch_size)
                dataset_batch = self.train_dataset.select(indices)
                questions_text = dataset_batch["question"]
                # cot_text = dataset_batch["reasoning"]
                answers_text = dataset_batch["answer"]
                if self.predict_answer_prompt:
                    answers_text_input = [self.answer_prompt_text + " " + a.strip() for a in answers_text]
                else:
                    answers_text_input = answers_text

                if self.loss_type == "sft":
                    raise NotImplementedError
                
                else:
                    questions_inputs = self.tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
                    answers_inputs = self.tokenizer(answers_text_input, return_tensors="pt", padding=True)
                    questions_inputs = {k: v.to(self.model.device) for k, v in questions_inputs.items()}
                    answers_inputs = {k: v.to(self.model.device) for k, v in answers_inputs.items()}

                    # repeat for generations_per_prompt
                    questions_inputs = careful_repeat(questions_inputs, self.generations_per_prompt)
                    answers_inputs = careful_repeat(answers_inputs, self.generations_per_prompt)

                    # generate
                    x, generations, generation_metrics, _ = batch_generate_rnn(
                        model=self.model,
                        ref_model=self.ref_model,
                        questions_inputs=questions_inputs,
                        answers_inputs=answers_inputs,
                        max_steps=self.max_steps,
                        step_for_answer=self.step_for_answer,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        as_full_distribution=self.as_full_distribution,
                        dot_by_dot=self.dot_by_dot,
                        dot_by_dot_id=self.dot_by_dot_id,
                        inject_answer_prompt=self.inject_answer_prompt,
                        answer_prompt_ids=self.answer_prompt_ids,
                        loss_type=self.loss_type,
                        compute_everything=self.compute_everything,
                    )
                    generation_metrics = {f"gen/{k}": v for k, v in generation_metrics.items()}

                    ### rewards
                    rewards, normalized_rewards, decoded_generations, extracted_answers, reward_metrics = self.get_rewards(generations, answers_text)
                    reward_metrics = {k if k == "REWARD" else f"reward/{k}": v for k, v in reward_metrics.items()}
                    
                # COMPUTE LOSS
                with self.ctx: 
                    loss, loss_metrics = self.get_loss(x, normalized_rewards)
                    self.scaler.scale(loss).backward()
                loss_metrics = {f"loss/{k}": v for k, v in loss_metrics.items()}

                # UPDATE METRICS
                if j == 0:
                    metrics_s = {**generation_metrics, **loss_metrics, **reward_metrics}
                else:
                    metrics = {**generation_metrics, **loss_metrics, **reward_metrics}
                    metrics_s = {k: v + metrics[k] for k, v in metrics_s.items()}

            # UPDATE MODEL
            self.apply_update()

            # LOG
            metrics_s = {k: v / self.gradient_accumulation_steps for k, v in metrics_s.items()}
            metrics_s = {k: v.item() / self.world_size for k, v in metrics_s.items()}
            param_metrics = get_model_param_stats(self.model, self.ref_model)
            metrics_s.update({f"params/{k}": v for k, v in param_metrics.items()})
            metrics_s.update({"iter": i, "lr": self.optimizer.param_groups[0]["lr"]})
            if self.use_wandb:
                wandb.log(metrics_s)
            if i % 1 == 0 or i == self.max_iters - 1:
                lengths = (torch.cumsum((generations == self.tokenizer.eos_token_id).int(), dim=1) == 0).int().sum(dim=-1)
                print(f"({metrics_s})\n\niter {i}: REWARD={metrics_s['REWARD']:.2f}")
                num_to_print = min(3, self.per_device_batch_size)
                print("-"*50)
                for k in range(num_to_print):
                    print(f"EXAMPLE {k}: (REWARD={rewards[k].item():.4f}):")
                    print(f"    QUESTION: {questions_text[k//self.generations_per_prompt]}")
                    print(f"    GENERATION: {decoded_generations[k]}")
                    print(f"    EXTRACTED ANSWER: {extracted_answers[k]}")
                    print(f"    ANSWER: {answers_text[k//self.generations_per_prompt]}")
                    print(f"    LENGTH: {lengths[k]}")
                    print("-"*50)
                print("\n\n")

            # clenup
            del x, normalized_rewards, generations
            gc.collect()
            torch.cuda.empty_cache()


        print(f"Training time: {time.time()-start_time:.1f}s")
        print("Training complete")
        wandb.finish()
            
    def train(self, num_iters=None):
        """Train the model"""
        # set seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.run_training_loop(num_iters=num_iters)

@hydra.main(config_path="args", config_name="full_test", version_base=None)
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()



# CUDA_VISIBLE_DEVICES=0 python run_full_single_gpu.py