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

from my_utils import careful_repeat, batch_generate_rnn, get_model_param_stats, extract_solution

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
        self.cot_length = cfg.cot_length
        self.as_full_distribution = cfg.as_full_distribution
        self.generations_per_prompt = cfg.generations_per_prompt
        self.normalization_type = cfg.normalization_type if self.generations_per_prompt != 1 else None
        self.temperature = cfg.temperature
        assert not cfg.generations_per_prompt > 1 and cfg.as_full_distribution, f"{cfg.generations_per_prompt=} {cfg.as_full_distribution=}"
        # reward
        self.reward_type = cfg.reward_type
        self.teacher_forcing = cfg.teacher_forcing
        self.include_cot = cfg.include_cot
        self.include_answer = cfg.include_answer
        self.answer_prompt = cfg.answer_prompt
        assert self.normalization_type in [None, "grpo", "rloo"], f"{self.normalization_type=}"
        assert self.reward_type in ["answer_generated", "answer_prob"], f"{self.reward_type=}"
        assert not self.as_full_distribution and self.include_cot, f"{self.as_full_distribution=} {self.include_cot=}"
        # loss
        self.kl_type = cfg.kl_type
        self.kl_coef = cfg.kl_coef
        self.entropy_coef = cfg.entropy_coef

        if self.rank == 0:
            
            # run name
            run_name = f"{cfg.run_name_prefix}-" if cfg.run_name_prefix != "" else ""
            
            # generation
            run_name += f"-L{self.cot_length}"
            run_name += f"-FULL_DIST" if self.as_full_distribution else ""
            run_name += f"-g{self.generations_per_prompt}" if self.generations_per_prompt != 1 else ""
            run_name += f"-{self.normalization_type}" if self.normalization_type != None else ""
            run_name += f"-T{self.temperature}" if self.temperature != 1.0 else ""
            # reward
            run_name += f"--generated" if self.reward_type == "answer_generated" else "--prob"
            run_name += f"-TF{self.teacher_forcing}" if self.reward_type == "answer_prob" and self.as_full_distribution else ""
            run_name += f"-COT" if self.include_cot else "noCOT"
            run_name += f"-ANS" if self.include_answer else "noANS"
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
            print(f"Per device prompt batch size: {self.per_device_prompt_batch_size}")
            print(f"Dataset size: {self.dataset_size} (smaller for debugging)")
            print(f"Lr: {cfg.lr}")
            print(f"-----------------------------------\n")
            print(f"---GENERATION CONFIG:")
            print(f"Cot length: {self.cot_length}")
            print(f"As full distribution: {self.as_full_distribution}")
            print(f"Generations per prompt: {self.generations_per_prompt}")
            print(f"Normalization type: {self.normalization_type}")
            print(f"Temperature: {self.temperature}")
            print(f"-----------------------------------\n")
            print(f"---REWARD CONFIG:")
            print(f"Reward type: {self.reward_type}")
            print(f"Teacher forcing: {self.teacher_forcing}")
            print(f"Include cot: {self.include_cot}")
            print(f"Include answer: {self.include_answer}")
            print(f"Answer prompt: {self.answer_prompt}")
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

    def get_loss(self, cot_logp, answer_logp, kl, entropy, normalized_rewards):
        metrics = {}
        # calculate pg / logp loss
        if self.reward_type == "answer_generated":
            answer_loss = -answer_logp * normalized_rewards
        elif self.reward_type == "answer_prob":
            answer_loss = -answer_logp
        else:
            raise ValueError(f"{self.reward_type=}")
        cot_loss = -cot_logp * normalized_rewards
        pg_loss = answer_loss if self.include_answer else 0.0
        pg_loss += cot_loss if self.include_cot else 0.0

        # kl and entropy
        loss = pg_loss + self.kl_coef * kl + self.entropy_coef * entropy

        metrics["answer_loss"] = answer_loss.mean()
        metrics["cot_loss"] = cot_loss.mean()
        metrics["pg_loss"] = pg_loss.mean()
        metrics["kl"] = kl.mean()
        metrics["entropy"] = entropy.mean()
        metrics["loss"] = loss.mean()
        
        return loss.mean() / self.gradient_accumulation_steps, metrics


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
                # get questions (and answers)
                # sample batch randomly from the dataset
                indices = random.sample(range(len(self.train_dataset)), self.per_device_prompt_batch_size)
                dataset_batch = self.train_dataset.select(indices)
                questions_text = dataset_batch["question"]
                # cot_text = dataset_batch["reasoning"]
                answers_text = dataset_batch["answer"]

                questions_inputs = self.tokenizer(questions_text, return_tensors="pt", padding=True, padding_side="left")
                questions_inputs = {k: v.to(self.model.device) for k, v in questions_inputs.items()}

                # repeat for generations_per_prompt
                questions_inputs = careful_repeat(questions_inputs, self.generations_per_prompt)

                # generate
                start_time = time.time()
                x, generations, generation_metrics = batch_generate_rnn(
                    model=self.model,
                    ref_model=self.ref_model,
                    tokenizer=self.tokenizer,
                    questions_inputs=questions_inputs,
                    answers_text=answers_text,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    loss_type=self.loss_type,
                    logp_teacher_forcing=self.logp_teacher_forcing,
                    logp_steps_if_no_eot=self.logp_steps_if_no_eot,
                )
                generation_metrics = {f"gen/{k}": v for k, v in generation_metrics.items()}
                gen_time = time.time()-start_time
                generation_metrics["gen/time"] = torch.tensor(gen_time, device=self.device)

                ### rewards
                rewards, normalized_rewards, decoded_generations, extracted_answers, reward_metrics = self.get_rewards(generations, answers_text)
                reward_metrics = {k if k == "REWARD" else f"reward/{k}": v for k, v in reward_metrics.items()}
                    
                # COMPUTE LOSS
                start_time = time.time()
                with self.ctx: 
                    loss, loss_metrics = self.get_loss(x, normalized_rewards)
                    self.scaler.scale(loss).backward()
                loss_metrics = {f"loss/{k}": v for k, v in loss_metrics.items()}
                loss_time = time.time()-start_time
                loss_metrics["loss/time"] = torch.tensor(loss_time, device=self.device)

                # UPDATE METRICS
                if j == 0:
                    metrics_s = {**generation_metrics, **loss_metrics, **reward_metrics}
                else:
                    metrics = {**generation_metrics, **loss_metrics, **reward_metrics}
                    metrics_s = {k: v + metrics[k] for k, v in metrics_s.items()}

                if j % 1 == 0:
                    # print
                    print("-"*50)
                    print(f"Iter {i+1}, Accumulation step {j+1}/{self.gradient_accumulation_steps}")
                    print(f"Generation time: {gen_time:.1f}s")
                    print(f"Loss time: {loss_time:.1f}s")
                    print("-"*50)
                    print(f"              QUESTION: {questions_text[0]}")
                    print(f"            GENERATION: {decoded_generations[0]}")
                    print(f"                ANSWER: {answers_text[0]}")
                    print(f"      EXTRACTED ANSWER: {extracted_answers[0]}")
                    print(f"                 MATCH: --- {answers_text[0] == extracted_answers[0]}")
                    print(f"                LENGTH: {(generations[0] != self.tokenizer.eos_token_id).int().sum().item()}")
                    print(f"                REWARD: {rewards[0]:.2f}")
                    print("-"*50)
                    print("\n")

                # cleanup
                del x, normalized_rewards, questions_inputs, dataset_batch, generation_metrics, loss_metrics, reward_metrics, generations, rewards, questions_text, answers_text, decoded_generations, extracted_answers
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
            if self.use_wandb:
                wandb.log(metrics_s)
            del metrics_s, loss
            gc.collect()
            torch.cuda.empty_cache()


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

@hydra.main(config_path="args", config_name="full_test", version_base=None)
def main(cfg):
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()


# CUDA_VISIBLE_DEVICES=0 python run_full_single_gpu.py