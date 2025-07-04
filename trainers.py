import contextlib
import functools
import json
import os
import time
from collections import defaultdict
from copy import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers
import wandb
from omegaconf import DictConfig
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from data_selection import (
    get_active_iterator,
    get_online_iterator,
    get_shuffle_iterator,
)
from utils import (
    _get_batch_logps,
    all_gather_if_needed,
    formatted_dict,
    get_block_class_from_model,
    get_local_dir,
    pad_to_length,
    rank0_print,
    slice_and_move_batch_for_device,
)
import numpy as np
import random

def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(
        batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
    )
    concatenated_batch = {}
    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(
                batch[k], max_length, pad_value=pad_value
            )
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            )
    return concatenated_batch


class BasicTrainer(object):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer for a language model, supporting either SFT or DPO training.

        If multiple GPUs are present, naively splits the model across them, effectively
        offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        tokenizer_name_or_path = (
            config.model.tokenizer_name_or_path or config.model.name_or_path
        )
        rank0_print(f"Loading tokenizer {tokenizer_name_or_path}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == "sft",
            pretrain_fraction=config.pretrain_fraction,
            policy=policy,
            ref_policy=reference_model,
            beta=config.loss.beta,
            active_beta=config.active_beta,
            num_action_samples=config.num_action_samples,
            selection_ratio=config.selection_ratio,
            selection_strategy=config.selection_strategy,
            oracle=config.oracle,
            active_minibatch_size=config.active_minibatch_size,
            gen_active_batch_size=config.gen_active_batch_size,
        )

        self.policy = policy
        self.reference_model = reference_model
        print("config.active", config.active)
        print("config.online", config.online)
        # exit(0)
        if config.active:
            self.train_iterator = get_active_iterator(
                **data_iterator_kwargs,
                split="train",
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
                cache_dir=get_local_dir(config.local_dirs),
            )
        elif config.online:
            self.train_iterator = get_online_iterator(
                **data_iterator_kwargs,
                split="train",
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
                cache_dir=get_local_dir(config.local_dirs),
            )
        else:
            self.train_iterator = get_shuffle_iterator(
                **data_iterator_kwargs,
                split="train",
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
                cache_dir=get_local_dir(config.local_dirs),
            )
        rank0_print("Loaded train data iterator")
        self.eval_iterator = get_shuffle_iterator(
            **data_iterator_kwargs,
            split="test",
            n_examples=config.n_eval_examples,
            batch_size=config.eval_batch_size,
            silent=rank != 0,
            cache_dir=get_local_dir(config.local_dirs),
        )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(
            f"Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}"
        )

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""
        min_new_tokens = 1 if self.is_jeopardy else 0
        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(self.policy, writeback=False, recurse=False)
            if "FSDP" in self.config.trainer
            else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                inputs=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.config.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                min_new_tokens=min_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False,
            )
        if self.config.loss.name == "dpo":
            ctx = lambda: (
                FSDP.summon_full_params(
                    self.reference_model, writeback=False, recurse=False
                )
                if "FSDP" in self.config.trainer
                else contextlib.nullcontext()
            )
            with ctx():
                reference_output = self.reference_model.generate(
                    inputs=batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.config.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    min_new_tokens=min_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=False,
                )
        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        if self.config.loss.name == "dpo":
            reference_output = pad_to_length(
                reference_output, self.config.max_length, self.tokenizer.pad_token_id
            )
            reference_output = all_gather_if_needed(
                reference_output, self.rank, self.world_size
            )
            reference_output_decoded = self.tokenizer.batch_decode(
                reference_output, skip_special_tokens=True
            )
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        rejected_logps = all_logps[batch["chosen_input_ids"].shape[0] :]
        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        loss_config: DictConfig,
        train=True,
    ):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = "train" if train else "eval"

        if loss_config.name == "dpo" and not self.config.online:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(
                self.policy, batch
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = (
                    self.concatenated_forward(self.reference_model, batch)
                )

            losses, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=loss_config.beta,
                reference_free=loss_config.reference_free,
            )
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(
                chosen_rewards, self.rank, self.world_size
            )
            rejected_rewards = all_gather_if_needed(
                rejected_rewards, self.rank, self.world_size
            )
            reward_accuracies = all_gather_if_needed(
                reward_accuracies, self.rank, self.world_size
            )

            metrics[f"rewards_{train_test}/chosen"] = (
                chosen_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/rejected"] = (
                rejected_rewards.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/accuracies"] = (
                reward_accuracies.cpu().numpy().tolist()
            )
            metrics[f"rewards_{train_test}/margins"] = (
                (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            )

            policy_rejected_logps = all_gather_if_needed(
                policy_rejected_logps.detach(), self.rank, self.world_size
            )
            metrics[f"logps_{train_test}/rejected"] = (
                policy_rejected_logps.cpu().numpy().tolist()
            )

        else:
            policy_chosen_logits = self.policy(
                batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]
            ).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(
                policy_chosen_logits, batch["chosen_labels"], average_log_prob=False
            )
            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(
            policy_chosen_logps.detach(), self.rank, self.world_size
        )
        metrics[f"logps_{train_test}/chosen"] = (
            policy_chosen_logps.cpu().numpy().tolist()
        )

        all_devices_losses = all_gather_if_needed(
            losses.detach(), self.rank, self.world_size
        )
        metrics[f"loss/{train_test}"] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""
        rank0_print(f"Using {self.config.optimizer} optimizer")
        if self.config.optimizer == "Lion8bit":
            import bitsandbytes as bnb

            self.optimizer = getattr(bnb.optim, self.config.optimizer)(
                self.policy.parameters(), lr=self.config.lr
            )
        else:
            self.optimizer = getattr(torch.optim, self.config.optimizer)(
                self.policy.parameters(), lr=self.config.lr
            )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            ),
        )
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name == "dpo":
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None
        cols = ["step", "prompt", "sample"]
        self.is_jeopardy = self.config.datasets == ["jeopardy"]
        if self.is_jeopardy:
            cols.append("null_prob")
            cols.append("correct_answer")
            cols.append("null_is_max")
            self.null_token = self.tokenizer.eos_token_id
        if self.config.sample_during_eval:
            self.policy_text_table = wandb.Table(columns=cols)
            if self.config.loss.name == "dpo":
                cols = ["step", "prompt", "sample"]
                if self.is_jeopardy:
                    cols.append("correct_answer")
                self.reference_text_table = wandb.Table(columns=cols)
        if self.config.online:
            cols_online = ["step", "prompt", "chosen_sample", "rejected_sample"]
            self.policy_generated_training_table = wandb.Table(columns=cols_online)

        print(f"DTYPE: {next(self.policy.parameters()).dtype=}")
        for batch in self.train_iterator:
            cur_gpu_mem = torch.cuda.memory_allocated()
            torch.cuda.empty_cache()
            print(f"currently allocated: {cur_gpu_mem / 1e9:.2f} GB")
            if self.config.online:
                for i, (prompt, chosen_sample, rejected_sample) in enumerate(
                    zip(
                        batch["prompt"],
                        batch["chosen_response_only"],
                        batch["rejected_response_only"],
                    )
                ):
                    inputs_gen = [
                        self.example_counter,
                        prompt,
                        chosen_sample,
                        rejected_sample,
                    ]
                    self.policy_generated_training_table.add_data(*inputs_gen)
                wandb.log(
                    {
                        "policy_generated_training": copy(
                            self.policy_generated_training_table
                        )
                    },
                    step=self.example_counter,
                )
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (
                self.example_counter > 0 or self.config.do_first_eval
            ):
                self.evaluate()
            #### END EVALUATION ####

            torch.cuda.empty_cache()
            #### BEGIN TRAINING ####
            self.policy.train()
            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(
                    batch,
                    microbatch_idx,
                    self.config.gradient_accumulation_steps,
                    self.rank,
                )
                local_microbatch = slice_and_move_batch_for_device(
                    global_microbatch, self.rank, self.world_size, self.rank
                )
                loss, metrics = self.get_batch_metrics(
                    local_microbatch, self.config.loss, train=True
                )
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)
            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics["examples_per_second"].append(examples_per_second)
            batch_metrics["grad_norm"].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if (
                last_log is None
                or time.time() - last_log > self.config.minimum_log_interval_secs
            ):
                mean_train_metrics = {
                    k: sum(v) / len(v) for k, v in batch_metrics.items()
                }
                mean_train_metrics["counters/examples"] = self.example_counter
                mean_train_metrics["counters/updates"] = self.batch_counter
                rank0_print(
                    f"train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}"
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(
                    f"skipping logging after {self.example_counter} examples to avoid logging too frequently"
                )
            max_gpu_mem_so_far = torch.cuda.max_memory_allocated()
            print(f"Max allocated so far: {max_gpu_mem_so_far / 1e9:.2f} GB")
            cur_gpu_mem = torch.cuda.memory_allocated()
            print(f"currently allocated (after train): {cur_gpu_mem / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
            if (
                self.config.max_train_examples is not None
                and self.example_counter > self.config.max_train_examples
            ):
                break
            #### END TRAINING ####
        # evaluate one last time after training
        self.evaluate()

    def evaluate(self):
        rank0_print(f"Running evaluation after {self.example_counter} train examples")
        print("Beginning evaluation")
        cur_gpu_mem = torch.cuda.memory_allocated()
        torch.cuda.empty_cache()
        print(f"currently allocated: {cur_gpu_mem / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
        self.policy.eval()

        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples, all_reference_samples = [], []
        print(
            "rank",
            self.rank,
            self.world_size,
        )
        for eval_batch in (
            tqdm.tqdm(self.eval_batches, desc="Computing eval metrics")
            if self.rank == 0
            else self.eval_batches
        ):
            local_eval_batch = slice_and_move_batch_for_device(
                eval_batch, self.rank, self.world_size, self.rank
            )
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(
                    local_eval_batch, self.config.loss, train=False
                )

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        if self.config.sample_during_eval:
            if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(
                    f"Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts."
                )
                sample_batches = self.eval_batches[:1]
            else:
                n_sample_batches = (
                    self.config.n_eval_model_samples // self.config.eval_batch_size
                )
                sample_batches = self.eval_batches[:n_sample_batches]
            for eval_batch in (
                tqdm.tqdm(sample_batches, desc="Generating samples...")
                if self.rank == 0
                else sample_batches
            ):
                print(
                    "eval info",
                    len(eval_batch["prompt"]),
                    self.rank,
                    self.world_size,
                    self.rank,
                )
                local_eval_batch = slice_and_move_batch_for_device(
                    eval_batch, self.rank, self.world_size, self.rank
                )
                policy_samples, reference_samples = self.get_batch_samples(
                    local_eval_batch
                )
                # if Jeopardy data, get probability of null token
                if self.is_jeopardy:
                    with torch.no_grad():
                        outputs = self.policy(
                            local_eval_batch["prompt_input_ids"],
                            attention_mask=local_eval_batch["prompt_attention_mask"],
                        )
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=-1)
                        null_probs = probs[:, -1, self.null_token]
                        null_is_max = (
                            torch.argmax(probs[:, -1, :], dim=-1) == self.null_token
                        )
                all_policy_samples.extend(policy_samples)
                all_reference_samples.extend(reference_samples)

                for i, (prompt, sample) in enumerate(
                    zip(eval_batch["prompt"], policy_samples)
                ):
                    inputs = [self.example_counter, prompt, sample]
                    if self.is_jeopardy:
                        inputs.append(null_probs[i].item())
                        inputs.append(eval_batch["chosen_response_only"][i])
                        inputs.append(null_is_max[i].item())
                    self.policy_text_table.add_data(*inputs)
                if self.config.loss.name == "dpo":
                    for i, (prompt, sample) in enumerate(
                        zip(eval_batch["prompt"], reference_samples)
                    ):
                        inputs = [self.example_counter, prompt, sample]
                        if self.is_jeopardy:
                            inputs.append(eval_batch["chosen_response_only"][i])
                        self.reference_text_table.add_data(*inputs)
        max_gpu_mem_so_far = torch.cuda.max_memory_allocated()
        print(f"Max allocated so far: {max_gpu_mem_so_far / 1e9:.2f}GB")
        cur_gpu_mem = torch.cuda.memory_allocated()
        print(f"currently allocated: {cur_gpu_mem / 1e9:.2f}GB")

        mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
        rank0_print(
            f"eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}"
        )
        if self.config.sample_during_eval:
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))
            if self.config.loss.name == "dpo":
                rank0_print(json.dumps(all_reference_samples[:10], indent=2))

        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)

            if self.config.sample_during_eval:
                print("saving table")
                wandb.log(
                    {"policy_samples": copy(self.policy_text_table)},
                    step=self.example_counter,
                )
                if self.config.loss.name == "dpo":
                    wandb.log(
                        {"reference_samples": copy(self.reference_text_table)},
                        step=self.example_counter,
                    )

        if self.example_counter > 0:
            if self.config.debug:
                rank0_print("skipping save in debug mode")
            else:
                if (self.example_counter % self.config.eval_every == 0) or (
                    self.example_counter == self.config.eval_every
                ):
                    output_dir = os.path.join(
                        self.run_dir, f"step-{self.example_counter}"
                    )
                    rank0_print(f"creating checkpoint to write to {output_dir}...")
                    self.save(output_dir, mean_eval_metrics)
                self.save()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def write_state_dict(
        self,
        step: int,
        state: Dict[str, torch.Tensor],
        metrics: Dict,
        filename: str,
        dir_name: Optional[str] = None,
    ):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, "LATEST")

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f"writing checkpoint to {output_path}...")
        torch.save(
            {
                "step_idx": step,
                "state": state,
                "metrics": metrics if metrics is not None else {},
            },
            output_path,
        )

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(
            self.example_counter, policy_state_dict, metrics, "policy.pt", output_dir
        )
        if output_dir is None:
            optimizer_state_dict = self.optimizer.state_dict()
            self.write_state_dict(
                self.example_counter,
                optimizer_state_dict,
                metrics,
                "optimizer.pt",
                output_dir,
            )
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(
                self.example_counter,
                scheduler_state_dict,
                metrics,
                "scheduler.pt",
                output_dir,
            )


class FSDPTrainer(BasicTrainer):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.

        This trainer will shard both the policy and reference model across all available GPUs.
        Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )
        assert (
            config.model.block_name is not None
        ), "must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP"

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False,
        )

        rank0_print("Sharding policy...")
        mp_dtype = (
            getattr(torch, config.model.fsdp_policy_mp)
            if config.model.fsdp_policy_mp is not None
            else None
        )
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(
            policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy
        )

        if config.activation_checkpointing:
            rank0_print("Attempting to enable activation checkpointing...")
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )

                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print("FSDP activation checkpointing not available:", e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print("Applying activation checkpointing wrapper to policy...")
                apply_activation_checkpointing(
                    self.policy,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn,
                )
                rank0_print("FSDP activation checkpointing enabled!")

        if config.loss.name == "dpo":
            rank0_print("Sharding reference model...")
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print("Loaded model on rank", rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                "policy.pt",
                output_dir,
            )
        dist.barrier()
        if output_dir is None:
            save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.policy,
                StateDictType.FULL_STATE_DICT,
                optim_state_dict_config=save_policy,
            ):
                optimizer_state_dict = FSDP.optim_state_dict(
                    self.policy, self.optimizer
                )

            if self.rank == 0:
                self.write_state_dict(
                    self.example_counter,
                    optimizer_state_dict,
                    metrics,
                    "optimizer.pt",
                    output_dir,
                )
            dist.barrier()

            if self.rank == 0:
                scheduler_state_dict = self.scheduler.state_dict()
                self.write_state_dict(
                    self.example_counter,
                    scheduler_state_dict,
                    metrics,
                    "scheduler.pt",
                    output_dir,
                )
            dist.barrier()
