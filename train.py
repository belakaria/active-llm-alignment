import json
import os
import resource
import socket
from typing import Optional, Set

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import transformers
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers import BitsAndBytesConfig

import trainers
from utils import (
    DropoutModel,
    disable_dropout,
    get_local_dir,
    get_local_run_dir,
    get_open_port,
    init_distributed,
)

torch.backends.cuda.matmul.allow_tf32 = True
mp.set_sharing_strategy("file_system")

OmegaConf.register_new_resolver(
    "get_local_run_dir",
    lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs),
)


def worker_main(
    rank: int,
    world_size: int,
    config: DictConfig,
    policy: nn.Module,
    reference_model: Optional[nn.Module] = None,
):
    """Main function for each worker process (may be only 1 for BasicTrainer)."""
    if "FSDP" in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ["WANDB_SERVICE_WAIT"] = "90"
        os.environ["WANDB_CACHE_DIR"] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f"Creating trainer on process {rank} with world size {world_size}")
    print("local_run_dir", config.local_run_dir)
    trainer = TrainerClass(
        policy,
        config,
        config.seed,
        config.local_run_dir,
        reference_model=reference_model,
        rank=rank,
        world_size=world_size,
    )

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print("WARNING: eval_every must be divisible by batch_size")
        print(
            "Setting eval_every to",
            config.eval_every - config.eval_every % config.batch_size,
        )
        config.eval_every = config.eval_every - config.eval_every % config.batch_size
    if "FSDP" in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print("no FSDP port specified; using open port for FSDP:", free_port)
        config.fsdp_port = free_port
    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, "config.yaml")
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)

    print("=" * 80)
    print(f"Writing to {socket.gethostname()}:{config.local_run_dir}")
    print("=" * 80)

    os.environ["XDG_CACHE_HOME"] = get_local_dir(config.local_dirs)
    print("building policy")
    model_kwargs = (
        {"device_map": "balanced"} if config.trainer == "BasicTrainer" else {}
    )
    policy_dtype = getattr(torch, config.model.policy_dtype)
    if config.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            output_hidden_states=True,
            trust_remote_code=True,
            torch_dtype=policy_dtype,
            use_auth_token=True,
            **model_kwargs,
        )
    else:
        policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            torch_dtype=policy_dtype,
            use_auth_token=True,
            **model_kwargs,
        )
    if not config.have_llm_dropout:
        disable_dropout(policy)

    if config.qlora:
        policy.gradient_checkpointing_enable()
        policy = prepare_model_for_kbit_training(policy)

        if "pythia" in config.model.name_or_path:
            target_modules = ["query_key_value"]
        elif "llama" in config.model.name_or_path:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gpt2" in config.model.name_or_path:
            target_modules = ["attn.c_attn", "attn.c_proj"]
        loraconfig = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        policy = get_peft_model(policy, loraconfig)
        for name, module in policy.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.bfloat16)
            if hasattr(module, "weight"):
                if module.weight is not None and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if config.loss.name == "dpo":
        print("building reference model")
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        if config.qlora:
            reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model.name_or_path,
                cache_dir=get_local_dir(config.local_dirs),
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
                output_hidden_states=True,
                trust_remote_code=True,
                torch_dtype=reference_model_dtype,
                use_auth_token=True,
                **model_kwargs,
            )
        else:
            reference_model = transformers.AutoModelForCausalLM.from_pretrained(
                config.model.name_or_path,
                cache_dir=get_local_dir(config.local_dirs),
                low_cpu_mem_usage=True,
                torch_dtype=reference_model_dtype,
                use_auth_token=True,
                **model_kwargs,
            )
        if not config.have_llm_dropout:
            disable_dropout(reference_model)
        if config.qlora:
            reference_model.gradient_checkpointing_enable()
            reference_model = prepare_model_for_kbit_training(reference_model)
            reference_model = get_peft_model(reference_model, loraconfig)
            for name, module in reference_model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.bfloat16)
                if hasattr(module, "weight"):
                    if (
                        module.weight is not None
                        and module.weight.dtype == torch.float32
                    ):
                        module = module.to(torch.bfloat16)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        if module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
    else:
        reference_model = None

    if config.model.archive is not None:
        archive_path = to_absolute_path(config.model.archive)
        state_dict = torch.load(archive_path, map_location="cpu")
        step, metrics = state_dict["step_idx"], state_dict["metrics"]
        print(
            f"loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}"
        )
        policy.load_state_dict(state_dict["state"], strict=False)
        policy.tie_weights()
        if config.loss.name == "dpo":
            reference_model.load_state_dict(state_dict["state"], strict=False)
            reference_model.tie_weights()
        print("loaded pre-trained weights")
    if config.have_llm_dropout:
        policy = DropoutModel(policy, config.llm_dropout, lora=False)
        if config.loss.name == "dpo":
            reference_model = DropoutModel(reference_model, 0.0, lora=False)
    if "FSDP" in config.trainer:
        if config.qlora:
            raise NotImplementedError("Lora + FSDP doesn't work yet")
        world_size = torch.cuda.device_count()
        print("starting", world_size, "processes for FSDP training")
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"setting RLIMIT_NOFILE soft limit to {hard} from {soft}")
        mp.spawn(
            worker_main,
            nprocs=world_size,
            args=(world_size, config, policy, reference_model),
            join=True,
        )
    else:
        print("starting single-process worker")
        worker_main(0, 1, config, policy, reference_model)


if __name__ == "__main__":
    main()
