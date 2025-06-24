import asyncio
import random
from typing import Dict, Iterator, List, Optional

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from preference_datasets import (
    get_collate_fn,
    get_dataset,
    get_winners,
    tokenize_batch_element,
)
from utils import (
    TemporarilySeededRandom,
    predict_logits_with_dropout,
    truncate_and_mask,
)


def get_shuffle_iterator(
    names: List[str],
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    sft_mode: bool = False,
    pretrain_fraction: float = 1.0,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        pretrain_fraction: Fraction of the dataset to use for pretraining. If 1.0, use the whole dataset for pretraining.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        kwargs: this function should be "nice" and ignore other kwargs so that it can have a unified interface with our data selection. We don't use them here.
    """
    assert (
        n_epochs is not None or n_examples is not None
    ), "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            this_flat_data = []
            truncation_mode = "keep_end" if name == "hh" else "keep_start"
            dataset = get_dataset(name, split, silent=silent, cache_dir=cache_dir)
            for prompt, data in dataset.items():
                this_flat_data.append(
                    (
                        prompt,
                        data["responses"],
                        data["pairs"],
                        data["sft_target"],
                        truncation_mode,
                    )
                )
            if split == "train":
                split_idx = int(pretrain_fraction * len(this_flat_data))
                if sft_mode:
                    this_flat_data = this_flat_data[:split_idx]
                else:
                    this_flat_data = this_flat_data[split_idx:]
            flat_data.extend(this_flat_data)

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    is_train = split == "train"
    done = False
    dataset_is_online = names[0] == "haikus"

    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f"Finished generating {n_epochs} epochs on {split} split")
            break
        if shuffle:
            print("permutation_seeds", permutation_seeds, type(permutation_seeds))
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode or dataset_is_online:
                batch_element = tokenize_batch_element(
                    prompt,
                    sft_target,
                    sft_target,
                    truncation_mode,
                    tokenizer,
                    max_length,
                    max_prompt_length,
                )
                batch_element = {
                    k: v for k, v in batch_element.items() if "rejected" not in k
                }
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(
                                f"Finished generating {n_examples} examples on {split} split"
                            )
                        done = True

                    batch = []
            else:
                for p in pairs:
                    if done:
                        break
                    batch_element = tokenize_batch_element(
                        prompt,
                        responses[p[0]],
                        responses[p[1]],
                        truncation_mode,
                        tokenizer,
                        max_length,
                        max_prompt_length,
                    )
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(
                                    f"FINISHED {n_examples} EXAMPLES on {split} split"
                                )
                            done = True
                        batch = []

                    if not is_train:
                        break
        if done:
            break

        epoch_idx += 1


def get_active_iterator(
    names: List[str],
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    cache_dir: Optional[str] = None,
    policy: Optional[torch.nn.Module] = None,
    ref_policy: Optional[torch.nn.Module] = None,
    beta: float = 0.1,
    n_samples: int = 5,
    selection_strategy: str = "ae",  # 'ae' or 'us'
    selection_ratio: float = 3.0,
    active_beta: float = 2.0,
    active_minibatch_size: int = 10,
    pretrain_fraction: float = 0.0,
    **kwargs,
) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        policy: pointer to current model
        ref_policy: pointer to reference model
        beta: beta for dpo
        n_samples: number of samples to draw from the policy for uncertainty estimation
        selection_strategy: 'ae' or 'us' for active exploration or uncertainty sampling
        selection_ratio: how many times to select from the batch, e.g. 3 means we select 3 times more than batch size
        active_beta: beta for active exploration
        active_minibatch_size: minibatch size for active exploration
        pretrain_fraction: Fraction of the dataset to use for pretraining. If 1.0, use the whole dataset for pretraining.
        kwargs: this function should be "nice" and ignore other kwargs so that it can have a unified interface with our data selection. We don't use them here.
    """

    assert policy is not None, "need a model for the active iterator"

    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            this_flat_data = []
            truncation_mode = "keep_end" if name == "hh" else "keep_start"
            dataset = get_dataset(name, split, silent=silent, cache_dir=cache_dir)
            for prompt, data in dataset.items():
                this_flat_data.append(
                    (
                        prompt,
                        data["responses"],
                        data["pairs"],
                        data["sft_target"],
                        truncation_mode,
                    )
                )
            if split == "train":
                split_idx = int(pretrain_fraction * len(this_flat_data))
                this_flat_data = this_flat_data[split_idx:]
            flat_data.extend(this_flat_data)

    # should now have flat_data = [(prompt, responses, pairs, sft_target, truncation_mode), ...]

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f"Finished generating {n_epochs} epochs on {split} split")
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break
            for p in pairs:
                if done:
                    break
                batch_element = tokenize_batch_element(
                    prompt,
                    responses[p[0]],
                    responses[p[1]],
                    truncation_mode,
                    tokenizer,
                    max_length,
                    max_prompt_length,
                )
                batch.append(batch_element)
                example_idx += 1
                if len(batch) >= batch_size * selection_ratio:
                    collated_batch = collate_fn(batch)
                    if selection_strategy == "ae":
                        selected_batch = select_ae_elements(
                            batch=collated_batch,
                            num_to_select=batch_size,
                            policy=policy,
                            ref_policy=ref_policy,
                            n_samples=n_samples,
                            active_beta=active_beta,
                            active_minibatch_size=active_minibatch_size,
                        )
                    elif selection_strategy == "us":
                        selected_batch = select_us_elements(
                            batch=collated_batch,
                            num_to_select=batch_size,
                            policy=policy,
                            ref_policy=ref_policy,
                            n_samples=n_samples,
                            active_minibatch_size=active_minibatch_size,
                        )
                    else:
                        raise NotImplementedError(
                            f"Selection strategy {selection_strategy} not implemented"
                        )
                    yield selected_batch
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f"FINISHED {n_examples} EXAMPLES on {split} split")
                        done = True
                    batch = []
        if done:
            break

        epoch_idx += 1


def select_ae_elements(
    batch: List[Dict],
    num_to_select: int,
    policy: torch.nn.Module,
    ref_policy: torch.nn.Module,
    n_samples: int,
    active_beta: float = 2.0,
    active_minibatch_size: int = 10,
):

    device = next(policy.parameters()).device
    a1_input_ids = batch["chosen_input_ids"].to(device)
    a1_attention_mask = batch["chosen_attention_mask"].to(device)
    a1_labels = batch["chosen_labels"].to(device)
    a2_input_ids = batch["rejected_input_ids"].to(device)
    a2_attention_mask = batch["rejected_attention_mask"].to(device)
    a2_labels = batch["rejected_labels"].to(device)
    ga1_mean, ga1_variance = predict_logits_with_dropout(
        policy,
        a1_input_ids,
        a1_attention_mask,
        a1_labels,
        n_samples,
        minibatch_size=active_minibatch_size,
    )
    a1_mean = ga1_mean.to("cpu").float()
    a1_variance = ga1_variance.to("cpu").float()
    ga2_mean, ga2_variance = predict_logits_with_dropout(
        policy,
        a2_input_ids,
        a2_attention_mask,
        a2_labels,
        n_samples,
        minibatch_size=active_minibatch_size,
    )
    a2_mean = ga2_mean.to("cpu").float()
    a2_variance = ga2_variance.to("cpu").float()
    gref_logits_a1, todel1 = predict_logits_with_dropout(
        ref_policy,
        a1_input_ids,
        a1_attention_mask,
        a1_labels,
        1,
        minibatch_size=active_minibatch_size,
    )
    ref_logits_a1 = gref_logits_a1.to("cpu").float()
    gref_logits_a2, todel2 = predict_logits_with_dropout(
        ref_policy,
        a2_input_ids,
        a2_attention_mask,
        a2_labels,
        1,
        minibatch_size=active_minibatch_size,
    )
    ref_logits_a2 = gref_logits_a2.to("cpu").float()
    a1_std = torch.sqrt(a1_variance)
    a2_std = torch.sqrt(a2_variance)
    upper_bounds = torch.max(
        a1_mean + active_beta * a1_std - ref_logits_a1,
        a2_mean + active_beta * a2_std - ref_logits_a2,
    )
    lower_bounds = torch.max(
        a1_mean - active_beta * a1_std - ref_logits_a1,
        a2_mean - active_beta * a2_std - ref_logits_a2,
    )
    uncertainties = upper_bounds - lower_bounds
    values, indices = torch.topk(uncertainties, num_to_select, sorted=False)
    out_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out_batch[k] = v[indices, ...]
        else:
            out_batch[k] = [v[i] for i in indices.tolist()]
    return out_batch


def select_us_elements(
    batch: List[Dict],
    num_to_select: int,
    policy: torch.nn.Module,
    ref_policy: torch.nn.Module,
    n_samples: int,
    active_minibatch_size: int = 10,
):
    device = next(policy.parameters()).device
    a1_input_ids = batch["chosen_input_ids"].to(device)
    a1_attention_mask = batch["chosen_attention_mask"].to(device)
    a1_labels = batch["chosen_labels"].to(device)
    a2_input_ids = batch["rejected_input_ids"].to(device)
    a2_attention_mask = batch["rejected_attention_mask"].to(device)
    a2_labels = batch["rejected_labels"].to(device)
    ga1_mean, ga1_variance = predict_logits_with_dropout(
        policy,
        a1_input_ids,
        a1_attention_mask,
        a1_labels,
        n_samples,
        minibatch_size=active_minibatch_size,
    )
    a1_variance = ga1_variance.to("cpu").float()
    ga2_mean, ga2_variance = predict_logits_with_dropout(
        policy,
        a2_input_ids,
        a2_attention_mask,
        a2_labels,
        n_samples,
        minibatch_size=active_minibatch_size,
    )
    a2_variance = ga2_variance.to("cpu").float()
    uncertainties = (a1_variance + a2_variance) / 2
    values, indices = torch.topk(uncertainties, num_to_select, sorted=False)
    out_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out_batch[k] = v[indices, ...]
        else:
            out_batch[k] = [v[i] for i in indices.tolist()]
    return out_batch


def get_online_iterator(
    names: List[str],
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    selection_ratio: float = 3.0,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    cache_dir: Optional[str] = None,
    policy: Optional[torch.nn.Module] = None,
    ref_policy: Optional[torch.nn.Module] = None,
    n_samples: int = 5,
    selection_strategy: str = "borda",
    active_beta: float = 2.0,
    beta: float = 0.1,
    num_action_samples: int = 5,
    oracle: Optional[str] = None,
    active_minibatch_size: int = 16,
    gen_active_batch_size: int = 10,
    used_samples_num: int = 0,
    **kwargs,
) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
        policy: pointer to current model
        ref_policy: pointer to reference model
        n_samples: number of samples to draw from the policy for uncertainty estimation
        selection_strategy: 'ae' or 'us' for active exploration or uncertainty sampling
        active_beta: beta for active exploration
        beta: beta for dpo
        kwargs: this function should be "nice" and ignore other kwargs so that it can have a unified interface with our data selection. We don't use them here.
    """
    assert policy is not None, "need a model for the active iterator"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            this_flat_data = []
            truncation_mode = "keep_end" if name == "hh" else "keep_start"
            # this needs to be some kinda context only thing
            dataset = get_dataset(name, split, silent=silent, cache_dir=cache_dir)
            for prompt, data in dataset.items():
                # different than usual, we ignore the responses and pairs since we will need to generate them on the fly
                this_flat_data.append((prompt, data["sft_target"], truncation_mode))
            flat_data.extend(this_flat_data)

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f"Finished generating {n_epochs} epochs on {split} split")
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, sft_target, truncation_mode in flat_data:
            if done:
                break
            # the (for p in pairs) is removed because the pairs are not used and model is used to generate them
            batch_element = tokenize_batch_element(
                prompt,
                sft_target,
                sft_target,
                truncation_mode,
                tokenizer,
                max_length,
                max_prompt_length,
            )
            batch.append(batch_element)
            if len(batch) >= batch_size * selection_ratio:
                collated_batch = collate_fn(batch)
                if selection_strategy == "borda":
                    if selection_ratio == 1:
                        print(
                            "Warning: selection ratio is 1, so we are not actually doing Borda"
                        )
                    prompts, a_ids, a_prime_ids = select_borda_elements(
                        batch=collated_batch,
                        num_to_select=batch_size,
                        policy=policy,
                        ref_policy=ref_policy,
                        n_samples=n_samples,
                        beta=beta,
                        active_beta=active_beta,
                        pad_token_id=tokenizer.pad_token_id,
                        max_length=max_length,
                        num_action_samples=num_action_samples,
                        active_minibatch_size=active_minibatch_size,
                        gen_active_batch_size=gen_active_batch_size,
                    )
                elif selection_strategy == "uniref":
                    # since we are uniformly selecting contexts, we do NOT need extra data here
                    assert selection_ratio == 1
                    prompts, a_ids, a_prime_ids = select_uniref_elements(
                        batch=collated_batch,
                        num_to_select=batch_size,
                        policy=policy,
                        ref_policy=ref_policy,
                        n_samples=n_samples,
                        beta=beta,
                        active_beta=active_beta,
                        pad_token_id=tokenizer.pad_token_id,
                        num_action_samples=num_action_samples,
                        gen_active_batch_size=gen_active_batch_size,
                    )

                else:
                    raise NotImplementedError(
                        f"Selection strategy {selection_strategy} not implemented"
                    )
                actions = tokenizer.batch_decode(a_ids, skip_special_tokens=True)
                a_primes = tokenizer.batch_decode(a_prime_ids, skip_special_tokens=True)
                if names[0] in ["hh", "shp", "jeopardy"]:
                    best_actions = collated_batch["chosen_response_only"]
                else:
                    best_actions = None
                winners = asyncio.run(
                    get_winners(
                        names[0],
                        prompts,
                        best_actions=best_actions,
                        actions=actions,
                        a_primes=a_primes,
                        model=oracle,
                    )
                )
                selected_batch = []
                skips = 0
                for i in range(len(prompts)):
                    if winners[i] is None:
                        # openai failed
                        skips += 1
                        print(f"Winner {i} is None, probably because OpenAI failed")
                        continue
                    winner = actions[i] if winners[i] else a_primes[i]
                    loser = a_primes[i] if winners[i] else actions[i]
                    try:
                        elt = tokenize_batch_element(
                            prompts[i],
                            winner,
                            loser,
                            truncation_mode,
                            tokenizer,
                            max_length,
                            max_prompt_length,
                        )
                    except Exception:
                        print(
                            f"Failed to tokenize for \n{prompts[i]=}\n{winner=}\n{loser=}"
                        )
                        skips += 1
                        continue
                    if elt is None:
                        # tokenization failed for some reason (usually EOS in generation somehow)
                        print(
                            f"Tokenization failed for element {i}, probably because there was EOS in the string"
                        )
                        continue
                    selected_batch.append(elt)
                if skips > 0:
                    print(f"Skipping {skips} batch elements due to OpenAI errors")
                collated_selected_batch = collate_fn(selected_batch)
                example_idx += len(selected_batch)
                yield collated_selected_batch
                if n_examples is not None and example_idx >= n_examples:
                    if not silent:
                        print(f"FINISHED {n_examples} EXAMPLES on {split} split")
                    done = True
                batch = []
        if done:
            break

        epoch_idx += 1


def cat_pad_actions(action_list, pad_token_id: int):
    # Find the maximum size in each dimension, excluding the concatenation dimension
    max_width = max([x.shape[1] for x in action_list])
    padded_action_list = [
        F.pad(x, (0, max_width - x.shape[1]), "constant", pad_token_id)
        for x in action_list
    ]
    return torch.cat(padded_action_list, dim=0)


def select_borda_elements(
    batch: List[Dict],
    num_to_select: int,
    policy: torch.nn.Module,
    ref_policy: torch.nn.Module,
    n_samples: int,
    pad_token_id: int,
    active_beta: float = 2.0,
    beta: float = 0.1,
    max_length: int = 512,
    min_new_tokens: int = 1,
    num_action_samples: int = 5,
    active_minibatch_size: int = 16,
    gen_active_batch_size: int = 10,
):
    device = next(policy.parameters()).device
    # generate actions from current and reference policy
    big_batch = batch
    policy_actions = []
    ref_policy_actions = []
    acqs = []
    with torch.no_grad():
        for i in trange(
            0,
            len(big_batch["prompt_input_ids"]),
            gen_active_batch_size,
            desc="Generating candidate actions",
        ):
            batch = {}
            for k, v in big_batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[i : i + gen_active_batch_size].to(device)
                else:
                    batch[k] = v[i : i + gen_active_batch_size]
            this_batch_size = batch["prompt_input_ids"].shape[0]
            policy_output = policy.generate(
                inputs=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=max_length,
                do_sample=True,
                pad_token_id=pad_token_id,
                min_new_tokens=min_new_tokens,
                num_return_sequences=num_action_samples,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=pad_token_id,
            )
            reference_output = ref_policy.generate(
                inputs=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=max_length,
                do_sample=True,
                pad_token_id=pad_token_id,
                min_new_tokens=min_new_tokens,
                num_return_sequences=num_action_samples,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=pad_token_id,
            )

            # compute \pi(a |  x)
            prompt_len = batch["prompt_input_ids"].shape[1]
            policy_completion_ids = policy_output.sequences[:, prompt_len:]
            policy_completion_ids, policy_completion_mask = truncate_and_mask(
                policy_completion_ids, pad_token_id
            )
            prompt_input_ids = batch["prompt_input_ids"].repeat_interleave(
                num_action_samples, dim=0
            )
            prompt_attention_mask = batch["prompt_attention_mask"].repeat_interleave(
                num_action_samples, dim=0
            )
            policy_full_ids = torch.cat(
                [prompt_input_ids, policy_completion_ids], axis=1
            )
            policy_full_mask = torch.cat(
                [prompt_attention_mask, policy_completion_mask], axis=1
            )
            policy_prompt_labels = torch.ones_like(prompt_input_ids) * -100
            policy_completion_labels = (
                policy_completion_ids * (policy_completion_mask)
                + -100 * (~policy_completion_mask).int()
            )
            policy_labels = torch.cat(
                [policy_prompt_labels, policy_completion_labels], axis=1
            )
            mean_pi_logits_a, var_pi_logits_a = predict_logits_with_dropout(
                policy,
                policy_full_ids,
                policy_full_mask,
                policy_labels,
                n_samples,
                minibatch_size=active_minibatch_size,
            )
            pi_ref_logits_a, _ = predict_logits_with_dropout(
                ref_policy,
                policy_full_ids,
                policy_full_mask,
                policy_labels,
                1,
                minibatch_size=active_minibatch_size,
            )
            pi_ref_logits_a = pi_ref_logits_a.reshape(
                (this_batch_size, num_action_samples)
            )
            pi_logits_a_ucb = (
                mean_pi_logits_a + active_beta * torch.sqrt(var_pi_logits_a)
            ).reshape((this_batch_size, num_action_samples))
            pi_logits_a_lcb = (
                mean_pi_logits_a - active_beta * torch.sqrt(var_pi_logits_a)
            ).reshape((this_batch_size, num_action_samples))

            # compute \pi(a'; |  x)
            ref_policy_completion_ids = reference_output.sequences[:, prompt_len:]
            ref_policy_completion_ids, ref_policy_completion_mask = truncate_and_mask(
                ref_policy_completion_ids, pad_token_id
            )
            ref_policy_full_ids = torch.cat(
                [prompt_input_ids, ref_policy_completion_ids], axis=1
            )
            ref_policy_full_mask = torch.cat(
                [prompt_attention_mask, ref_policy_completion_mask], axis=1
            )
            ref_policy_completion_labels = (
                ref_policy_completion_ids * (ref_policy_completion_mask)
                + -100 * (~ref_policy_completion_mask).int()
            )
            ref_policy_labels = torch.cat(
                [policy_prompt_labels, ref_policy_completion_labels], axis=1
            )
            mean_pi_logits_a_prime, var_pi_logits_a_prime = predict_logits_with_dropout(
                policy,
                ref_policy_full_ids,
                ref_policy_full_mask,
                ref_policy_labels,
                n_samples,
                minibatch_size=active_minibatch_size,
            )
            pi_ref_logits_a_prime, _ = predict_logits_with_dropout(
                ref_policy,
                ref_policy_full_ids,
                ref_policy_full_mask,
                ref_policy_labels,
                n_samples,
                minibatch_size=active_minibatch_size,
            )
            pi_ref_logits_a_prime = pi_ref_logits_a_prime.reshape(
                (this_batch_size, num_action_samples)
            )
            pi_logits_a_prime_ucb = (
                mean_pi_logits_a_prime + active_beta * torch.sqrt(var_pi_logits_a_prime)
            ).reshape((this_batch_size, num_action_samples))
            pi_logits_a_prime_lcb = (
                mean_pi_logits_a_prime - active_beta * torch.sqrt(var_pi_logits_a_prime)
            ).reshape((this_batch_size, num_action_samples))
            ucb_a_prime_term = pi_logits_a_prime_lcb - pi_ref_logits_a_prime
            ucb_a_term = pi_ref_logits_a - pi_logits_a_ucb
            lcb_a_prime_term = pi_logits_a_prime_ucb - pi_ref_logits_a_prime
            lcb_a_term = pi_ref_logits_a - pi_logits_a_lcb
            ucb_logits = beta * (ucb_a_term[:, :, None] + ucb_a_prime_term[:, None, :])
            lcb_logits = beta * (lcb_a_term[:, :, None] + lcb_a_prime_term[:, None, :])
            ucb_logistics = 1 / (1 + torch.exp(ucb_logits))
            lcb_logistics = 1 / (1 + torch.exp(lcb_logits))
            ucb_borda = torch.mean(ucb_logistics, dim=2)
            lcb_borda = torch.mean(lcb_logistics, dim=2)
            ucb_values, ucb_indices = ucb_borda.max(dim=1)
            acq = ucb_values - lcb_borda.max(dim=1).values
            acqs.append(acq)
            ref_actions = ref_policy_completion_ids.reshape(
                (this_batch_size, num_action_samples, -1)
            )
            random_indices = torch.randint(0, num_action_samples, (this_batch_size,))
            sampled_ref_actions = ref_actions[
                torch.arange(this_batch_size), random_indices, :
            ]
            ref_policy_actions.append(sampled_ref_actions)
            pi_actions = policy_completion_ids.reshape(
                (this_batch_size, num_action_samples, -1)
            )
            chosen_pi_actions = pi_actions[
                torch.arange(this_batch_size), ucb_indices, :
            ]
            policy_actions.append(chosen_pi_actions)
        acqs = torch.cat(acqs, dim=0)
        policy_actions = cat_pad_actions(policy_actions, pad_token_id=pad_token_id)
        ref_policy_actions = cat_pad_actions(
            ref_policy_actions, pad_token_id=pad_token_id
        )
        contexts = big_batch["prompt"]

        values, indices = torch.topk(acqs, num_to_select, sorted=False)
        selected_policy_actions = policy_actions[indices, :]
        selected_ref_policy_actions = ref_policy_actions[indices, :]
        selected_contexts = [contexts[i] for i in indices.tolist()]
        return selected_contexts, selected_policy_actions, selected_ref_policy_actions


def select_uniref_elements(
    batch: List[Dict],
    num_to_select: int,
    policy: torch.nn.Module,
    ref_policy: torch.nn.Module,
    n_samples: int,
    pad_token_id: int,
    max_length: int = 512,
    min_new_tokens: int = 4,
    gen_active_batch_size: int = 10,
    **kwargs,
):
    device = next(policy.parameters()).device
    # generate actions from current and reference policy
    big_batch = batch
    policy_actions = []
    ref_policy_actions = []
    with torch.no_grad():
        for i in trange(
            0,
            len(big_batch["prompt_input_ids"]),
            gen_active_batch_size,
            desc="Generating candidate actions",
        ):
            batch = {}
            for k, v in big_batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v[i : i + gen_active_batch_size].to(device)
                else:
                    batch[k] = v[i : i + gen_active_batch_size]
            policy_output = policy.generate(
                inputs=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=max_length,
                do_sample=True,
                pad_token_id=pad_token_id,
                min_new_tokens=min_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=pad_token_id,
            )
            reference_output = ref_policy.generate(
                inputs=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=max_length,
                do_sample=True,
                pad_token_id=pad_token_id,
                min_new_tokens=min_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=pad_token_id,
            )
            prompt_len = batch["prompt_input_ids"].shape[1]
            policy_completion_ids = policy_output.sequences[:, prompt_len:]
            ref_policy_completion_ids = reference_output.sequences[:, prompt_len:]
            policy_actions.append(policy_completion_ids)
            ref_policy_actions.append(ref_policy_completion_ids)
        policy_actions = cat_pad_actions(policy_actions, pad_token_id=pad_token_id)
        ref_policy_actions = cat_pad_actions(
            ref_policy_actions, pad_token_id=pad_token_id
        )
        contexts = big_batch["prompt"]
        return contexts, policy_actions, ref_policy_actions
