# Sample-Efficient Preference Alignment in LLMs via Active Exploration

## Overview

This repository contains the official implementation of **AE-BORDA-DPO** and **AE-DPO**, methods we proposed in the [Sample-Efficient Preference Alignment in LLMs via Active Exploration](https://arxiv.org/abs/2312.00267) paper.

These methods extend the Direct Preference Optimization (DPO) pipeline by introducing **active exploration** strategies that reduce the number of preference queries needed during training.

---

## What's in this repo?

The DPO pipeline we build upon has two main stages:

1. **Supervised fine-tuning (SFT)** on an initial dataset.
2. **Active preference optimization** to refine the model using preference pairs.

Our codebase is structured similarly to the [original DPO implementation](https://github.com/eric-mitchell/direct-preference-optimization), and supports both **offline** and **online** active learning setups.

---

### Key Components

- `train.py`: Main entry point for training across all methods. Supports both base and QLoRA-based models.
- `trainers.py`: Training loop logic, including batch-wise active data selection.
- `data_selection.py`: Implementations of AE, Borda, Uniform Sampling, and Uncertainty selection strategies.
- `preference_datasets.py`: Dataset loading and oracle integration. **To train on your own data, modify this file.**
- `utils.py`: Shared utilities.
- `data/`: Contains two contributed datasets — **Jeopardy!** and **Haikus**.
- `config/`: Configuration files. Variables may be set in config files or via command-line flags.
- `setup_nltk.py`: Setup nltk library if the Haiku dataset is needed.

---

## Running Online Active Learning (AE-BORDA-DPO, Uniform-DPO)

These methods assume that preference pairs are generated **online** by sampling from the model and sending them to an **oracle** (e.g., GPT-4o) for preference labeling.

Since all offline data can be used during SFT, the `pretrain_fraction` parameter is ignored in this setting.

```bash
python -u train.py model=gpt2-large datasets=[hh] loss=dpo loss.beta=0.1 \
  model.archive=sft_policy.pt exp_name=hh_borda_gpt2-large \
  gradient_accumulation_steps=2 batch_size=32 eval_batch_size=16 \
  trainer=BasicTrainer sample_during_eval=true online=true selection_strategy=borda
````

```bash
python -u train.py model=gpt2-large datasets=[hh] loss=dpo loss.beta=0.1 \
  model.archive=sft_policy.pt exp_name=hh_uniref_gpt2-large \
  gradient_accumulation_steps=2 batch_size=32 eval_batch_size=16 \
  trainer=BasicTrainer sample_during_eval=true online=true selection_ratio=1 \
  selection_strategy=uniref
```

---

## Running Offline Active Learning (AE-DPO, US-DPO)

These methods simulate active learning on **pre-collected preference data**. To ensure a realistic evaluation, data used for SFT is excluded from the active learning phase using the `pretrain_fraction` parameter.

```bash
python -u train.py model=gpt2-large datasets=[hh] loss=dpo loss.beta=0.1 \
  model.archive=sft_policy exp_name=hh_ae_gpt2-large \
  gradient_accumulation_steps=2 batch_size=32 eval_batch_size=16 \
  trainer=BasicTrainer sample_during_eval=true pretrain_fraction=0.3 \
  active=true qlora=true selection_strategy=ae
```

```bash
python -u train.py model=gpt2-large datasets=[hh] loss=dpo loss.beta=0.1 \
  model.archive=sft_policy exp_name=hh_us_gpt2-large \
  gradient_accumulation_steps=2 batch_size=32 eval_batch_size=16 \
  trainer=BasicTrainer sample_during_eval=true pretrain_fraction=0.3 \
  active=true qlora=true selection_strategy=us
```

---

> **Note:** Some environment variables must be set depending on the method and configuration you are running:
>
> * `HF_TOKEN`: Required for downloading certain Hugging Face models or datasets.
> * `WANDB_API_KEY`: Required if Weights & Biases (`wandb`) is enabled in the config file.
> * `OPENAI_API_KEY`: Required if you are using any **online methods** (e.g., AE-BORDA-DPO) that query an API-based oracle such as OpenAI.

---

## Citing

If you find this work helpful in your research or applications, please consider citing:

```bibtex
@article{mehta2023sample,
  title   = {Sample Efficient Preference Alignment in LLMs via Active Exploration},
  author  = {Viraj Mehta and Syrine Belakaria and Vikramjeet Das and Ojash Neopane and Yijia Dai and Ilija Bogunovic and Barbara Engelhardt and Stefano Ermon and Jeff Schneider and Willie Neiswanger},
  journal = {arXiv preprint arXiv:2312.00267},
  year    = {2023},
  url     = {https://arxiv.org/abs/2312.00267}
}
```

---

## Acknowledgments

This project builds upon the [Direct Preference Optimization (DPO)](https://github.com/eric-mitchell/direct-preference-optimization) framework by Eric Mitchell et al. We thank the authors for their open-source contributions.