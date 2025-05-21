# 🚀 Pixtral‑12B Instruction Fine‑Tuning

Fine‑tune the multimodal **Pixtral‑12B** model (Mistral × PixArt) on custom vision‑language instruction datasets using **LoRA** adapters and Hugging Face's 🤗 ecosystem.

---

## 🔥 Highlights
- 🧠 **Lightweight LoRA tuning** (~3% trainable params)
- 🎯 Supports multimodal JSON with `[IMG]` token injection
- 📦 Self-contained `train.py` script powered by 🤗 PEFT + `Trainer`
- 🚀 Compatible with Flash-Attn 2 for faster training (optional)
- 🧩 Easily pluggable into Hugging Face Hub

---
## ⚙️ Setup: Environment Installation

You can install all dependencies using the provided `environment.yml` file (recommended for conda users):

```bash
# Step 1: Create conda environment from YAML
conda env create -f environment.yml

# Step 2: Activate the environment
conda activate pixtral-ft
```

If you prefer `pip`, use the `requirements.txt` instead:

```bash
pip install -r requirements.txt
```


## ⚙️ Setup: Environment Installation

You can install all dependencies using the provided `environment.yml` file (recommended for conda users):

```bash
# Step 1: Create conda environment from YAML
conda env create -f environment.yml

# Step 2: Activate the environment
conda activate pixtral-ft
```

If you prefer `pip`, use the `requirements.txt` instead:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Script Arguments

Here are the most commonly used arguments in `train.py`:

| Argument                    | Description                                                              | Example                                      |
|-----------------------------|---------------------------------------------------------------------------|----------------------------------------------|
| `--model_id`                | Base model to fine-tune                                                   | `mistral-community/pixtral-12b`              |
| `--train_json`              | Path to training dataset JSON                                             | `data/train.json`                            |
| `--eval_json`               | Path to validation dataset JSON                                           | `data/val.json`                              |
| `--output_dir`              | Where to save checkpoints and adapters                                    | `out/pixtral-ft`                             |
| `--epochs`                  | Number of training epochs                                                 | `3`                                          |
| `--lr`                      | Learning rate                                                             | `3e-5`                                       |
| `--batch_size`              | Per-device batch size                                                     | `3`                                          |
| `--gradient_accumulation_steps` | Steps to accumulate gradients (useful for small VRAM)             | `4`                                          |
| `--flash_attn`              | Enable Flash-Attn 2 for faster attention (if available)                   | *(flag only, no value needed)*               |
| `--push_to_hub`             | Push final model to Hugging Face Hub                                     | *(flag only)*                                |

To see the full list of arguments at any time:

```bash
python scripts/train.py --help
```

## 🛠️ Script Arguments

Here are the most commonly used arguments in `train.py`:

| Argument                    | Description                                                              | Example                                      |
|-----------------------------|---------------------------------------------------------------------------|----------------------------------------------|
| `--model_id`                | Base model to fine-tune                                                   | `mistral-community/pixtral-12b`              |
| `--train_json`              | Path to training dataset JSON                                             | `data/train.json`                            |
| `--eval_json`               | Path to validation dataset JSON                                           | `data/val.json`                              |
| `--output_dir`              | Where to save checkpoints and adapters                                    | `out/pixtral-ft`                             |
| `--epochs`                  | Number of training epochs                                                 | `3`                                          |
| `--lr`                      | Learning rate                                                             | `3e-5`                                       |
| `--batch_size`              | Per-device batch size                                                     | `3`                                          |
| `--gradient_accumulation_steps` | Steps to accumulate gradients (useful for small VRAM)             | `4`                                          |
| `--flash_attn`              | Enable Flash-Attn 2 for faster attention (if available)                   | *(flag only, no value needed)*               |
| `--push_to_hub`             | Push final model to Hugging Face Hub                                     | *(flag only)*                                |

To see the full list of arguments at any time:

```bash
python scripts/train.py --help
```
