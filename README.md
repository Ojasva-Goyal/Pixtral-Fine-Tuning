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

---

## 📐 Dataset Format

Each file is a **list of conversations**. Every message can contain `text` and/or `image` parts:

```jsonc
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text",  "text": "What’s in this image?" },
        { "type": "image", "image_path": "img/apple.jpg" }
      ]
    },
    {
      "role": "assistant",
      "content": [
        { "type": "text",  "text": "A red apple on a wooden table." }
      ]
    }
  ]
}
```

Place your JSON under `data/` and point `--train_json` / `--eval_json` to the files.

---

## ⚡ Quick Start

```bash
# clone repo
git clone https://github.com/<your‑handle>/pixtral-finetune.git
cd pixtral-finetune
```

# 1️⃣  Install dependencies (conda preferred)
```bash
conda env create -f environment.yml   # full spec
conda activate pixtral-ft
#  └─ or:  pip install -r requirements.txt  # minimal spec
```

# 2️⃣  Launch training
```bash
python scripts/train.py \
  --model_id mistral-community/pixtral-12b \
  --train_json data/train.json \
  --eval_json  data/val.json \
  --output_dir out/pixtral-ft
```

*Run `python scripts/train.py --help` to see all flags.*

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


## 🖥️ Inference

A runnable demo lives in [`examples/inference.py`](examples/inference.py):

```bash
python examples/inference.py \
  --adapter_path out/pixtral-ft \
  --image demo.jpg \
  --prompt "Describe this image."
```

---

## 🧯 Troubleshooting

| Issue                  | Hint                                                                         |
| ---------------------- | ---------------------------------------------------------------------------- |
| **CUDA out of memory** | Lower `--batch_size`, increase gradient accumulation, or enable Flash‑Attn 2 |
| **Image token error**  | Ensure images are RGB and ≤ 4096 px on the long side                         |
| **Sequence too long**  | Shorten prompts or raise `--max_seq_len`                                     |

---

# Contributing
PRs & issues welcome! 🎉 Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.


---


# Contact
Created by `Ojasva Goyal` - feel free to contact me at ojasvagoyal9@gmail.com for any questions or feedback.





