"""
inference.py - CLI-based single-image inference using fine-tuned Pixtral-12B

Usage:
    python examples/inference.py \
        --adapter_path out/pixtral-ft \
        --image demo.jpg \
        --prompt "Describe this image."
"""

import argparse
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel


def load_model_and_processor(base_model_id, adapter_path, device="cuda"):
    base_model = LlavaForConditionalGeneration.from_pretrained(base_model_id, device_map=device)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    processor = AutoProcessor.from_pretrained(base_model_id)
    return model.eval(), processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True, help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Prompt to query the image")
    args = parser.parse_args()

    model_id = "mistral-community/pixtral-12b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = load_model_and_processor(model_id, args.adapter_path, device)

    # Load and preprocess image
    image = Image.open(args.image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image"}
            ]
        }
    ]

    # Apply prompt template and tokenize
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text.strip()],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)

    # Run generation
    generate_ids = model.generate(**inputs, max_new_tokens=64)
    generated_text = processor.batch_decode(
        generate_ids[:, inputs["input_ids"].size(1):],
        skip_special_tokens=True
    )

    print(f"🧠 Model Output:
{generated_text[0]}")


if __name__ == "__main__":
    main()
