"""
inference_with_adapters.py - Run inference on fine-tuned Pixtral-12B model

Loads the Pixtral-12B base model and applies LoRA adapters from the given path.
Then runs inference on a few samples from a provided dataset.

Author: Ojasva Goyal
"""

import torch
from PIL import Image
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel


def load_model_and_processor(base_model_id, adapter_path, device="cuda"):
    print(f"🔧 Loading base model: {base_model_id}")
    base_model = LlavaForConditionalGeneration.from_pretrained(base_model_id, device_map=device)

    print(f"🎯 Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)

    processor = AutoProcessor.from_pretrained(base_model_id)
    return model.eval(), processor


def run_model_evaluation(model, processor, dataset, num_samples=None, device="cuda", constant_query=None):
    results = []

    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(num_samples))

    for example in dataset:
        image = example["image"]
        query = constant_query if constant_query else example["query"]["en"]

        # Display image (optional)
        aspect_ratio = image.width / image.height
        new_width = 300
        new_height = int(new_width / aspect_ratio)
        display_image = resize(image, (new_width, new_height))
        display_image.show()

        # Build vision-language prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"}
                ]
            }
        ]

        formatted_prompt = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        print(f"📝 Prompt: {formatted_prompt}")

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=[text.strip()],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(device)

        # Generate prediction
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=64
        )
        generated_text = processor.batch_decode(
            generate_ids[:, inputs["input_ids"].size(1):],
            skip_special_tokens=True
        )
        print(f"🧠 Prediction: {generated_text}")
        results.append(generated_text[0])

    return results


# Example usage
if __name__ == "__main__":
    # Paths
    adapter_path = "out/pixtral-ft"
    base_model_id = "mistral-community/pixtral-12b"

    # Dummy dataset loading (replace with real dataset)
    # from datasets import load_dataset
    # eval_dataset = load_dataset(...)

    print("⚙️ Loading model and processor...")
    model, processor = load_model_and_processor(base_model_id, adapter_path)

    # eval_results = run_model_evaluation(model, processor, eval_dataset, num_samples=3, device="cuda", constant_query="Describe the image.")
    # for result in eval_results:
    #     print(result)
