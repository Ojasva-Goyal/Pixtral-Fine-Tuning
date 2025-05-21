"""
train.py - Pixtral-12B Fine-Tuning Script for Instruction-Following Multimodal Tasks

This script fine-tunes the Pixtral-12B model using parameter-efficient LoRA adapters.
It uses vision-language instruction-style data in a JSON format and supports [IMG] tokens.

Author: Ojasva Goyal
"""

import os
# Disable HF Hub transfer and set visible CUDA devices
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import torch
from datasets import Features, Array3D, Value
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate
import numpy as np
from transformers import AutoConfig
from transformers import BitsAndBytesConfig
from torchvision.transforms.functional import to_pil_image, resize


# === Load Model & Processor ===
# Load the Pixtral-12B model and processor
model_id = "mistral-community/pixtral-12b"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for reduced memory usage
    device_map="auto"  # Automatically map model to available devices
    # attn_implementation="flash_attention_2"  # Optional: Uncomment for faster attention
)
processor = AutoProcessor.from_pretrained(model_id)

# 🔧 Force all images to be resized to a fixed size (for consistent [IMG] tokens)
processor.image_processor.size = {"height": 336, "width": 336}


def get_image_token_count(image, dummy_text="describe this image"):
    """
    Compute the number of tokens generated for an image using the model's vision tower.
    Returns 0 if token computation fails.
    """
    try:
        inputs = processor(images=image, text=dummy_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.vision_tower(pixel_values=inputs["pixel_values"])
        token_count = output.last_hidden_state.shape[1]
        if token_count == 0:
            raise ValueError("Image token count is zero.")
        return token_count
    except Exception as e:
        print(f"[ERROR] Failed to compute image tokens: {e}")
        return 0  # Return zero to flag as invalid


CHAT_TEMPLATE = """
{%- for message in messages %}
  {%- if message.role == "user" %}
      <s>[INST]
      {%- for item in message.content %}
          {%- if item.type == "text" %}
              {{ item.text }}
          {%- elif item.type == "image" %}
              \n[IMG]
          {%- endif %}
      {%- endfor %}
      [/INST]
  {%- elif message.role == "assistant" %}
      {%- for item in message.content %}
          {%- if item.type == "text" %}
              {{ item.text }}
          {%- endif %}
      {%- endfor %}
      </s>
  {%- endif %}
{%- endfor %}
"""

# Set the chat template for tokenization
processor.tokenizer.chat_template = CHAT_TEMPLATE

# Define special tokens for the tokenizer
processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.add_special_tokens({'additional_special_tokens': ['[IMG]']})

# Load the dataset in JSON format
with open("/path/to/dataset/converted_pixtral_format_Topic.json", "r") as f: 
    raw_data = json.load(f)

# Extract the first conversation
messages = raw_data[0]["messages"]

# Apply the template
formatted_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

print(formatted_text)

def prepare_sample_batch(raw_data, count):
    """
    Convert raw JSON dataset into the format expected by the DataCollator.
    """
    sample_batch = []
    for item in raw_data[:count]:
        try:
            messages = item["messages"]
            question = messages[0]["content"][0]["text"]
            image_path = messages[0]["content"][1]["image_path"]
            answer = messages[1]["content"][0]["text"]
            image = Image.open(image_path).convert("RGB")
            # image = image.resize((336, 336))

            sample_batch.append({
                "question": question,
                "answer": answer,
                "image": image,
                "image_path": image_path
            })
        except Exception as e:
            print(f"Skipping sample due to error: {e}")
    return sample_batch

train_dataset = prepare_sample_batch(raw_data, count=len(raw_data)-1)  # Prepare training dataset
eval_dataset = prepare_sample_batch(raw_data, count=1)  # Prepare evaluation dataset

class MyDataCollator:
    """
    Custom data collator to process and tokenize multimodal data (text + images).
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        """
        Process a batch of examples into tokenized inputs and labels.
        """
        texts = []
        images = []
        assistant_responses = []  # Track assistant responses to mask correctly
        max_tokens = 2048

        for example in examples:
            image = example["image"]
            question = example["question"]
            answer = example["answer"]

            # 🔁 Get number of image tokens for this image
            img_count = get_image_token_count(image)

            if img_count == 0:
                print(f"[SKIP] Skipping sample due to invalid image token count.")
                print(f"[WARN] Image caused failure: {example['image_path']}")
                continue

            # Tokenize answer only
            answer_ids = self.processor.tokenizer(answer, truncation=True, max_length=512)["input_ids"]
            answer_token_count = len(answer_ids)

            # Compute max allowable user tokens = total - [IMG] - image tokens - answer
            max_user_tokens = max_tokens - img_count - answer_token_count - 31  # 20 for buffer and special tokens

            # Tokenize and trim question if needed
            user_tokens = self.processor.tokenizer.tokenize(question)
            if len(user_tokens) > max_user_tokens:
                trimmed_question = self.processor.tokenizer.convert_tokens_to_string(user_tokens[:max_user_tokens])
                question = trimmed_question

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        # {"type": "text", "text": "\n".join(["[IMG]"] * img_count)}
                        {"type": "image"}  # No need to include path; image is already loaded
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]

            # Convert to prompt using the processor’s template
            text = self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )

            texts.append(text.strip())
            images.append(image)
            assistant_responses.append(answer)

        # Tokenize everything
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2048
        )

        # Prepare labels (we'll mask everything except assistant tokens)
        labels = batch["input_ids"].clone()

        for i, (input_ids, response) in enumerate(zip(batch["input_ids"], assistant_responses)):
            # Tokenize just the assistant response
            assistant_tokens = self.processor.tokenizer(
                response, return_tensors="pt", truncation=True, max_length=512
            )["input_ids"][0][1:]  # remove BOS token

            # Find where the assistant tokens begin
            start_idx = self.find_subsequence(input_ids, assistant_tokens)

            if start_idx is not None:
                # Mask everything before the assistant response
                labels[i, :start_idx] = -100
                # Optionally mask anything after the assistant response
                labels[i, start_idx + len(assistant_tokens):] = -100
            else:
                # If not found, mask everything
                labels[i] = -100

        batch["labels"] = labels
        return batch
    
    def find_subsequence(self, sequence, subsequence):
        """
        Helper function to find the starting index of a subsequence within a sequence.
        """
        seq_len = len(sequence)
        sub_len = len(subsequence)
        for i in range(seq_len - sub_len + 1):
            if torch.equal(sequence[i:i+sub_len], subsequence):
                return i
        return None
data_collator = MyDataCollator(processor)  # Initialize the custom data collator

processed_batch = data_collator(train_dataset)  # Process the training dataset

# Print processed batch keys for debugging
print("Processed batch keys:", processed_batch.keys())
print("\nTokenized input IDs (before padding):")
print(processed_batch["input_ids"])

# Decode tokens to readable text
# for input_id in processed_batch["input_ids"]:
#     print("\nDecoded Prompt:")
#     print(processor.tokenizer.decode(input_id, skip_special_tokens=False))

lora_config = LoraConfig(
    r=32,  # Rank - Higher values for larger datasets
    lora_alpha=32,  # Scaling factor for LoRA
    use_rslora=True,  # Use RS-LoRA for better performance
    target_modules="all-linear",  # Apply LoRA to all linear layers
    lora_dropout=0.1,  # Dropout rate for LoRA
    bias="none",  # No bias adjustment
    task_type="CAUSAL_LM"  # Task type: Causal Language Modeling
)

# Wrap the model with LoRA configuration
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable parameters for debugging

# Training configuration
epochs = 3  # Number of training epochs
lr = 3e-5  # Learning rate
schedule = "constant"  # Learning rate schedule

training_args = TrainingArguments(
    num_train_epochs=epochs,
    per_device_train_batch_size=3,  # Batch size per device for training
    per_device_eval_batch_size=1,  # Batch size per device for evaluation
    gradient_accumulation_steps=1,  # Accumulate gradients over multiple steps
    learning_rate=lr,
    weight_decay=0.01,  # Weight decay for regularization
    logging_steps=0.1,  # Log every 0.1 steps
    output_dir="/path/to/output/fine-tuned-model", # Output directory for saving the model
    eval_strategy="steps",  # Evaluation strategy
    eval_steps=0.2,  # Evaluate every 0.2 steps
    lr_scheduler_type=schedule,  # Learning rate scheduler type
    bf16=True,  # Use bfloat16 precision
    remove_unused_columns=False,  # Keep all columns in the dataset
    gradient_checkpointing=True,  # Enable gradient checkpointing
    gradient_checkpointing_kwargs={'use_reentrant': True}  # Additional checkpointing options
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=processor.tokenizer  # Tokenizer for processing
)

trainer.train()  # Start training
