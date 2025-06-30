# CUDA_VISIBLE_DEVICES=1,0 python train_medgemma.py for GPDS 67 Machine

from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from datasets import load_dataset, Dataset
import torch
from PIL import Image
import os
import json
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO
from peft import LoraConfig, get_peft_model


# Path to local base model.
model_name = "/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma/cache/models--google--medgemma-4b-it/snapshots/698f7911b8e0569ff4ebac5d5552f02a9553063c"
# Path to training data (JSON).
dataset_path = "/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma_GigaV2/training_dataset_gigav2.json"  # IMPORTANT: Update this path
output_model_dir = "/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma_GigaV2"
# Max model input sequence length.
max_seq_length = 2048


# 4-bit quantization configuration.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for compute stability.
)

# Load model with quantization and device mapping.
print("Loading MedGemma model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load processor for tokenization and image processing.
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = processor.tokenizer

# Set pad token if not defined.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# LoRA configuration for PEFT.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Apply LoRA to the model.
model = get_peft_model(model, lora_config)
# Enable gradient checkpointing.
model.gradient_checkpointing_enable()  # type: ignore
# Show number of trainable parameters.
model.print_trainable_parameters()

# Preprocesses a single data sample.
def preprocess_data(example: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    try:
        image = Image.open(example["image_path"]).convert("RGB")
        
        lang = example.get("language", "en")  # Default to English

        if lang == "pt":
            system_prompt_template = "Você é um radiologista especialista. Estas são imagens de radiografia processadas pelas redes EfficientNet, FastViT, FasterCNN e UNet. Sua tarefa é analisar essas imagens fornecidas e fornecer um relatório detalhado, estruturado e medicamente preciso com base na solicitação do usuário. Este paciente tem indicações de Ateroma: {atheroma} e Osteoporose: {osteo}."
            user_prompt_text = "Analise e descreva os resultados extraídos desta imagem fornecida."
        else:  # Default to English
            system_prompt_template = "You are an expert radiologist. These are radiographic images processed by the EfficientNet, FastViT, FasterCNN, and UNet networks. Your task is to analyze these provided images and provide a detailed, structured, and medically accurate report based on the user's request. This patient has indications of Atheroma: {atheroma} and Osteoporosis: {osteo}."
            user_prompt_text = "Analyze and describe the results extracted from this provided image."

        system_prompt = system_prompt_template.format(
            atheroma=example['atheroma'], 
            osteo=example['osteo']
        )
        # Chat-formatted input for the model.
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_text},
                    {"type": "image", "image": image},
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["text"]}]  # Ground truth label
            }
        ]
        # Tokenize and process the chat template for training.
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            max_length=max_seq_length,
            padding="max_length",
            truncation=True
        )

        # Remove batch dimension.
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        # Set labels for loss calculation.
        inputs["labels"] = inputs["input_ids"].clone()
        
        return inputs
    except Exception as e:
        # Skip corrupted samples.
        print(f"Skipping corrupted sample {example.get('image_path', 'N/A')}: {e}")
        return None

print("Loading dataset...")
if os.path.exists(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    dataset = Dataset.from_list(json_data)
    
    # Remove samples with missing image paths.
    initial_count = len(dataset)
    dataset = dataset.filter(lambda x: x.get('image_path') and os.path.exists(x['image_path']))
    filtered_count = len(dataset)
    if initial_count > filtered_count:
        print(f"Warning: Removed {initial_count - filtered_count} samples with invalid or missing image paths.")

    # Preprocess the entire dataset.
    print("Processing and tokenizing dataset...")
    processed_dataset = dataset.map(
        preprocess_data,
        remove_columns=dataset.column_names
    )
    
    # Filter out samples that failed preprocessing.
    valid_indices = [i for i, x in enumerate(processed_dataset) if x is not None]
    processed_dataset = processed_dataset.select(valid_indices)
    
    print(f"Dataset loaded and processed with {len(processed_dataset)} samples.")
        
else:
    # Exit if dataset JSON is not found.
    print(f"Dataset not found at {dataset_path}")
    print("Please update the dataset_path variable with your actual dataset file.")
    print("\nExpected JSON format: [{\"image_path\": \"...\", \"text\": \"...\"}]")
    exit(1)

training_args = TrainingArguments(
    output_dir=f"{output_model_dir}/training_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch size = 4
    warmup_steps=10,
    learning_rate=2e-4,
    bf16=True,  # Use bfloat16 for mixed-precision training.
    logging_steps=1,
    save_steps=100,
    eval_strategy="no",  # Disable evaluation during training.
    save_total_limit=2,  # Limit total number of saved checkpoints.
    remove_unused_columns=False,
    push_to_hub=False,
    report_to=None,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory.
    gradient_checkpointing_kwargs={'use_reentrant': False}, # Make gradient checkpointing PEFT-compatible.
)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

# Start training.
print("Starting training...")
trainer.train()

# Save the LoRA adapter weights.
print("Saving model...")
trainer.save_model(output_model_dir)

print("Training completed!")
print(f"Model saved to {output_model_dir}")

# Generates a description for an image.
def generate_medical_description(image_path: str, prompt: str = "Analyze this medical image and provide a detailed description.") -> str:
    # Set model to evaluation mode.
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    
    # Format input for inference.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]
    
    # Process input for generation.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate text.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated tokens.
    input_len = inputs['input_ids'].shape[-1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    return response

# Example inference call.
# print("--- Running example inference ---")
# description = generate_medical_description("path/to/your/test/image.jpg")
# print(description)