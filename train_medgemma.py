from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from datasets import load_dataset, Dataset
import torch
from PIL import Image
import os
import json
from typing import Dict, List, Any
import base64
from io import BytesIO

# Configuration
# Path to the training dataset JSON file
# Please update this path to your actual dataset file
dataset_path = "/path/to/your/dataset.json" # CHANGE THIS

max_seq_length = 2048
# Path to the local MedGemma model
model_name = "/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma/cache/models--google--medgemma-4b-it/snapshots/698f7911b8e0569ff4ebac5d5552f02a9553063c"

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model and tokenizer
print("Loading MedGemma model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for MedGemma format"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_medgemma_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Create a dataset compatible with MedGemma's format.
    Expected JSON format:
    [
        {
            "image_path": "path/to/image.jpg",
            "text": "medical description of the image"
        }
    ]
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        try:
            image_path = item["image_path"]
            if os.path.exists(image_path):
                # Convert image to base64
                image_base64 = image_to_base64(image_path)
                
                # Create MedGemma format
                medgemma_format = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this medical image and provide a detailed description."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": item["text"]
                        }
                    ]
                }
                
                processed_data.append(medgemma_format)
            else:
                print(f"Warning: Image not found at {image_path}")
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    return processed_data

def tokenize_medgemma_data(examples):
    """Tokenize data in MedGemma format"""
    # Convert messages to text format for tokenization
    texts = []
    for messages in examples["messages"]:
        text_parts = []
        for message in messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if content["type"] == "text":
                        text_parts.append(f"<|user|>\n{content['text']}")
                    elif content["type"] == "image_url":
                        text_parts.append("<|image|>")
            elif message["role"] == "assistant":
                text_parts.append(f"<|assistant|>\n{message['content']}")
        
        # Join with special tokens
        text = "\n".join(text_parts) + "<|endoftext|>"
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_seq_length,
        return_tensors="pt"
    )
    
    # Set labels for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Alternative: Create dataset from your existing format
def create_simple_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Create a simple dataset from image-text pairs.
    This is a fallback if the MedGemma format doesn't work.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        try:
            image_path = item["image_path"]
            if os.path.exists(image_path):
                # Load image and convert to base64
                image = Image.open(image_path).convert('RGB')
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Create simple format
                processed_data.append({
                    "image": image_base64,
                    "text": item["text"],
                    "prompt": "Analyze this medical image and provide a detailed description."
                })
            else:
                print(f"Warning: Image not found at {image_path}")
        except Exception as e:
            print(f"Error processing item: {e}")
            continue
    
    return processed_data

# Load your dataset
print("Loading dataset...")
if os.path.exists(dataset_path):
    # Try MedGemma format first
    try:
        raw_data = create_medgemma_dataset(dataset_path)
        dataset = Dataset.from_list(raw_data)
        
        # Tokenize the dataset
        print("Tokenizing dataset with MedGemma format...")
        tokenized_dataset = dataset.map(
            tokenize_medgemma_data,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"Dataset loaded with {len(tokenized_dataset)} samples")
        
    except Exception as e:
        print(f"MedGemma format failed: {e}")
        print("Trying simple format...")
        
        # Fallback to simple format
        raw_data = create_simple_dataset(dataset_path)
        dataset = Dataset.from_list(raw_data)
        
        # Simple tokenization
        def simple_tokenize(examples):
            texts = []
            for i in range(len(examples["text"])):
                text = f"User: {examples['prompt'][i]}\n[IMAGE]\nAssistant: {examples['text'][i]}"
                texts.append(text)
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        tokenized_dataset = dataset.map(
            simple_tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"Dataset loaded with {len(tokenized_dataset)} samples (simple format)")
        
else:
    print(f"Dataset not found at {dataset_path}")
    print("Please update the dataset_path variable with your actual dataset file")
    print("\nExpected dataset format (JSON):")
    print("""[
    {
        "image_path": "path/to/image1.jpg",
        "text": "Medical description of image 1"
    },
    {
        "image_path": "path/to/image2.jpg", 
        "text": "Medical description of image 2"
    }
]""")
    exit(1)

# Training arguments
training_args = TrainingArguments(
    output_dir="/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma_GigaTrained/training_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduced for multimodal training
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to=None,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,  # Save memory
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start training
print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
trainer.save_model("/mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma_GigaTrained")

print("Training completed!")
print("Model saved to /mnt/nas/BrunoScholles/Gigasistemica/Models/MedGemma_GigaTrained")

# Example usage after training
def generate_medical_description(image_path: str, prompt: str = "Analyze this medical image and provide a detailed description."):
    """Generate medical description for an image using the trained model"""
    # Load and convert image to base64
    image = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Create input format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ]
    
    # Convert to text format for generation
    text = f"<|user|>\n{prompt}\n<|image|>\n<|assistant|>\n"
    
    # Generate response
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response

# Example usage (uncomment to test)
# description = generate_medical_description("path/to/your/test/image.jpg")
# print(description)