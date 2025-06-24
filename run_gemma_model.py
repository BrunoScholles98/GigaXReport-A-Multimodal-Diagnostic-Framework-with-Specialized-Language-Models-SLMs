import os
# Disable TorchDynamo before importing PyTorch
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import torch._dynamo
torch._dynamo.disable()

from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

model_id = "google/medgemma-4b-it"
cache_dir = "/mnt/nas/BrunoScholles/Gigasistemica/MedGemma/cache"

# Force use of CUDA:0 if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the multimodal model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.bfloat16,
    device_map=None
)
model.to(device)

# Load the processor
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

# Open the local image and extract the result code from its filename
image_path = "/mnt/nas/BrunoScholles/Gigasistemica/MedGemma/Test_Imgs/Pacient1_Result0.png"
image = Image.open(image_path).convert("RGB")

# Extract result code and map
result_code = os.path.splitext(os.path.basename(image_path))[0].split("_")[-1]
result_text = {
    "Result0": "the patient is healthy",
    "Result1": "the patient has osteoporosis"
}.get(result_code, "diagnosis unknown")

#TODO: Use the generic_diagnosis_prompt.txt to generate the prompt

# Build the multimodal chat messages with detailed‐description prompt:
# Load the diagnosis template
with open("prompt/generic_diagnosis_template.txt", "r") as f:
    diagnosis_template = f.read()

messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
    {"role": "user",   "content": [
        {"type": "text", "text":
            f"The EfficientNet classification for this X-ray is: {result_text}. "
            "Based on this classification, please provide a detailed description of the bone structure and bone health observed in the image. "
            f"Please follow this diagnosis template:\n\n{diagnosis_template}"
        },
        {"type": "image", "image": image}
    ]}
]

# Prepare inputs and move them to CUDA:0
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

# Run generation in inference mode
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
    )
    # Slice off the prompt tokens
    generated_ids = generated_ids[0][input_len:]

# Decode and print the model’s response
decoded_output = processor.decode(generated_ids, skip_special_tokens=True)
print(decoded_output)
