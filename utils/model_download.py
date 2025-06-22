from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/medgemma-4b-it"
cache_dir = "/mnt/nas/BrunoScholles/Gigasistemica/MedGemma/cache"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)