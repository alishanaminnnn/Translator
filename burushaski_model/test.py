import os

import torch
from transformers import MarianMTModel, MarianTokenizer

# Load your model
print("Loading model...")
model_path = os.path.dirname(os.path.abspath(__file__))
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✅ Model loaded! Using: {device}")

# Translation function
def translate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test translations
print("\nTest translations:")
print("-" * 50)
test_sentences = [
    "I do not know",
    "my name is Ahmed ALi",
]

for sentence in test_sentences:
    translation = translate(sentence)
    print(f"EN:  {sentence}")
    print(f"BUR: {translation}")
    print()
