# Burushaski Translator 🌍

> **The world's first AI-powered English ↔ Burushaski neural machine translation model.**


## About Burushaski

**Burushaski** is a language isolate spoken by approximately **100,000 people** in the Hunza, Nagar, and Yasin valleys of **Gilgit-Baltistan, Pakistan**. It has no known relatives making it one of the most unique and fascinating languages on Earth.

This project is the **first ever** neural machine translation system for Burushaski — a major step toward preserving and digitizing this remarkable endangered language.

---

## Sample Translations

| English | Burushaski |
|---|---|
| Come here | zu naa |
| Where are you | amulo ba |
| I have three sons | ja iski yuwa baan |
| Was he a child | giyaas bam |
| I do not know | jaa leel api |
| The mountain is high | chare t'aan nimi |
| This is my mother | khote jaa mama bo |
| I am hungry | je channe baa |
| The water is cold | cʰil cʰayurum bila |

---

## Model Details

| Detail | Value |
|---|---|
| Base model | Helsinki-NLP/opus-mt-en-mul |
| Task | English → Burushaski Translation |
| Training data | ~8,000 verified sentence pairs |
| Data source | Professional linguistic texts by Sadaf Munshi, University of North Texas |
| Architecture | MarianMT (Seq2Seq Transformer) |
| Training platform | Google Colab |

---

## Installation

```bash
pip install transformers torch sentencepiece sacremoses
```

---

## Usage

```python
from transformers import MarianMTModel, MarianTokenizer
import torch

# Load model
model_path = "./burushaski_model"
tokenizer  = MarianTokenizer.from_pretrained(model_path)
model      = MarianMTModel.from_pretrained(model_path)

# Translate
def translate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(translate("How are you?"))
print(translate("The water is cold."))
print(translate("I am hungry."))
```

---

## Model Files

```
burushaski_model/
├── config.json               # Model configuration
├── generation_config.json    # Generation settings
├── model.safetensors         # Trained model weights
├── source.spm                # Source language tokenizer
├── target.spm                # Target language tokenizer
├── tokenizer_config.json     # Tokenizer configuration
└── vocab.json                # Vocabulary file
```

---

## Training Data

The model was trained on **8,000+ verified Burushaski-English sentence pairs** sourced from:

- Professional linguistic documentation by **Sadaf Munshi** (University of North Texas)

---

## Limitations

- This is **Version 1.0** — first ever Burushaski translator
- Works best on simple sentences
- May struggle with modern vocabulary not in training data
- Grammar occasionally differs from native speaker usage
- Improving continuously as more data is collected

---

## Roadmap

- [ ] Add more training data (50,000+ pairs)
- [ ] Build web application
- [ ] Add Burushaski → English direction
- [ ] Add voice input (English speech)
- [ ] Collect native speaker audio recordings
- [ ] Build mobile app
- [ ] Add Burushaski text to speech

---

## Contributing

Contributions are welcome! Especially:

- **Native Burushaski speakers** who can verify and correct translations
- **Linguists** with access to Burushaski parallel texts
- **Developers** who want to help build the web app

---

## Citation

If you use this model in your research please cite:

```
@misc{burushaski-translator-2026,
  title={Burushaski Translator: First Neural Machine Translation Model for Burushaski},
  author={Alishan Amin},
  year={2026},
  url={https://github.com/alishanaminnnn/Translator}
}
```

---


The training data is sourced from linguistic documentation by Sadaf Munshi, University of North Texas. Please credit appropriately if using this model.

---

## Contact

- GitHub: [@alishanaminnnn](https://github.com/alishanaminnnn)

---

*Built with ❤️ for the Burushaski speaking community of Gilgit-Baltistan, Pakistan*