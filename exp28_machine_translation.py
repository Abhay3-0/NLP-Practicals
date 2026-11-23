"""Experiment 28: Machine Translation using MarianMT (and T5) from Hugging Face."""
from transformers import MarianMTModel, MarianTokenizer, pipeline

sentences = [
    "Machine learning is transforming the world.",
    "Where is the nearest railway station?",
    "I would like to learn natural language processing.",
]

# ---- MarianMT: change the target language by swapping the model name ----
# en->de: Helsinki-NLP/opus-mt-en-de | en->fr: opus-mt-en-fr | en->hi: opus-mt-en-hi
MODEL = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(MODEL)
model = MarianMTModel.from_pretrained(MODEL)

batch = tokenizer(sentences, return_tensors="pt", padding=True)
generated = model.generate(**batch, num_beams=4, max_length=64)
print(f"--- MarianMT ({MODEL}) ---")
for src, tgt in zip(sentences, tokenizer.batch_decode(generated, skip_special_tokens=True)):
    print(f"{src}\n  -> {tgt}")

# ---- T5 (t5-small supports English -> German/French/Romanian) ----
t5 = pipeline("translation_en_to_fr", model="t5-small")
print("\n--- T5 (en -> fr) ---")
for s in sentences:
    print(f"{s}\n  -> {t5(s, max_length=64)[0]['translation_text']}")
