"""Experiment 25: Sentence Embeddings using BERT and GPT-2.
Mean-pooled hidden states (masked) + cosine similarity."""
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

sentences = [
    "The cat sits on the mat.",
    "A kitten is resting on the rug.",
    "Stock markets fell sharply today.",
    "Investors sold shares as prices dropped.",
]


def embed(model_name, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:                  # GPT-2 has no pad token
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name).eval()
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state      # (batch, seq, dim)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return ((hidden * mask).sum(1) / mask.sum(1)).numpy()   # mean pooling


for name in ["bert-base-uncased", "gpt2"]:
    emb = embed(name, sentences)
    print(f"\n=== {name} | embedding shape: {emb.shape} ===")
    sim = cosine_similarity(emb)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"sim({i},{j}) = {sim[i, j]:.3f}  | {sentences[i]!r} vs {sentences[j]!r}")
