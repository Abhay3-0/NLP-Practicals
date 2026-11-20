"""Experiment 30: Chatbot development using a pre-trained transformer (DialoGPT).
Type 'quit' to exit."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).eval()

history_ids = None
print("Chatbot ready. Type 'quit' to exit.")
while True:
    user = input("You: ").strip()
    if user.lower() in {"quit", "exit"}:
        break
    new_ids = tokenizer.encode(user + tokenizer.eos_token, return_tensors="pt")
    input_ids = new_ids if history_ids is None else torch.cat([history_ids, new_ids], dim=-1)
    input_ids = input_ids[:, -512:]                       # keep recent context only
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        history_ids = model.generate(
            input_ids, attention_mask=attention_mask, max_length=input_ids.shape[-1] + 60,
            pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.92,
            temperature=0.7, no_repeat_ngram_size=3)
    reply = tokenizer.decode(history_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
    print("Bot:", reply)
