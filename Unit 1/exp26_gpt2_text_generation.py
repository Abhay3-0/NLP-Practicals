"""Experiment 26: Text Generation using GPT-2 on a custom dataset.
Usage: python exp26_gpt2_text_generation.py [path/to/custom.txt]
If no file is given, a tiny demo corpus is used (replace it with your own text)."""
import sys
import torch
from datasets import Dataset
from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          GPT2Tokenizer, Trainer, TrainingArguments)

BLOCK = 128
if len(sys.argv) > 1:
    text = open(sys.argv[1], encoding="utf-8").read()
else:
    text = ("Machine learning is the study of algorithms that improve through experience. "
            "Neural networks learn representations from data. "
            "Transformers use attention to model long-range dependencies. ") * 100

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
ids = tokenizer(text)["input_ids"]
blocks = [ids[i:i + BLOCK] for i in range(0, len(ids) - BLOCK, BLOCK)]
dataset = Dataset.from_dict({"input_ids": blocks, "attention_mask": [[1] * BLOCK] * len(blocks)})

model = GPT2LMHeadModel.from_pretrained("gpt2")
args = TrainingArguments(output_dir="./gpt2-custom", num_train_epochs=3,
                         per_device_train_batch_size=4, learning_rate=5e-5,
                         save_strategy="no", logging_steps=10, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=dataset,
                  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
trainer.train()

model.eval()
prompt = "Neural networks"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=60, do_sample=True, top_k=50, top_p=0.95,
                     temperature=0.8, pad_token_id=tokenizer.eos_token_id)
print("\nGenerated:\n", tokenizer.decode(out[0], skip_special_tokens=True))
model.save_pretrained("./gpt2-custom"); tokenizer.save_pretrained("./gpt2-custom")
