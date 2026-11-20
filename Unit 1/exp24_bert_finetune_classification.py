"""Experiment 24: Fine-tuning BERT for text classification using Hugging Face.
Task: IMDB sentiment (subset for speed)."""
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL = "bert-base-uncased"
dataset = load_dataset("imdb")
train = dataset["train"].shuffle(seed=42).select(range(2000))
test = dataset["test"].shuffle(seed=42).select(range(500))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tok = lambda b: tokenizer(b["text"], truncation=True, padding="max_length", max_length=256)
train, test = train.map(tok, batched=True), test.map(tok, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


args = TrainingArguments(
    output_dir="./bert-imdb", num_train_epochs=2, per_device_train_batch_size=16,
    per_device_eval_batch_size=32, learning_rate=2e-5, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="no", logging_steps=50, report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=test,
                  compute_metrics=compute_metrics)
trainer.train()
print(trainer.evaluate())

# Inference
model.eval()
for text in ["An absolutely wonderful film!", "Terrible plot and awful acting."]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    pred = model(**inputs).logits.argmax(-1).item()
    print(f"{text!r} -> {'positive' if pred else 'negative'}")
