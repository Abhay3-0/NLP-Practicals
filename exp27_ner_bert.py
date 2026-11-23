"""Experiment 27: Named Entity Recognition using Transformers (BERT) on CoNLL-2003."""
import numpy as np
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments, pipeline)

MODEL = "bert-base-cased"
ds = load_dataset("eriktks/conll2003")
label_names = ds["train"].features["ner_tags"].feature.names
train = ds["train"].select(range(4000))
valid = ds["validation"].select(range(1000))

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def tokenize_align(batch):
    enc = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    all_labels = []
    for i, tags in enumerate(batch["ner_tags"]):
        prev, labels = None, []
        for wid in enc.word_ids(batch_index=i):
            labels.append(-100 if wid is None or wid == prev else tags[wid])  # label 1st sub-token only
            prev = wid
        all_labels.append(labels)
    enc["labels"] = all_labels
    return enc


train = train.map(tokenize_align, batched=True, remove_columns=train.column_names)
valid = valid.map(tokenize_align, batched=True, remove_columns=valid.column_names)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL, num_labels=len(label_names),
    id2label=dict(enumerate(label_names)), label2id={n: i for i, n in enumerate(label_names)})


def compute_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, -1)
    true_l = [[label_names[l] for l in row if l != -100] for row in labels]
    true_p = [[label_names[pr] for pr, l in zip(prow, lrow) if l != -100]
              for prow, lrow in zip(preds, labels)]
    return {"f1": f1_score(true_l, true_p)}


args = TrainingArguments(output_dir="./bert-ner", num_train_epochs=2,
                         per_device_train_batch_size=16, learning_rate=3e-5,
                         eval_strategy="epoch", save_strategy="no", report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
                  data_collator=DataCollatorForTokenClassification(tokenizer),
                  compute_metrics=compute_metrics)
trainer.train()
print(trainer.evaluate())

ner = pipeline("token-classification", model=model, tokenizer=tokenizer,
               aggregation_strategy="simple", device=model.device.index if model.device.type == "cuda" else -1)
print(ner("Sundar Pichai works at Google in California and visited New Delhi."))
