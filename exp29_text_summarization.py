"""Experiment 29: Text Summarization using BART or T5."""
from transformers import pipeline

ARTICLE = """
Artificial intelligence has progressed rapidly over the past decade, driven by larger datasets,
more powerful hardware, and new neural architectures. The transformer, introduced in 2017,
replaced recurrence with self-attention and now underpins models for language, vision, and
speech. Companies across healthcare, finance, and education are deploying these models to
automate documentation, assist diagnosis, detect fraud, and personalise learning. At the same
time, researchers and regulators are debating issues such as bias, transparency, energy use,
and the impact of automation on jobs. Many experts argue that careful evaluation, human
oversight, and clear governance will be essential as the technology becomes more widespread.
"""

# Abstractive summarisation with BART (fine-tuned on CNN/DailyMail)
bart = pipeline("summarization", model="facebook/bart-large-cnn")
print("BART:", bart(ARTICLE, max_length=60, min_length=20, do_sample=False)[0]["summary_text"])

# T5 (the 'summarize:' prefix is added automatically by the t5-small config)
t5 = pipeline("summarization", model="t5-small")
print("T5  :", t5(ARTICLE, max_length=60, min_length=20, do_sample=False)[0]["summary_text"])

# Optional: evaluate with ROUGE (pip install rouge_score evaluate)
# import evaluate
# print(evaluate.load("rouge").compute(predictions=[...], references=[...]))
