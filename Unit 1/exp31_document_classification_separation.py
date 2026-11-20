"""Experiment 31: Automatic Document Classification and Separation using ML.
1) Train a TF-IDF + Logistic Regression classifier on 20 Newsgroups.
2) Take a stream of concatenated pages (from different documents) and separate it by
   detecting where the predicted class or content similarity changes."""
import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

random.seed(0)
CATS = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast"]
strip = ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", categories=CATS, remove=strip)
test = fetch_20newsgroups(subset="test", categories=CATS, remove=strip)

# ---- Part 1: classification ----
clf = make_pipeline(TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True),
                    LogisticRegression(max_iter=1000))
clf.fit(train.data, train.target)
print(classification_report(test.target, clf.predict(test.data), target_names=test.target_names))

# ---- Part 2: separation of a page stream ----
def make_pages(text, words_per_page=120):
    w = text.split()
    return [" ".join(w[i:i + words_per_page]) for i in range(0, len(w), words_per_page)
            if len(w[i:i + words_per_page]) > 30]

docs = [d for d in test.data if len(d.split()) > 300]
chosen = random.sample(range(len(docs)), 8)
pages, truth = [], []
for d in chosen:
    p = make_pages(docs[d])[:4]
    truth += [1] + [0] * (len(p) - 1)               # 1 = first page of a new document
    pages += p

proba = clf.predict_proba(pages)
labels = proba.argmax(1)
tfidf = clf.steps[0][1]
X = tfidf.transform(pages)
pred = [1]
for i in range(1, len(pages)):
    sim = cosine_similarity(X[i], X[i - 1])[0, 0]
    changed = labels[i] != labels[i - 1]
    pred.append(int(changed or sim < 0.05))           # boundary if class changes or pages unrelated

truth, pred = np.array(truth), np.array(pred)
tp = int(((truth == 1) & (pred == 1)).sum())
prec, rec = tp / max(pred.sum(), 1), tp / max(truth.sum(), 1)
print(f"\nBoundary detection -> precision={prec:.2f} recall={rec:.2f}")
print("True boundaries     :", np.where(truth == 1)[0].tolist())
print("Predicted boundaries:", np.where(pred == 1)[0].tolist())
for i, (b, l) in enumerate(zip(pred, labels)):
    if b:
        print(f"-- new document at page {i}: predicted class = {test.target_names[l]}")
