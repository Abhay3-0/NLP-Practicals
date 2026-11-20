"""Experiment 32: Hybrid Probabilistic + FSM-based Sequence Modeling for document boundary detection.

Hidden states per page: START (first page of a document) / CONT (continuation page).
 - Probabilistic part: a classifier gives P(state | page features); transition probabilities are
   estimated from training streams; decoding is Viterbi (an HMM-style model).
 - FSM part: hard structural rules mask illegal transitions/states:
       * the first page of the stream must be START
       * a document may not exceed MAX_PAGES (a CONT is illegal once MAX_PAGES is reached)
       * a document must have at least 1 page (START->START allowed = single-page doc)
"""
import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

random.seed(1)
START, CONT = 0, 1
MAX_PAGES = 5
CATS = ["sci.space", "rec.sport.hockey", "comp.graphics", "talk.politics.mideast", "sci.med"]
strip = ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", categories=CATS, remove=strip)
test = fetch_20newsgroups(subset="test", categories=CATS, remove=strip)


def paginate(text, n=100):
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n) if len(w[i:i + n]) > 30]


def build_stream(texts, n_docs):
    """Concatenate pages of random docs (1..MAX_PAGES pages each) into one stream."""
    pool = [t for t in texts if len(t.split()) > 150]
    pages, states = [], []
    for t in random.sample(pool, n_docs):
        p = paginate(t)[:random.randint(1, MAX_PAGES)]
        pages += p
        states += [START] + [CONT] * (len(p) - 1)
    return pages, np.array(states)


tr_pages, tr_states = build_stream(train.data, 300)
te_pages, te_states = build_stream(test.data, 60)

tfidf = TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True).fit(tr_pages)


def features(pages):
    X = tfidf.transform(pages)
    f = [[1.0, 0.0, 0.0]]                                     # first page: no previous
    for i in range(1, len(pages)):
        s_prev = cosine_similarity(X[i], X[i - 1])[0, 0]
        s_next = cosine_similarity(X[i], X[i + 1])[0, 0] if i + 1 < len(pages) else 0.0
        f.append([0.0, s_prev, s_next])
    return np.array(f)


# ---- Probabilistic components ----
clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(features(tr_pages), tr_states)
prior = np.bincount(tr_states, minlength=2) / len(tr_states)

trans = np.ones((2, 2))                                       # Laplace-smoothed transition counts
for a, b in zip(tr_states[:-1], tr_states[1:]):
    trans[a, b] += 1
trans /= trans.sum(1, keepdims=True)

# ---- FSM: allowed transitions ----
ALLOWED = np.array([[1, 1],      # START -> START (1-page doc) | START -> CONT
                    [1, 1]])     # CONT  -> START              | CONT  -> CONT


def viterbi_fsm(pages):
    F = features(pages)
    post = clf.predict_proba(F)                                # P(state | features)
    emit = np.log(post / prior + 1e-12)                        # scaled likelihood
    T = len(pages)
    delta = np.full((T, 2, MAX_PAGES + 1), -np.inf)            # state x current doc length
    back = {}
    delta[0, START, 1] = emit[0, START]                        # FSM rule: stream starts with START
    for t in range(1, T):
        for s in (START, CONT):
            for prev in (START, CONT):
                if not ALLOWED[prev, s]:
                    continue
                for L in range(1, MAX_PAGES + 1):
                    if delta[t - 1, prev, L] == -np.inf:
                        continue
                    newL = 1 if s == START else L + 1
                    if newL > MAX_PAGES:                       # FSM rule: max document length
                        continue
                    score = delta[t - 1, prev, L] + np.log(trans[prev, s]) + emit[t, s]
                    if score > delta[t, s, newL]:
                        delta[t, s, newL] = score
                        back[(t, s, newL)] = (prev, L)
    s, L = np.unravel_index(np.argmax(delta[T - 1]), delta[T - 1].shape)
    path = [s]
    for t in range(T - 1, 0, -1):
        s, L = back[(t, s, L)]
        path.append(s)
    return np.array(path[::-1])


def report(name, pred, truth):
    tp = int(((pred == START) & (truth == START)).sum())
    p, r = tp / max((pred == START).sum(), 1), tp / max((truth == START).sum(), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    print(f"{name:28s} precision={p:.2f} recall={r:.2f} F1={f1:.2f} | accuracy={(pred == truth).mean():.2f}")


baseline = clf.predict(features(te_pages))                     # per-page, no sequence model
hybrid = viterbi_fsm(te_pages)
print(f"Test stream: {len(te_pages)} pages, {int((te_states == START).sum())} documents")
report("Classifier only (baseline)", baseline, te_states)
report("Hybrid HMM + FSM (Viterbi)", hybrid, te_states)
print("True boundaries     :", np.where(te_states == START)[0].tolist())
print("Predicted boundaries:", np.where(hybrid == START)[0].tolist())
