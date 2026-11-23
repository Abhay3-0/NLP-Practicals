
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

text = "Natural Language Processing is fascinating. It helps computers understand human language! Isn't that amazing?"

print("=" * 60)
print("NLTK TOKENIZATION")
print("=" * 60)

nltk_sentences = sent_tokenize(text)
print("\nSentence Tokens (NLTK):")
for i, sent in enumerate(nltk_sentences, 1):
    print(f"{i}. {sent}")

nltk_words = word_tokenize(text)
print("\nWord Tokens (NLTK):")
print(nltk_words)
print(f"\nTotal words: {len(nltk_words)}")


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print("\n" + "=" * 60)
print("spaCy TOKENIZATION")
print("=" * 60)

spacy_sentences = list(doc.sents)
print("\nSentence Tokens (spaCy):")
for i, sent in enumerate(spacy_sentences, 1):
    print(f"{i}. {sent.text}")

spacy_words = [token.text for token in doc]
print("\nWord Tokens (spaCy):")
print(spacy_words)
print(f"\nTotal words: {len(spacy_words)}")
