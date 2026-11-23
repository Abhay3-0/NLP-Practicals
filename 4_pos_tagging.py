
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.tokenize import word_tokenize

sentence = "The quick brown fox jumps over the lazy dog near the riverbank."

print("=" * 60)
print("POS TAGGING USING NLTK")
print("=" * 60)

tokens = word_tokenize(sentence)
nltk_pos_tags = nltk.pos_tag(tokens)

print(f"\nSentence: {sentence}\n")
print(f"{'Word':<15}{'POS Tag':<10}")
print("-" * 25)
for word, tag in nltk_pos_tags:
    print(f"{word:<15}{tag:<10}")


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(sentence)

print("\n" + "=" * 60)
print("POS TAGGING USING spaCy")
print("=" * 60)
print(f"\n{'Word':<15}{'POS':<10}{'Tag':<10}{'Explanation'}")
print("-" * 65)
for token in doc:
    print(f"{token.text:<15}{token.pos_:<10}{token.tag_:<10}{spacy.explain(token.tag_)}")
