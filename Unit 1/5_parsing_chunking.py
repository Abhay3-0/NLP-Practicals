
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk import pos_tag, RegexpParser
from nltk.tokenize import word_tokenize

sentence = "The little yellow dog barked loudly at the mysterious stranger near the old gate."

print("=" * 60)
print("CHUNKING USING NLTK RegexpParser (Rule-based Parsing)")
print("=" * 60)

tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)
print("\nPOS Tags:", tagged)

grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = RegexpParser(grammar)
chunk_tree = chunk_parser.parse(tagged)

print("\nChunked Tree (Noun Phrases marked as NP):")
print(chunk_tree)

print("\nExtracted Noun Phrase Chunks:")
for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'NP'):
    phrase = " ".join(word for word, tag in subtree.leaves())
    print(" -", phrase)

chunk_tree.pretty_print()


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(sentence)

print("\n" + "=" * 60)
print("DEPENDENCY PARSING USING spaCy")
print("=" * 60)

print(f"\n{'Token':<15}{'Dependency':<15}{'Head':<15}{'Children'}")
print("-" * 65)
for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:<15}{token.dep_:<15}{token.head.text:<15}{children}")

print("\nNoun Chunks (spaCy):")
for chunk in doc.noun_chunks:
    print(" -", chunk.text)
