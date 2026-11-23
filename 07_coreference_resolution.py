"""Practical 7: Coreference Resolution using spaCy and FastCoref.

FastCoref is a neural coreference resolver that plugs into a spaCy v3 pipeline.
It identifies mentions that refer to the same entity and can rewrite pronouns.
Install requirements and the spaCy English model before running this script.
"""
import spacy
from fastcoref import spacy_component  # registers the FastCoref spaCy component

TEXT = (
    "John met Mary after class. He thanked her for the notes "
    "she had shared with him."
)

# FastCoref uses spaCy's tokenizer; the other pipeline components are not needed.
nlp = spacy.load(
    "en_core_web_sm",
    exclude=["parser", "lemmatizer", "ner", "textcat"],
)
nlp.add_pipe("fastcoref")

# Resolve pronouns to their most likely antecedents.
doc = nlp(TEXT, component_cfg={"fastcoref": {"resolve_text": True}})

print("Original text:")
print(TEXT)
print("\nResolved text:")
print(doc._.resolved_text)
print("\nCoreference clusters (character spans):")
for number, cluster in enumerate(doc._.coref_clusters or [], start=1):
    mentions = [TEXT[start:end] for start, end in cluster]
    print(f"{number}. {mentions}")
