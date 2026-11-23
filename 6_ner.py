

import spacy

nlp = spacy.load("en_core_web_sm")

text = ("Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976. "
        "The company is now led by Tim Cook and generated over $380 billion "
        "in revenue last year. In September 2024, Apple announced a "
        "partnership with Microsoft.")

doc = nlp(text)

print("=" * 60)
print("NAMED ENTITY RECOGNITION USING spaCy")
print("=" * 60)

print(f"\nText:\n{text}\n")

print(f"{'Entity':<25}{'Label':<10}{'Explanation'}")
print("-" * 70)
for ent in doc.ents:
    print(f"{ent.text:<25}{ent.label_:<10}{spacy.explain(ent.label_)}")

print(f"\nTotal Named Entities Found: {len(doc.ents)}")

from collections import defaultdict
entities_by_type = defaultdict(list)
for ent in doc.ents:
    entities_by_type[ent.label_].append(ent.text)

print("\nEntities Grouped by Type:")
for label, entities in entities_by_type.items():
    print(f"  {label} ({spacy.explain(label)}): {entities}")

# Render entities with inline highlighting (saved as HTML file)
from spacy import displacy
html = displacy.render(doc, style="ent", page=True)
with open("ner_visualization.html", "w") as f:
    f.write(html)
print("\nVisualization saved to 'ner_visualization.html'")
