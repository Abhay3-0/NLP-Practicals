
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

document = """Natural Language Processing is a subfield of Artificial Intelligence
that helps computers understand, interpret, and manipulate human language.
It is used in a wide range of applications such as chatbots, translators, and search engines."""

words = word_tokenize(document)

stop_words = set(stopwords.words('english'))
print(f"Total stop words in NLTK English list: {len(stop_words)}")

filtered_words = [
    word for word in words
    if word.lower() not in stop_words and word.isalpha()
]

removed_words = [
    word for word in words
    if word.lower() in stop_words
]

print("\nOriginal Document:")
print(document)

print("\nOriginal Word Count :", len([w for w in words if w.isalpha()]))
print("Words Removed (stop words):", removed_words)
print("\nFiltered Word Count  :", len(filtered_words))
print("Filtered Words       :", filtered_words)

print("\nText after Stop-word Removal:")
print(" ".join(filtered_words))
