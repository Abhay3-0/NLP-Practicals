
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

text = "The children were playing happily and running faster than the studies had shown"
words = word_tokenize(text)

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]

print("=" * 60)
print("STEMMING (Porter Stemmer)")
print("=" * 60)
for original, stemmed in zip(words, stemmed_words):
    print(f"{original:15} -> {stemmed}")


lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(nltk_tag):
    """Convert NLTK POS tag to WordNet POS tag for accurate lemmatization."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

pos_tags = nltk.pos_tag(words)
lemmatized_words = [
    lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags
]

print("\n" + "=" * 60)
print("LEMMATIZATION (WordNet Lemmatizer, POS-aware)")
print("=" * 60)
for original, lemma in zip(words, lemmatized_words):
    print(f"{original:15} -> {lemma}")

print("\nOriginal Text :", " ".join(words))
print("Stemmed Text  :", " ".join(stemmed_words))
print("Lemmatized Text:", " ".join(lemmatized_words))
