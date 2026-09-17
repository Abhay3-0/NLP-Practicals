# 🧠 Natural Language Processing — Unit 1 Practicals

Welcome to the **NLP Unit 1 Practicals** repository.

This repository contains Python implementations of fundamental **Natural Language Processing techniques** using **NLTK, spaCy, and Regular Expressions**. These programs demonstrate how raw text can be processed and analyzed using different NLP techniques.

---

## 📌 Practical List

| No. | Experiment | Technology |
|:---:|---|---|
| 01 | Sentence and Word Tokenization | NLTK, spaCy |
| 02 | Stemming and Lemmatization | NLTK |
| 03 | Stop-word Removal | NLTK |
| 04 | Part-of-Speech Tagging | NLTK |
| 05 | Parsing and Chunking | RegEx, NLTK, spaCy |
| 06 | Named Entity Recognition | spaCy |

---

## 📂 Repository Contents

```text
NLP-Unit-1-Practicals/
│
├── 01_tokenization.py
├── 02_stemming_lemmatization.py
├── 03_stopword_removal.py
├── 04_pos_tagging.py
├── 05_parsing_chunking.py
├── 06_ner.py
│
└── README.md
```

---

## 🔧 Requirements

The programs require:

- Python 3.x
- NLTK
- spaCy
- English spaCy language model

### Install Dependencies

```bash
pip install nltk spacy
```

Install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

# 🧪 Experiments

## 1️⃣ Tokenization

Tokenization is the process of breaking a text into smaller units called **tokens**.

The program demonstrates:

- Sentence tokenization
- Word tokenization
- NLTK tokenization
- spaCy tokenization

Example:

```text
Input:
Natural Language Processing is interesting.

Output:
Natural
Language
Processing
is
interesting
```

---

## 2️⃣ Stemming and Lemmatization

This experiment demonstrates two important text normalization techniques.

### Stemming

Stemming attempts to reduce a word to its root form.

```text
playing → play
```

### Lemmatization

Lemmatization converts words into their meaningful dictionary form.

```text
studies → study
```

The experiment uses the **Porter Stemmer** and **WordNet Lemmatizer** provided by NLTK.

---

## 3️⃣ Stop-word Removal

Stop-word removal eliminates commonly occurring words that may not provide significant information for certain NLP applications.

Examples include:

```text
the
is
a
an
of
and
```

The program:

1. Tokenizes the input text.
2. Loads English stop words.
3. Compares each token with the stop-word list.
4. Removes matching words.

---

## 4️⃣ Part-of-Speech Tagging

POS tagging assigns a grammatical category to every word in a sentence.

Example:

```text
The → DT
quick → JJ
fox → NN
jumps → VBZ
```

Some commonly used POS tags are:

| Tag | Description |
|---|---|
| NN | Noun |
| VB | Verb |
| JJ | Adjective |
| RB | Adverb |
| DT | Determiner |
| IN | Preposition |

---

## 5️⃣ Parsing and Chunking

This experiment explores the grammatical structure of sentences.

### RegEx Chunking

Regular expressions are used to identify patterns of POS tags and group words into phrases.

Example:

```text
The young boy
```

can be identified as a **Noun Phrase (NP)**.

### spaCy Parsing

spaCy is used to examine:

- Part of Speech
- Dependency relationship
- Head word
- Noun phrases

---

## 6️⃣ Named Entity Recognition

Named Entity Recognition, or **NER**, identifies real-world entities present in text.

For example:

```text
Google → ORG
Abhay → PERSON
Greater Noida → GPE
```

Common entity categories include:

```text
PERSON
ORG
GPE
LOC
DATE
MONEY
PRODUCT
```

The experiment uses the **spaCy English language model**.

---

# ▶️ Running the Programs

After installing the dependencies, execute any practical from the terminal.

For example:

```bash
python 01_tokenization.py
```

Run the remaining programs using:

```bash
python 02_stemming_lemmatization.py
python 03_stopword_removal.py
python 04_pos_tagging.py
python 05_parsing_chunking.py
python 06_ner.py
```

---

# 🎯 Learning Objectives

The practicals provide hands-on understanding of basic NLP operations, including:

- Text tokenization
- Text normalization
- Stop-word filtering
- Grammatical analysis
- Phrase identification
- Dependency parsing
- Entity extraction

These techniques form the foundation for more advanced NLP applications such as **text classification, sentiment analysis, information extraction, chatbots, and question-answering systems**.

---

# 📚 Tools & Technologies

**Programming Language**

```text
Python
```

**Libraries**

```text
NLTK
spaCy
```

**Other Technology**

```text
Regular Expressions
```

---

## 👨‍💻 Author

**Abhay Pratap Singh**

B.Tech — Computer Science / Artificial Intelligence

---

## 📄 Academic Information

**Course:** Natural Language Processing  
**Unit:** Unit 1  
**Course Outcome:** CO1  

> This repository is maintained for academic learning and practical implementation of fundamental NLP concepts.
