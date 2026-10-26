🧠 Natural Language Processing — Unit 1 Practicals

Welcome to the Natural Language Processing (NLP) Unit 1 Practicals repository.

This repository contains Python implementations of fundamental and intermediate Natural Language Processing techniques. These practicals demonstrate how raw text can be processed, transformed, analyzed, and used in various NLP applications using Python and libraries such as NLTK, spaCy, Gensim, and Scikit-learn.

📌 Practical List
No.	Experiment	Description
01	Tokenization	Splitting text into sentences and words
08	Bag of Words (BOW)	Representing text using word frequencies
09	TF-IDF	Measuring the importance of words in documents
10	N-Gram Model	Analyzing sequences of words
11	Cosine Similarity	Measuring similarity between text documents
12	Word2Vec	Generating word embeddings using contextual relationships
13	GloVe	Representing words using global co-occurrence statistics
14	Word Mover's Distance (WMD)	Measuring semantic distance between documents
15	Text Classification	Categorizing text into predefined classes
16	Sentiment Analysis	Identifying positive, negative, or neutral sentiment
17	Latent Dirichlet Allocation (LDA)	Discovering topics in a collection of documents
18	Latent Semantic Analysis (LSA)	Discovering latent semantic relationships in text
19	Opinion Mining	Extracting opinions and attitudes from text
20	Information Extraction	Identifying structured information in unstructured text
21	Information Retrieval	Finding relevant documents for a query
Note: The screenshot also contains 02_stemming_lemmatization.py, 03_stopword_removal.py, 04_pos_tagging.py, 05_parsing_chunking.py, and 06_ner.py only if those files are present in your repository. The practical numbers above follow the filenames visible in the screenshot.
📂 Repository Structure
text
NLP-Practicals/
│
└── Unit 1/
    ├── 1_tokenization.py
    ├── 08_BOW.py
    ├── 09_TFIDF.py
    ├── 10_NGram.py
    ├── 11_Cosine_Similarity.py
    ├── 12_Word2Vec.py
    ├── 13_GloVe.py
    ├── 14_WMD.py
    ├── 15_Text_Classification.py
    ├── 16_Sentiment_Analysis.py
    ├── 17_LDA.py
    ├── 18_LSA.py
    ├── 19_Opinion_Mining.py
    ├── 20_Information_Extraction.py
    └── 21_Information_Retrieval.py

🔧 Requirements
Python 3.9 or later
NLTK
spaCy
NumPy
Pandas
Scikit-learn
Gensim
Matplotlib (optional, for visualizations)

The exact dependencies depend on the practical being executed.

⚙️ Installation
1. Clone the repository
bash
git clone <YOUR_REPOSITORY_URL>
cd NLP-Practicals


Replace <YOUR_REPOSITORY_URL> with your GitHub repository URL.

2. Create a virtual environment (recommended)
bash
python -m venv venv


Activate it on Windows:

bash
venv\Scripts\activate


Activate it on Linux or macOS:

bash
source venv/bin/activate

3. Install dependencies
bash
pip install nltk spacy numpy pandas scikit-learn gensim matplotlib

4. Download required NLP resources

For NLTK:

python
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


For spaCy:

bash
python -m spacy download en_core_web_sm


Some practicals may require additional corpora, pretrained embeddings, or datasets.

🧪 Experiments
1. Tokenization

Tokenization is the process of splitting text into smaller units called tokens.

Objectives:

Perform word tokenization.
Perform sentence tokenization.
Understand basic text preprocessing.

Example:

Input:

text
Natural Language Processing is interesting.


Output:

text
['Natural', 'Language', 'Processing', 'is', 'interesting', '.']


Technologies: Python, NLTK, spaCy

2. Bag of Words (BOW)

The Bag of Words model represents a document as a vector of word counts, without considering word order.

Objectives:

Create a vocabulary from text.
Convert documents into numerical vectors.
Understand text vectorization.

Example:

Documents:

text
Document 1: I like NLP
Document 2: I like Python


Vocabulary:

text
['I', 'like', 'NLP', 'Python']


Each document is represented by the frequencies of vocabulary terms.

Technology: Scikit-learn

File: 08_BOW.py

3. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF measures the importance of a word in a document relative to a collection of documents.

Objectives:

Calculate term importance.
Convert text into weighted numerical vectors.
Reduce the relative importance of common terms.

Concept:
Where:

TF represents the frequency of a term in a document.
IDF measures how rare or informative the term is across documents.

Technology: Scikit-learn

File: 09_TFIDF.py

4. N-Gram Model

An N-gram is a contiguous sequence of n tokens extracted from text.

Types:

Unigram: one word
Bigram: two consecutive words
Trigram: three consecutive words

Example:

Sentence:

text
Natural Language Processing is useful


Bigrams:

text
Natural Language
Language Processing
Processing is
is useful


Applications:

Language modeling
Text prediction
Text classification
Phrase analysis

File: 10_NGram.py

5. Cosine Similarity

Cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them.

Cosine Similarity
Objectives:

Convert documents into vectors.
Calculate similarity scores.
Compare documents based on their textual content.

A score closer to 1 generally indicates greater similarity for nonnegative text vectors.

Applications:

Document comparison
Search engines
Recommendation systems

File: 11_Cosine_Similarity.py

6. Word2Vec

Word2Vec learns dense word embeddings from surrounding-word patterns in a text corpus.

Architectures:

CBOW (Continuous Bag of Words)
Skip-gram

Objectives:

Generate vector representations of words.
Explore relationships between words.
Understand distributional semantics.

Applications:

Semantic similarity
Text classification
NLP feature extraction

Technology: Gensim

File: 12_Word2Vec.py

7. GloVe (Global Vectors for Word Representation)

GloVe is a word-embedding technique that learns vector representations using global word co-occurrence statistics.

Objectives:

Understand pretrained word embeddings.
Represent words as dense numerical vectors.
Explore semantic relationships between words.

Applications:

Semantic analysis
Text classification
Information retrieval

Note: Unlike Word2Vec, GloVe is commonly used through pretrained vectors or a separate training implementation. The practical may require downloading a pretrained embedding file.

File: 13_GloVe.py

8. Word Mover's Distance (WMD)

Word Mover's Distance measures the semantic distance between two documents by calculating the minimum transport cost required to move the word embeddings of one document to match those of another.

Objectives:

Compare documents using word embeddings.
Understand semantic distance.
Examine similarities that simple word-overlap methods may miss.

Applications:

Document clustering
Semantic search
Document similarity analysis

Note: WMD requires suitable word embeddings and may need additional Gensim configuration.

File: 14_WMD.py

9. Text Classification

Text classification assigns predefined categories or labels to text.

Objectives:

Prepare and preprocess text data.
Extract numerical text features.
Train a classification model.
Predict labels for new text.

Common algorithms:

Naive Bayes
Logistic Regression
Support Vector Machines (SVM)

Applications:

Spam detection
News categorization
Topic classification
Document labeling

File: 15_Text_Classification.py

10. Sentiment Analysis

Sentiment analysis determines the emotional polarity or opinion expressed in a text.

Common categories:

Positive
Negative
Neutral

Example:

Input:

text
The movie was excellent and enjoyable.


Expected interpretation:

text
Positive sentiment


Applications:

Product reviews
Customer feedback analysis
Social media monitoring
Brand reputation analysis

File: 16_Sentiment_Analysis.py

11. Latent Dirichlet Allocation (LDA)

LDA is a probabilistic topic-modeling algorithm that discovers hidden topics in a collection of documents.

Objectives:

Identify recurring themes.
Represent documents as topic distributions.
Represent topics as word distributions.

Applications:

Document clustering
Research paper analysis
News topic discovery
Large-scale text exploration

Technology: Gensim or Scikit-learn

File: 17_LDA.py

12. Latent Semantic Analysis (LSA)

LSA identifies latent semantic patterns in a document-term matrix, typically using Singular Value Decomposition (SVD).

Objectives:

Reduce the dimensionality of text representations.
Discover underlying semantic relationships.
Compare documents using reduced-dimensional vectors.

Applications:

Document similarity
Search engines
Information retrieval
Semantic indexing

Technology: Scikit-learn

File: 18_LSA.py

13. Opinion Mining

Opinion mining extracts opinions, attitudes, and sentiment from textual data. It is closely related to sentiment analysis and may include aspect-based analysis.

Objectives:

Identify subjective statements.
Determine opinion polarity.
Analyze opinions about specific products, services, or features.

Example:

Input:

text
The camera is excellent, but the battery life is poor.


Possible interpretation:

text
Camera: Positive
Battery life: Negative


Applications:

Product review analysis
Customer satisfaction measurement
Feedback summarization
Market research

File: 19_Opinion_Mining.py

14. Information Extraction

Information extraction identifies useful structured information from unstructured text.

Techniques may include:

Named Entity Recognition (NER)
Entity identification
Relationship extraction
Pattern matching with regular expressions

Example:

Input:

text
Google opened a new office in London.


Possible extracted entities:

text
Google  -> ORGANIZATION
London  -> LOCATION


Applications:

Knowledge graph construction
Automated document processing
Entity extraction
Business intelligence

Technology: spaCy, NLTK, Regular Expressions

File: 20_Information_Extraction.py

15. Information Retrieval

Information retrieval is the process of finding relevant documents from a collection in response to a user's query.

Objectives:

Process user queries.
Represent documents numerically.
Rank documents by relevance.
Return the most relevant results.

Common techniques:

TF-IDF
Cosine similarity
Inverted indexing
Query-document matching

Example:

Query:

text
machine learning applications


The system ranks documents according to how closely they match the query.

Applications:

Search engines
Digital libraries
Document search systems
Question-answering systems

File: 21_Information_Retrieval.py

▶️ Running the Practicals

Navigate to the Unit 1 directory:

bash
cd "Unit 1"


Run any practical using Python:

bash
python 1_tokenization.py
python 08_BOW.py
python 09_TFIDF.py
python 10_NGram.py
python 11_Cosine_Similarity.py
python 12_Word2Vec.py
python 13_GloVe.py
python 14_WMD.py
python 15_Text_Classification.py
python 16_Sentiment_Analysis.py
python 17_LDA.py
python 18_LSA.py
python 19_Opinion_Mining.py
python 20_Information_Extraction.py
python 21_Information_Retrieval.py


Run each command separately. Make sure any required datasets, pretrained embeddings, and language models have been installed or downloaded.

🎯 Learning Objectives

After completing these practicals, students should be able to:

Understand fundamental NLP concepts and text preprocessing.
Convert text into numerical representations.
Apply statistical and semantic text-similarity methods.
Understand word embeddings and their applications.
Build basic text-classification and sentiment-analysis systems.
Explore topic modeling and latent semantic structures.
Extract entities, opinions, and useful information from text.
Understand the basic workflow of information retrieval systems.
🛠️ Tools and Technologies
Tool	Purpose
Python	Programming language
NLTK	Text processing and linguistic analysis
spaCy	NLP pipelines and entity extraction
Scikit-learn	Vectorization, similarity, and machine learning
Gensim	Word embeddings and topic modeling
NumPy	Numerical computations
Pandas	Data handling
Regular Expressions	Pattern matching and text extraction
📚 Applications of NLP

The techniques implemented in this repository provide a foundation for building:

Chatbots and conversational assistants
Search engines
Text summarization systems
Sentiment-analysis applications
Document classification systems
Recommendation systems
Question-answering systems
Information extraction pipelines
👨‍💻 Author

Abhay Pratap Singh

B.Tech — Computer Science / Artificial Intelligence

📄 Academic Information
Course: Natural Language Processing
Unit: Unit 1
Course Outcome: CO1
🤝 Contributing

This repository is intended for academic learning and experimentation. Suggestions, improvements, and corrections are welcome.

⭐ Support

If this repository helps you understand NLP concepts or prepare for practical examinations, consider giving it a ⭐ on GitHub.

This repository is maintained for educational purposes and practical implementation of Natural Language Processing techniques.
