Natural Language Processing — Practicals (Units 1–3)

Welcome to the Natural Language Processing (NLP) Practicals repository.

This repository contains Python implementations of fundamental and advanced NLP techniques. Unit 1 covers classical text processing with NLTK, spaCy, Gensim, and Scikit-learn. Units 2 and 3 move to neural sequence models, attention, and Transformer-based applications using TensorFlow/Keras, PyTorch, and Hugging Face.

📌 Practical List
Unit 1 — Classical NLP (CO1)
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
Unit 2 — Neural Sequence Models (CO2)
No.	Experiment	Description
22	Sequence Classification using RNN/LSTM	Classifying text sequences with a Bidirectional LSTM (Keras/TensorFlow)
23	Attention Mechanism	Implementing a custom additive (Bahdanau-style) attention layer
Unit 3 — Transformers and Applications (CO3)
No.	Experiment	Description
24	Fine-Tuning BERT	Text classification with BERT using Hugging Face
25	Sentence Embeddings	Sentence representations using BERT and GPT-2
26	Text Generation with GPT-2	Fine-tuning GPT-2 on a custom dataset and generating text
27	Named Entity Recognition	Token classification with BERT on CoNLL-2003
28	Machine Translation	Translating text with MarianMT or T5
29	Text Summarization	Abstractive summarization with BART or T5
30	Chatbot Development	Conversational agent with a pre-trained DialoGPT model
31	Document Classification and Separation	Classifying pages and splitting a page stream into documents using ML
32	Hybrid Probabilistic + FSM Sequence Modeling	Document boundary detection with HMM-style Viterbi decoding and FSM constraints
📂 Repository Structure
text
NLP-Practicals/
│
├── requirements.txt
├── README.md
│
├── Unit 1/
│   ├── 1_tokenization.py
│   ├── 08_BOW.py
│   ├── 09_TFIDF.py
│   ├── 10_NGram.py
│   ├── 11_Cosine_Similarity.py
│   ├── 12_Word2Vec.py
│   ├── 13_GloVe.py
│   ├── 14_WMD.py
│   ├── 15_Text_Classification.py
│   ├── 16_Sentiment_Analysis.py
│   ├── 17_LDA.py
│   ├── 18_LSA.py
│   ├── 19_Opinion_Mining.py
│   ├── 20_Information_Extraction.py
│   └── 21_Information_Retrieval.py
│
├── Unit 2/
│   ├── 22_LSTM_Sequence_Classification.py
│   └── 23_Attention_Mechanism.py
│
└── Unit 3/
    ├── 24_BERT_Fine_Tuning.py
    ├── 25_Sentence_Embeddings.py
    ├── 26_GPT2_Text_Generation.py
    ├── 27_NER_BERT.py
    ├── 28_Machine_Translation.py
    ├── 29_Text_Summarization.py
    ├── 30_Chatbot.py
    ├── 31_Document_Classification_Separation.py
    └── 32_Hybrid_HMM_FSM_Boundary_Detection.py
🔧 Requirements
Python 3.9 or later
Unit 1: NLTK, spaCy, NumPy, Pandas, Scikit-learn, Gensim, Matplotlib (optional)
Unit 2: TensorFlow / Keras, NumPy
Unit 3: PyTorch, Transformers, Datasets, Scikit-learn, seqeval, sentencepiece, sacremoses, accelerate

The exact dependencies depend on the practical being executed. A GPU (for example a free Google Colab runtime) is recommended for Units 2 and 3.

⚙️ Installation
Clone the repository
bash
git clone <YOUR_REPOSITORY_URL>
cd NLP-Practicals

Replace <YOUR_REPOSITORY_URL> with your GitHub repository URL.

Create a virtual environment (recommended)
bash
python -m venv venv

Activate it on Windows:

bash
venv\Scripts\activate

Activate it on Linux or macOS:

bash
source venv/bin/activate
Install dependencies
bash
pip install -r requirements.txt
Download required NLP resources (Unit 1) For NLTK:
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

Some practicals may require additional corpora, pretrained embeddings, or datasets. In Units 2 and 3, datasets (IMDB, CoNLL-2003, 20 Newsgroups) and pretrained models (BERT, GPT-2, MarianMT, BART, T5, DialoGPT) are downloaded automatically on first run, so an internet connection is required.

🧪 Experiments — Unit 1
Tokenization

Tokenization is the process of splitting text into smaller units called tokens.

Objectives: perform word tokenization; perform sentence tokenization; understand basic text preprocessing.

Example: Natural Language Processing is interesting. → ['Natural', 'Language', 'Processing', 'is', 'interesting', '.']

Technologies: Python, NLTK, spaCy

Bag of Words (BOW)

The Bag of Words model represents a document as a vector of word counts, without considering word order.

Objectives: create a vocabulary from text; convert documents into numerical vectors; understand text vectorization.

Example: Document 1: I like NLP, Document 2: I like Python → vocabulary ['I', 'like', 'NLP', 'Python']. Each document is represented by the frequencies of vocabulary terms.

Technology: Scikit-learn | File: 08_BOW.py

TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF measures the importance of a word in a document relative to a collection of documents.

Objectives: calculate term importance; convert text into weighted numerical vectors; reduce the relative importance of common terms.

Concept: TF is the frequency of a term in a document; IDF measures how rare or informative the term is across documents.

Technology: Scikit-learn | File: 09_TFIDF.py

N-Gram Model

An N-gram is a contiguous sequence of n tokens extracted from text (unigram, bigram, trigram).

Example: Natural Language Processing is useful → bigrams: Natural Language, Language Processing, Processing is, is useful.

Applications: language modeling, text prediction, text classification, phrase analysis.

File: 10_NGram.py

Cosine Similarity

Cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them.

Objectives: convert documents into vectors; calculate similarity scores; compare documents based on their textual content. A score closer to 1 generally indicates greater similarity for non-negative text vectors.

Applications: document comparison, search engines, recommendation systems.

File: 11_Cosine_Similarity.py

Word2Vec

Word2Vec learns dense word embeddings from surrounding-word patterns in a text corpus.

Architectures: CBOW (Continuous Bag of Words), Skip-gram.

Objectives: generate vector representations of words; explore relationships between words; understand distributional semantics.

Technology: Gensim | File: 12_Word2Vec.py

GloVe (Global Vectors for Word Representation)

GloVe learns vector representations using global word co-occurrence statistics.

Objectives: understand pretrained word embeddings; represent words as dense numerical vectors; explore semantic relationships.

Unlike Word2Vec, GloVe is commonly used through pretrained vectors or a separate training implementation. The practical may require downloading a pretrained embedding file.

File: 13_GloVe.py

Word Mover's Distance (WMD)

WMD measures the semantic distance between two documents by calculating the minimum transport cost required to move the word embeddings of one document to match those of another.

Objectives: compare documents using word embeddings; understand semantic distance; examine similarities that simple word-overlap methods may miss.

WMD requires suitable word embeddings and may need additional Gensim configuration.

File: 14_WMD.py

Text Classification

Text classification assigns predefined categories or labels to text.

Objectives: prepare and preprocess text data; extract numerical text features; train a classification model; predict labels for new text.

Common algorithms: Naive Bayes, Logistic Regression, Support Vector Machines (SVM).

File: 15_Text_Classification.py

Sentiment Analysis

Sentiment analysis determines the emotional polarity or opinion expressed in a text.

Example: The movie was excellent and enjoyable. → Positive sentiment.

File: 16_Sentiment_Analysis.py

Latent Dirichlet Allocation (LDA)

LDA is a probabilistic topic-modeling algorithm that discovers hidden topics in a collection of documents. Documents are represented as topic distributions and topics as word distributions.

Technology: Gensim or Scikit-learn | File: 17_LDA.py

Latent Semantic Analysis (LSA)

LSA identifies latent semantic patterns in a document-term matrix, typically using Singular Value Decomposition (SVD).

Technology: Scikit-learn | File: 18_LSA.py

Opinion Mining

Opinion mining extracts opinions, attitudes, and sentiment from textual data, and may include aspect-based analysis.

Example: The camera is excellent, but the battery life is poor. → Camera: Positive, Battery life: Negative.

File: 19_Opinion_Mining.py

Information Extraction

Information extraction identifies useful structured information from unstructured text using NER, relationship extraction, and regular expressions.

Example: Google opened a new office in London. → Google → ORGANIZATION, London → LOCATION.

Technology: spaCy, NLTK, Regular Expressions | File: 20_Information_Extraction.py

Information Retrieval

Information retrieval is the process of finding relevant documents from a collection in response to a user's query, using TF-IDF, cosine similarity, inverted indexing, and query-document matching.

Example: Query machine learning applications → documents ranked by how closely they match.

File: 21_Information_Retrieval.py

🧪 Experiments — Unit 2 (CO2)
Practical 22 — Sequence Classification using RNN/LSTM

Recurrent networks process text one token at a time and carry a hidden state, which lets them model word order. An LSTM adds gates that help retain information over long sequences. This practical classifies IMDB movie reviews as positive or negative using an Embedding layer followed by a Bidirectional LSTM.

Objectives: pad and encode variable-length sequences; build an Embedding + BiLSTM classifier; train, evaluate, and predict on new reviews.

Technology: TensorFlow, Keras | File: Unit 2/22_LSTM_Sequence_Classification.py

Practical 23 — Attention Mechanism (Basic Custom Model)

Attention lets a model weigh each time step by relevance instead of relying on only the final hidden state. This practical implements a custom additive (Bahdanau-style) attention layer on top of a BiLSTM and prints the words the model attends to most.

Concept: score = vᵀ·tanh(W·h), α = softmax(score), context = Σ αₜ·hₜ.

Objectives: write a custom Keras layer; compute attention weights; inspect which words influence a prediction.

Technology: TensorFlow, Keras | File: Unit 2/23_Attention_Mechanism.py

🧪 Experiments — Unit 3 (CO3)
Practical 24 — Fine-Tuning BERT for Text Classification

BERT is a pre-trained bidirectional Transformer encoder. Fine-tuning adds a classification head and updates the weights on a labelled dataset.

Objectives: tokenize with a pre-trained tokenizer; fine-tune with the Hugging Face Trainer; evaluate accuracy and F1; run inference on new text.

Technology: Hugging Face Transformers, Datasets | File: Unit 3/24_BERT_Fine_Tuning.py

Practical 25 — Sentence Embeddings using BERT and GPT-2

Sentence embeddings turn a sentence into a single dense vector by pooling token representations. This practical computes mean-pooled embeddings from BERT (encoder) and GPT-2 (decoder) and compares sentences with cosine similarity.

Objectives: extract hidden states; apply attention-mask-aware mean pooling; compare how the two models place similar and unrelated sentences.

File: Unit 3/25_Sentence_Embeddings.py

Practical 26 — Text Generation using GPT-2 on a Custom Dataset

GPT-2 is an autoregressive language model. Fine-tuning it on your own text makes its generations follow the style and vocabulary of that text.

Objectives: prepare a custom text corpus; fine-tune GPT-2 with a causal language-modelling objective; generate text with top-k / top-p sampling.

Usage: python 26_GPT2_Text_Generation.py mydata.txt (a small demo corpus is used if no file is given).

File: Unit 3/26_GPT2_Text_Generation.py

Practical 27 — Named Entity Recognition using BERT

NER labels each token as part of an entity such as a person, organization, or location. It is treated as token classification.

Example: Sundar Pichai works at Google in California. → Sundar Pichai → PER, Google → ORG, California → LOC.

Objectives: align word-level labels with sub-word tokens; fine-tune a token-classification model on CoNLL-2003; evaluate with entity-level F1 (seqeval).

File: Unit 3/27_NER_BERT.py

Practical 28 — Machine Translation using MarianMT or T5

Neural machine translation uses sequence-to-sequence Transformers to translate between languages. MarianMT provides compact pre-trained models per language pair, while T5 treats translation as text-to-text.

Objectives: load a pre-trained translation model; translate batches with beam search; compare MarianMT and T5 output. Change the model name to switch language pairs (for example opus-mt-en-hi for Hindi).

File: Unit 3/28_Machine_Translation.py

Practical 29 — Text Summarization using BART or T5

Abstractive summarization generates a shorter text that captures the key points of a longer one. BART (fine-tuned on CNN/DailyMail) and T5 are encoder–decoder models that do this well.

Objectives: run summarization pipelines; control summary length; compare BART and T5 output.

File: Unit 3/29_Text_Summarization.py

Practical 30 — Chatbot Development using a Pre-trained Transformer

This practical builds a command-line chatbot on DialoGPT, a GPT-2 model trained on conversational data. The conversation history is fed back as context for every turn.

Objectives: manage multi-turn dialogue context; generate replies with sampling and repetition control; build an interactive loop (type quit to exit).

File: Unit 3/30_Chatbot.py

Practical 31 — Automatic Document Classification and Separation using ML

Real scanning pipelines often produce one long stream of pages from many documents. This practical first trains a TF-IDF + Logistic Regression classifier, then splits a concatenated page stream into documents by detecting changes in predicted class and in page-to-page similarity.

Objectives: classify pages by topic; detect document boundaries; evaluate boundary precision and recall.

File: Unit 3/31_Document_Classification_Separation.py

Practical 32 — Hybrid Probabilistic + FSM-Based Sequence Modeling

Each page is either a START (first page of a document) or a CONT (continuation). The probabilistic part scores pages with a classifier and learned transition probabilities and decodes them with the Viterbi algorithm, like an HMM. The finite-state-machine part adds hard rules: the stream must begin with START, and a document cannot exceed a maximum page count. The hybrid decoder is compared against a classifier-only baseline.

Objectives: model document boundaries as a sequence-labelling problem; combine probabilistic decoding with rule-based constraints; compare precision, recall, and F1 with and without the sequence model.

File: Unit 3/32_Hybrid_HMM_FSM_Boundary_Detection.py

▶️ Running the Practicals

Run each practical from inside its unit folder:

bash
cd "Unit 1"
python 1_tokenization.py
python 08_BOW.py
# ... and so on up to 21_Information_Retrieval.py

cd "../Unit 2"
python 22_LSTM_Sequence_Classification.py
python 23_Attention_Mechanism.py

cd "../Unit 3"
python 24_BERT_Fine_Tuning.py
python 25_Sentence_Embeddings.py
python 26_GPT2_Text_Generation.py
python 27_NER_BERT.py
python 28_Machine_Translation.py
python 29_Text_Summarization.py
python 30_Chatbot.py
python 31_Document_Classification_Separation.py
python 32_Hybrid_HMM_FSM_Boundary_Detection.py

Run each command separately. Make sure any required datasets, pretrained embeddings, and language models have been installed or downloaded. The larger Transformer practicals (26, 28, 29) run best on a GPU.

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
Build and train RNN/LSTM sequence models and implement a custom attention mechanism.
Fine-tune pre-trained Transformers (BERT, GPT-2) for classification, generation, and token-level tasks.
Apply Transformer models to translation, summarization, and dialogue.
Combine machine-learning models with rule-based state machines for document processing.
🛠️ Tools and Technologies
Tool	Purpose
Python	Programming language
NLTK	Text processing and linguistic analysis
spaCy	NLP pipelines and entity extraction
Scikit-learn	Vectorization, similarity, and machine learning
Gensim	Word embeddings and topic modeling
NumPy / Pandas	Numerical computation and data handling
TensorFlow / Keras	RNN/LSTM models and custom attention layers
PyTorch	Backend for Transformer models
Hugging Face Transformers and Datasets	Pre-trained BERT, GPT-2, MarianMT, BART, T5, DialoGPT
seqeval	Entity-level NER evaluation
Regular Expressions	Pattern matching and text extraction
📚 Applications of NLP

The techniques implemented in this repository provide a foundation for building:

Chatbots and conversational assistants
Search engines
Text summarization systems
Sentiment-analysis applications
Document classification and scanning pipelines
Recommendation systems
Question-answering systems
Information extraction pipelines
Machine translation systems
👨‍💻 Author

Abhay Pratap Singh

B.Tech — Computer Science / Artificial Intelligence

📄 Academic Information
Course: Natural Language Processing
Unit 1: Course Outcome CO1
Unit 2: Course Outcome CO2
Unit 3: Course Outcome CO3
🤝 Contributing

This repository is intended for academic learning and experimentation. Suggestions, improvements, and corrections are welcome.

⭐ Support

If this repository helps you understand NLP concepts or prepare for practical examinations, consider giving it a ⭐ on GitHub.

This repository is maintained for educational purposes and practical implementation of Natural Language Processing techniques.
