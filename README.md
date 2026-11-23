# Natural Language Processing Practicals (1-32)

Python implementations for all 32 practicals across Units 1-3. The repository is grouped by the course outcome (CO) from the supplied practical list.

Practical 7 uses **FastCoref**, a neural coreference resolver integrated with spaCy, as the requested neuralcoref-style alternative. See the [FastCoref project guide](https://github.com/shon-otmazgin/fastcoref) for its current usage details.

## Practical index

| No. | Practical | CO | Code |
|---:|---|:---:|---|
| 1 | Tokenization of Sentences and Words using NLTK and spaCy | CO1 | [`1_tokenization.py`](<Unit 1 - CO1/1_tokenization.py>) |
| 2 | Stemming and Lemmatization on sample text | CO1 | [`2_stemming_lemmatization.py`](<Unit 1 - CO1/2_stemming_lemmatization.py>) |
| 3 | Stop-word Removal from a document | CO1 | [`3_stopword_removal.py`](<Unit 1 - CO1/3_stopword_removal.py>) |
| 4 | Part-of-Speech (POS) Tagging of a given sentence | CO1 | [`4_pos_tagging.py`](<Unit 1 - CO1/4_pos_tagging.py>) |
| 5 | Parsing and Chunking using regex and spaCy | CO1 | [`5_parsing_chunking.py`](<Unit 1 - CO1/5_parsing_chunking.py>) |
| 6 | Named Entity Recognition (NER) using spaCy | CO1 | [`6_ner.py`](<Unit 1 - CO1/6_ner.py>) |
| 7 | Coreference Resolution using neuralcoref (or similar) | CO2 | [`07_coreference_resolution.py`](<Unit 2 - CO2/07_coreference_resolution.py>) |
| 8 | Bag-of-Words (BoW) vectorization and representation | CO2 | [`08_BOW.py`](<Unit 2 - CO2/08_BOW.py>) |
| 9 | TF-IDF Implementation and comparison with BoW | CO2 | [`09_TFIDF.py`](<Unit 2 - CO2/09_TFIDF.py>) |
| 10 | N-Gram Model (uni-, bi-, tri-gram) generation from corpus | CO2 | [`10_NGram.py`](<Unit 2 - CO2/10_NGram.py>) |
| 11 | Cosine Similarity computation between text documents | CO2 | [`11_Cosine_Similarity.py`](<Unit 2 - CO2/11_Cosine_Similarity.py>) |
| 12 | Word2Vec word embeddings using gensim on a custom corpus | CO2 | [`12_Word2Vec.py`](<Unit 2 - CO2/12_Word2Vec.py>) |
| 13 | GloVe Embeddings loading and vector representation | CO2 | [`13_GloVe.py`](<Unit 2 - CO2/13_GloVe.py>) |
| 14 | Text Similarity using Word Mover’s Distance (WMD) | CO2 | [`14_WMD.py`](<Unit 2 - CO2/14_WMD.py>) |
| 15 | Text Classification using Naive Bayes/SVM with TF-IDF | CO2 | [`15_Text_Classification.py`](<Unit 2 - CO2/15_Text_Classification.py>) |
| 16 | Sentiment Analysis using TextBlob and VADER | CO2 | [`16_Sentiment_Analysis.py`](<Unit 2 - CO2/16_Sentiment_Analysis.py>) |
| 17 | Topic Modeling using Latent Dirichlet Allocation (LDA) | CO2 | [`17_LDA.py`](<Unit 2 - CO2/17_LDA.py>) |
| 18 | Topic Modeling using Latent Semantic Analysis (LSA) | CO2 | [`18_LSA.py`](<Unit 2 - CO2/18_LSA.py>) |
| 19 | Opinion Mining on product/service reviews dataset | CO2 | [`19_Opinion_Mining.py`](<Unit 2 - CO2/19_Opinion_Mining.py>) |
| 20 | Information Extraction (IE) from structured/unstructured documents | CO2 | [`20_Information_Extraction.py`](<Unit 2 - CO2/20_Information_Extraction.py>) |
| 21 | Information Retrieval system with ranking using TF-IDF | CO2 | [`21_Information_Retrieval.py`](<Unit 2 - CO2/21_Information_Retrieval.py>) |
| 22 | Sequence Classification using RNN/LSTM (Keras/TensorFlow) | CO2 | [`exp22_lstm_sequence_classification.py`](<Unit 2 - CO2/exp22_lstm_sequence_classification.py>) |
| 23 | Implementing Attention Mechanism (Basic custom model) | CO2 | [`exp23_custom_attention.py`](<Unit 2 - CO2/exp23_custom_attention.py>) |
| 24 | Fine-Tuning BERT for text classification using Hugging Face | CO3 | [`exp24_bert_finetune_classification.py`](<Unit 3 - CO3/exp24_bert_finetune_classification.py>) |
| 25 | Sentence Embeddings using BERT and GPT-2 | CO3 | [`exp25_sentence_embeddings_bert_gpt2.py`](<Unit 3 - CO3/exp25_sentence_embeddings_bert_gpt2.py>) |
| 26 | Text Generation using GPT-2 on a custom dataset | CO3 | [`exp26_gpt2_text_generation.py`](<Unit 3 - CO3/exp26_gpt2_text_generation.py>) |
| 27 | Named Entity Recognition using Transformers (BERT) | CO3 | [`exp27_ner_bert.py`](<Unit 3 - CO3/exp27_ner_bert.py>) |
| 28 | Machine Translation using MarianMT or T5 (Hugging Face) | CO3 | [`exp28_machine_translation.py`](<Unit 3 - CO3/exp28_machine_translation.py>) |
| 29 | Text Summarization using BART or T5 | CO3 | [`exp29_text_summarization.py`](<Unit 3 - CO3/exp29_text_summarization.py>) |
| 30 | Chatbot Development using pre-trained transformer (DialoGPT/ChatGPT-style) | CO3 | [`exp30_chatbot_dialogpt.py`](<Unit 3 - CO3/exp30_chatbot_dialogpt.py>) |
| 31 | Automatic Document Classification and Separation using ML | CO3 | [`exp31_document_classification_separation.py`](<Unit 3 - CO3/exp31_document_classification_separation.py>) |
| 32 | Hybrid Probabilistic + FSM-Based Sequence Modeling for document boundary detection | CO3 | [`exp32_hybrid_hmm_fsm_boundary_detection.py`](<Unit 3 - CO3/exp32_hybrid_hmm_fsm_boundary_detection.py>) |

## Repository layout

- `Unit 1 - CO1/`: Practicals 1-6
- `Unit 2 - CO2/`: Practicals 7-23
- `Unit 3 - CO3/`: Practicals 24-32
- `docs/`: combined practical code and captured run-output PDF

## Requirements and setup

Use Python 3.10 or 3.11 for broad compatibility across the packages used here. Create a virtual environment, then install the full requirements set or install only the unit you need.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

The full requirements set includes both TensorFlow and PyTorch and can be large. To install by unit instead:

```bash
pip install -r requirements-unit1.txt
pip install -r requirements-unit2.txt
pip install -r requirements-unit3.txt
```

For the NLTK and spaCy examples, install language resources once:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(x) for x in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
```

## Run a practical

Run each script separately from the repository root. For example:

```bash
python "Unit 1 - CO1/1_tokenization.py"
python "Unit 2 - CO2/07_coreference_resolution.py"
python "Unit 3 - CO3/exp29_text_summarization.py"
```

Some practicals download datasets, pretrained word embeddings, or transformer models on first use, so an internet connection is required. Those downloads can use substantial disk space and memory. The chatbot practical is interactive. Transformer workloads run faster with a compatible GPU, but GPU use is optional for many examples.

## Coreference practical

Practical 7 runs FastCoref through spaCy and uses the pretrained `biu-nlp/f-coref` model by default. The model is downloaded the first time the practical runs. It prints the original text, resolved text, and detected mention clusters.

## GitHub upload

1. Create a new, empty repository on GitHub.
2. Extract the `NLP_Practicals_GitHub_Ready.zip` archive.
3. Open a terminal in the extracted `NLP_Practicals_GitHub` folder and run:

```bash
git init
git add .
git commit -m "Add NLP practicals 1-32"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

Replace `USERNAME/REPOSITORY` with the new repository’s address. Git must be installed and configured with your GitHub account first.

## About the included PDF

The PDF places the matching scanned output pages immediately after each of the 32 code programs. The output pages come from the supplied `NLP FINAL.pdf`; its practical order differs, so each program page cites the matching original PDF page range. The README remains a separate file and is not included in the PDF.
