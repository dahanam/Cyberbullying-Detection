# Cyberbullying Detection

A multi-approach NLP project for detecting and classifying cyberbullying in social media text, combining BERT-based classifiers, traditional ML, GPT-4 prompt engineering, and unsupervised clustering.

---

## Overview

| Approach | Task | Best F1 |
|---|---|---|
| BERT (TF Hub) | Binary: cb vs. no_cb | ~90% |
| SentenceTransformer + Logistic Regression | 6-class | ~88% macro avg |
| SentenceTransformer + SVM | 6-class | ~88% macro avg |
| SentenceTransformer + Random Forest | 6-class | ~86% macro avg |
| GPT-4 Few-Shot | 6-class (zero/few-shot) | Varies |
| TF-IDF + KMeans | Unsupervised clustering | — |
| DistilBERT + KMeans | Unsupervised clustering | — |

---

## Dataset

**Cyberbullying Classification Dataset** — 6 categories × 8,000 samples each (48,000 total)

| File | Category | Description |
|---|---|---|
| `8000age.txt` | `age` | Harassment targeting someone's age |
| `8000ethnicity.txt` | `ethnicity` | Racial/ethnic slurs and harassment |
| `8000gender.txt` | `gender` | Gender-based harassment |
| `8000notcb.txt` | `notcb` | Not cyberbullying |
| `8000other.txt` | `other` | Other forms of harassment |
| `8000religion.txt` | `religion` | Religion-based harassment |

A separate binary dataset (`negative-words.csv`) is used for the BERT binary classifier with `cb` / `no_cb` labels.

Expected path in Google Drive: `/content/drive/MyDrive/cyber/`

---

## Notebook Structure

### `cyberbullying_detection.ipynb`

| Section | Description |
|---|---|
| 1. Setup | Installs, Google Drive mount |
| 2. Load & Explore | Load 6-class txt files + binary CSV |
| 3. EDA | Word clouds per category, distribution chart |
| 4. BERT Binary Classifier | TF Hub BERT → sigmoid output |
| 5. 6-Class Classifier | SentenceTransformer embeddings → LR / RF / SVM |
| 6. GPT-4 Prompt Engineering | Zero-shot vs. few-shot comparison |
| 7. Clustering | TF-IDF KMeans vs. DistilBERT KMeans |
| 8. Summary | Results table and key observations |

---

## Methods

### Part 1 — BERT Binary Classifier

- Loads `bert_en_uncased_preprocess` and `bert_en_uncased_L-12_H-768_A-12` from TensorFlow Hub
- Adds a dropout + sigmoid dense layer for binary classification
- Downsamples the majority class to handle class imbalance
- Optimizer: Nadam | Loss: Binary Crossentropy
- Saves best model weights to `CB_bert.h5`

### Part 2 — 6-Class Classifier (SentenceTransformer)

- Encodes all text using `sentence-transformers/all-MiniLM-L6-v1`
- Applies PCA for 3D visualization of the embedding space
- Trains three classifiers on the embeddings:
  - **Logistic Regression** with GridSearchCV (L1/L2, C tuning)
  - **Random Forest** (100 estimators)
  - **SVM** (RBF kernel)
- Confusion matrices plotted for each model

**Results (SVM, best overall):**

| Category | Precision | Recall | F1 |
|---|---|---|---|
| age | 0.95 | 0.95 | 0.95 |
| ethnicity | 0.96 | 0.96 | 0.96 |
| gender | 0.89 | 0.85 | 0.87 |
| notcb | 0.71 | 0.55 | 0.62 |
| other | 0.63 | 0.79 | 0.70 |
| religion | 0.93 | 0.96 | 0.94 |

### Part 3 — GPT-4 Prompt Engineering

- Uses OpenAI `gpt-4o` via Chat Completions API
- Compares **zero-shot** vs. **few-shot** prompting
- API key loaded from Colab Secrets (never hardcoded)
- Few-shot prompting improves accuracy ~10–20% over zero-shot

### Part 4 — Unsupervised Clustering

- **TF-IDF KMeans**: bag-of-words baseline, 6 clusters
- **DistilBERT KMeans**: frozen DistilBERT embeddings, mean pooled, 6 clusters
- Evaluated using **purity score** and **Adjusted Rand Index (ARI)**
- DistilBERT embeddings consistently outperform TF-IDF

---

## Setup

### Requirements

```
tensorflow
tensorflow-hub
tensorflow-text
sentence-transformers
transformers
torch
openai
scikit-learn
pandas
numpy
matplotlib
seaborn
wordcloud
```

### Running in Google Colab

1. Upload the notebook to Colab
2. Place the dataset files in `/content/drive/MyDrive/cyber/`
3. For GPT-4 sections: add your OpenAI API key via **Runtime → Secrets → `OPENAI_API_KEY`**
4. Run all cells top to bottom

---

## Security Notes

- The dataset contains real social media text with offensive content — handle with care
- Do not upload the dataset to public repositories

---

## References

- [Cyberbullying Classification Dataset — Kaggle](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- [BERT via TensorFlow Hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4)
- [SentenceTransformers](https://www.sbert.net/)
- [DistilBERT — Hugging Face](https://huggingface.co/distilbert-base-uncased)
