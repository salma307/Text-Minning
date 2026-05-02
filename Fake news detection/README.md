# Text Mining Mini Project — Fake News Detection

> Comparison before/after Feature Selection using **Naive Bayes**

---

## Overview

This project is a **Text Mining** mini-project that explores the impact of different **feature selection** methods on the performance of a **Naive Bayes** classifier for fake news detection.

Two experimental setups are compared:
- **With data leakage** : feature selection is performed on the entire dataset before the train/test split.
- **Without data leakage** : feature selection is strictly applied after the split, using only training data — the correct approach.

---

## Datasets

### Dataset 1 — Fake News Dataset (English)

| File | Kaggle Link |
|---|---|
| `Fake.csv` | [https://www.kaggle.com/datasets/salmahafid/fake-csv](https://www.kaggle.com/datasets/salmahafid/fake-csv) |
| `True.csv` | [https://www.kaggle.com/datasets/salmahafid/true-csv](https://www.kaggle.com/datasets/salmahafid/true-csv) |

- **Total size :** 44,898 articles (after merging and shuffling)
- **Columns :** `title`, `text`, `subject`, `date`, `label` (0 = Fake, 1 = Real)
- **Language :** English



##  Feature Selection Methods

| Method | Description |
|---|---|
| No Selection | All TF-IDF features kept |
| Fisher Score (Chi²) | Selection via chi-squared test |
| ANOVA (f_classif) | Selection via analysis of variance |
| LDA | Dimensionality reduction to 1 discriminant component |

---

##  Results — Dataset 1 (English Fake News)

### With data leakage

| Method | Nb Features | Accuracy | F1-Score | CV Mean |
|---|---|---|---|---|
| No Selection | 7,043 | 99.92% | 99.92% | 99.91% |
| Fisher (chi2) | 5,000 | 99.92% | 99.92% | 99.90% |
| ANOVA | 5,000 | 99.93% | 99.93% | 99.92% |
| LDA | 1 | 99.48% | 99.48% | 99.45% |

### Without data leakage (real results)

| Method | Nb Features | Accuracy | F1-Score | CV Mean |
|---|---|---|---|---|
| Fisher (chi2) | 6,400 | 93.53% | 93.53% | 93.40% |
| ANOVA | 6,400 | 94.95% | 94.95% | 94.80% |
| LDA | 1 | 97.61% | 97.61% | 97.55% |

> ⚠️ The large gap between the two setups clearly illustrates the danger of **data leakage** in ML pipelines.

---

##  Tech Stack

- **Language :** Python 3.11
- **Main libraries :**
  - `pandas`, `numpy` — data manipulation
  - `scikit-learn` — TF-IDF vectorization, feature selection, Naive Bayes, evaluation
  - `nltk` — text preprocessing (stopwords, stemming)
  - `matplotlib`, `seaborn` — visualization
  - `scipy` — statistical tests

---


##  Project Structure

```
text-mining-fake-news/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── comparative-study-feature-selection-impact-on-nb.ipynb   
├── README.md
└── LICENSE
```

---

##  License

This project is distributed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
