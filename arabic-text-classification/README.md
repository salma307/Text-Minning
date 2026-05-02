# Impact of Feature Selection on Arabic Text Classification

##  Overview

This project investigates the impact of different **feature selection methods** on Arabic text classification using **Naive Bayes** classifiers. The study compares classification performance with and without feature selection on an Arabic news/text dataset.

## Project Structure

```
arabic-text-classification/
├── impact-of-feature-selection-on-arabic-text-classif.ipynb   # Main notebook
├── README.md
└── LICENSE
```

##  Dataset

- **Source:** [Arabic Classification Dataset — Kaggle](https://www.kaggle.com/datasets/salmahafid/arabic-classification)
- **Content:** Arabic texts with category labels
- **Preprocessing:** 10,000 balanced samples per class

> ⚠️ The dataset is not included in this repository. Download it from Kaggle and place the CSV file at:
> `/kaggle/input/datasets/salmahafid/arabic-classification/arabic_dataset_classifiction.csv`

##  Methodology

### Text Preprocessing
- Removal of diacritics, URLs, Latin characters, and punctuation
- Arabic normalization (e.g., `إأآا → ا`, `ى → ي`, `ة → ه`)
- Stop word removal and tokenization

### Feature Extraction
- **TF-IDF Vectorization** (unigrams + bigrams)
- **CountVectorizer**

### Feature Selection Methods Compared
| Method | Description |
|--------|-------------|
| No Feature Selection | Baseline — all features |
| Chi² (SelectKBest) | Statistical test for feature relevance |
| ANOVA F-score | Analysis of variance-based selection |

### Classifiers
- **Multinomial Naive Bayes**
- **Complement Naive Bayes**
- **Gaussian Naive Bayes**

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Cross-validation (Stratified K-Fold)

##  How to Run

1. Clone this repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/salmahafid/arabic-classification)
3. Open the notebook on **Kaggle** or locally with Jupyter
4. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk scipy
```

5. Run all cells in order

##  Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
scipy
```

##  Results

The notebook produces comparison charts and tables showing the effect of each feature selection method on classification accuracy across multiple Naive Bayes variants.

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
