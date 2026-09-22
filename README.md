# IMDb Review Rating Prediction - Classification, Regression and Clustering

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f5b)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

Predicting the star rating of $50.000$ IMDb movie reviews from their text alone, approached three ways: as an 8-class classification problem, as a regression problem over the rating scale, and as unsupervised topic discovery with K-Means.

The pipeline builds its own vocabulary and turns it into a $30.000$-feature TF-IDF matrix over unigrams and bigrams. Six supervised models are tuned with `GridSearchCV` and compared on held-out test and validation splits.

> Academic project - Machine Learning (*Aprendizagem Automática*), BSc in Computer Science and Multimedia Engineering (LEIM), ISEL.

---

## Dataset

The [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., Stanford), consisting of $50.000$ IMDb reviews, evenly balanced between negative and positive:

| | Ratings | Reviews |
| :-- | :-: | :-: |
| Negative | $1 – 4$ | $25,000$ |
| Positive | $7 – 10$ | $25,000$ |

The reviews ship with the repository as pickles in `data/`: `imdbFull.p` (raw) and `imdbPreProcessed.p`. Together they are about $124$ MB.

## Text preprocessing

Raw review text is normalized before vectorization:

- `<br />` tags stripped, non-alphabetic characters removed, whitespace collapsed, lowercased.
- Words of 20+ characters dropped, and words with a letter repeated 3+ times in a row (`yaaass`, `omgggg`).
- **Snowball stemming** to collapse inflected forms onto a common root.

Vectorization uses `TfidfVectorizer` with:

| Setting | Value | Why |
| :-- | :-- | :-- |
| `ngram_range` | $(1, 2)$ | Bigrams capture local context such as *not good* |
| stopwords | English, minus `no`, `not`, `nor` | Removing negations would invert the polarity of the phrase |
| `max_features` | $30,000$ | Caps dimensionality and memory |
| `min_df` / `max_df` | $3$ / $0.8$ | Drops typos and near-universal terms |
| `sublinear_tf` | True | Dampens the effect of term repetition |
| `token_pattern` | 2+ letters | Discards single characters |

## Method

The data is split $80$ / $10$ / $10$ with stratification on the rating, preserving the class distribution across all three subsets:

| Train | Test | Validation |
| :-: | :-: | :-: |
| $40,000$ | $5,000$ | $5,000$ |

Every model is tuned with `GridSearchCV` (3-fold) and then evaluated on both the test and validation splits. The regression models predict a continuous score, which is mapped back to the discrete rating scale by rounding and then snapping the non-existent 5 and 6 onto 4 and 7. This allows for classification and regression to be compared on the same accuracy footing.

## Results

### Classification (8 classes)

Test split / validation split:

| Model | Accuracy | Precision (macro) | Recall (macro) |
| :-- | :-: | :-: | :-: |
| Logistic Regression | $43.34$ / $43.46$ | $34.84$ / $34.71$ | $34.03$ / $34.25$ |
| **LinearSVC** | **$43.06$ / $43.62$** | $34.17$ / $33.83$ | $32.65$ / $33.29$ |
| Random Forest | $38.28$ / $38.52$ | $51.95$ / $53.35$ | $25.46$ / $25.82$ |

The two linear models are effectively tied, which is the expected outcome on high-dimensional sparse text. 

Random Forest is the outlier: far higher precision but recall near $25\%$, meaning it commits to a few ratings confidently and ignores the rest. Given that behaviour and its training cost, it was dropped from consideration.


### Regression

Test split / validation split:

| Model | MAE | MSE | R² | Accuracy after mapping |
| :-- | :-: | :-: | :-: | :-: |
| Lasso | $1.97$ / $1.94$ | $5.60$ / $5.48$ | 0.54 / 0.55 | 21.18 / 21.94 |
| **Ridge** | **1.62 / 1.61** | **4.12 / 4.02** | **0.66 / 0.67** | **27.02 / 27.88** |
| Gradient Boosting | 2.21 / 2.22 | 6.72 / 6.81 | 0.44 / 0.44 | 16.46 / 17.00 |

Ridge wins clearly. L2 regularization suits sparse text better than Lasso's feature selection — sentiment is carried by a weighted combination of many weak terms rather than a handful of strong ones. Gradient Boosting struggles with 30,000 sparse features at a tractable tree depth.

Regression scores lower on exact-match accuracy than classification, but its errors are better behaved: an MAE of 1.6 stars means the predictions land near the true rating even when they miss it, which classification accuracy doesn't reward.

### Clustering

K-Means over the same TF-IDF matrix, with clusters interpreted from the top terms of each centroid:

| k | What emerges |
| :-: | :-- |
| 2 | A clean positive / negative split (55.9% / 44.1%) — the dominant axis of variance is evaluative vocabulary |
| 4 | No thematic separation; still sentiment variants |
| 8 | Sentiment gives way to genre: comedy, TV series, horror — but 82% stays undifferentiated |
| 20 | Over-fragmented; several clusters hold under 0.1% of the data |
| **25** | Best structure: TV series, sci-fi, horror, family, animation, musicals, documentaries, war, book and video-game adaptations, and an evaluative "waste of time" cluster |

At k = 25 the model assigns meaningful topics to about 42% of the corpus, against 15% at k = 20 — raising k rescued nearly 30% of the data that had been scattered across uninterpretable groups.

Silhouette scores stay near zero at every k (0.0016 at k = 2, below 0.001 elsewhere). That is normal for high-dimensional sparse TF-IDF, where points are nearly equidistant: the clusters are interpretable by their top terms without being geometrically well separated. Silhouette is the wrong instrument here, and the top-term reading is what carries the analysis.

## Repository structure

```
.
├── A48630A51038A51811TP2.ipynb     # main notebook
├── data/
│   ├── imdbFull.p                  # raw dataset (66 MB)
│   └── imdbPreProcessed.p          # cleaned + stemmed dataset (58 MB)
├── drafts/
│   └── A48630A51038A51811TP2.ipynb # exploratory working notebook
├── src/
│   └── methods.py                  # metrics and cluster-inspection helpers
├── requirements.txt
└── README.md
```

`methods.py` holds the reusable pieces:

| Function | Purpose |
| :-- | :-- |
| `class_stats` | Accuracy, macro precision/recall and confusion matrix for a classifier |
| `reg_stats` | MAE, MSE, R², plus the continuous-to-class mapping and its accuracy |
| `cluster_top_words` | Top terms per K-Means centroid, for reading a cluster's topic |
| `cluster_metrics` | Sampled silhouette score and the distribution of reviews per cluster |

## Known limitations

- Silhouette scores near zero mean the clusters have no real geometric separation. The topic labels come from reading centroid terms, which is interpretation rather than measurement.
- Random Forest was tuned over a minimal grid; its grid search did not complete at full scale, so its comparison against the linear models is not entirely like-for-like.
- The 5 → 4 and 6 → 7 snapping in the regression mapping is a pragmatic choice, not a principled one; a properly ordinal model would handle the gap in the rating scale directly.
- Classification is evaluated on exact rating match, which treats a one-star miss the same as a nine-star miss. An ordinal metric would describe these models more fairly.

## Authors

Group project for Machine Learning (T52D), ISEL - DEI, 2025/26. Supervised by Prof. Gonçalo Xufre Silva.

- João Madeira (48630)
- Renata Góis (51038)
- Bruno Pereira (51811)
