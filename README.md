# Tweet Sentiment Classification

A natural language processing project for binary sentiment classification of tweets, progressing from a traditional bag-of-words baseline to an optimised sentiment-aware machine-learning pipeline.

The project explores feature extraction, supervised classification, cross-validation, systematic error analysis and iterative feature engineering to improve sentiment prediction on informal social-media text.

## Project Overview

The work begins with a bag-of-words representation and Linear Support Vector Machine classifier. Performance is evaluated using 10-fold cross-validation before misclassified tweets are analysed to identify weaknesses in the baseline approach.

The model is then progressively improved through text preprocessing, contextual n-gram features, sentiment-aware features and classifier tuning.

## Pipeline

### Baseline Classification

The initial system represents tweets using bag-of-words features and trains a LinearSVC classifier for binary sentiment prediction.

### Cross-Validation

The classifier is evaluated using 10-fold cross-validation. Precision, recall, F1 score and accuracy are calculated across folds to provide a more reliable estimate of model performance.

### Error Analysis

A confusion-matrix-based error analysis investigates false positives and false negatives. The analysis highlights challenges including sarcasm, slang, implicit sentiment, political language and sentiment that depends on conversational context rather than individual words.

### Feature Engineering

The optimisation stage experiments with a range of NLP features and preprocessing strategies, including:

- improved text cleaning and tokenisation
- bigrams and trigrams
- negation handling
- intensifier features
- VADER sentiment scores
- lexicon-based sentiment features
- polarity ratios
- token and sentence-level statistical features
- class balancing and regularisation tuning

### Model Optimisation

Multiple modelling and feature configurations are evaluated experimentally. LinearSVC is retained and tuned using regularisation, balanced class weighting and extended optimisation iterations.

## Results

The baseline achieved an F1 score of **0.8697** during development.

After feature engineering and model optimisation, development F1 increased to **0.9043**.

The final held-out test achieved an F1 score of **0.8761**, compared with a test baseline of **0.8338**.

The experiments suggest that sentiment-aware features provided more useful improvements than many purely structural text features.

## Repository Structure

- `01_baseline_and_error_analysis.ipynb` - feature extraction, baseline classifier, cross-validation and error analysis
- `02_feature_engineering_and_optimisation.ipynb` - feature engineering, model experimentation, optimisation and final evaluation
- `requirements.txt` - Python dependencies

## Technologies

- Python
- scikit-learn
- NLTK
- NumPy
- pandas
- Matplotlib
- Linear Support Vector Machines
- VADER sentiment analysis

## Installation

```bash
pip install -r requirements.txt
```

## Concepts Demonstrated

- Natural language processing
- Sentiment analysis
- Text preprocessing and tokenisation
- Bag-of-words and n-gram representations
- Feature engineering
- Support Vector Machines
- Cross-validation
- Precision, recall, F1 and accuracy evaluation
- Confusion-matrix analysis
- Error analysis
- Hyperparameter tuning
- Model generalisation and overfitting analysis

## Key Findings

Error analysis showed that simple token-based models struggle particularly with sarcasm, slang, implicit emotion and context-dependent language. Sentiment-aware features such as VADER and lexicon polarity measures produced some of the strongest improvements, suggesting that modelling contextual sentiment signals is more useful than simply adding additional surface-level text features.
