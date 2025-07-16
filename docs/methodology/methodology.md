# Methodology Documentation

## Overview

This document describes the methodology used in our research on "Enhancing Sentiment-driven Recommender Systems with LLM-Based Feature Engineering: A Case Study in Drug Review Analysis."

## Research Objectives

1. **Primary Objective**: Develop an enhanced recommender system that incorporates sentiment analysis features derived from LLM-based feature engineering
2. **Secondary Objectives**:
   - Compare different sentiment analysis approaches (InferSent, LLM2Vec, FastText, GloVe)
   - Evaluate the impact of sentiment features on recommendation quality
   - Analyze the interpretability of sentiment-aware recommendations

## Dataset

### Drug Review Dataset
- **Source**: Drugs.com
- **Size**: ~107MB total (train + test)
- **Format**: TSV files
- **Features**:
  - `drugName`: Name of the drug
  - `condition`: Medical condition being treated
  - `review`: User review text
  - `rating`: User rating (1-10)
  - `date`: Review date
  - `usefulCount`: Number of helpful votes

### Data Preprocessing
1. **Text Cleaning**: Remove special characters, normalize text
2. **Tokenization**: Split text into tokens
3. **Lemmatization**: Reduce words to their base form
4. **Stop Word Removal**: Remove common words
5. **Feature Engineering**: Extract text length, word count, date features

## Methodology

### Phase 1: Sentiment Analysis

#### 1.1 InferSent Approach
- **Model**: InferSent sentence embeddings
- **Architecture**: Bi-directional LSTM with max pooling
- **Features**: 4096-dimensional sentence embeddings
- **Training**: Fine-tuned on drug review sentiment classification

#### 1.2 LLM2Vec Approach
- **Model**: Large Language Model based embeddings
- **Architecture**: Transformer-based sentence embeddings
- **Features**: 768-dimensional contextual embeddings
- **Training**: Pre-trained on large text corpus, fine-tuned on domain data

#### 1.3 FastText Approach
- **Model**: FastText for text classification
- **Architecture**: Character n-gram based model
- **Features**: 300-dimensional word vectors
- **Training**: Trained from scratch on drug reviews

#### 1.4 GloVe Approach
- **Model**: Global Vectors for Word Representation
- **Architecture**: Pre-trained word embeddings
- **Features**: 300-dimensional word vectors
- **Training**: Pre-trained on large corpus, adapted for sentiment

### Phase 2: Feature Engineering

#### 2.1 LLM-Based Feature Extraction
- **Contextual Embeddings**: Extract contextual features from LLM outputs
- **Sentiment Scores**: Generate sentiment probability scores
- **Attention Weights**: Extract attention patterns for interpretability
- **Domain Adaptation**: Adapt features to drug review domain

#### 2.2 Multi-Modal Feature Fusion
- **Text Features**: Sentiment embeddings, TF-IDF features
- **Numerical Features**: Rating, date, interaction counts
- **Categorical Features**: Drug names, conditions (encoded)
- **Fusion Strategy**: Weighted combination of feature types

### Phase 3: Recommender System

#### 3.1 Content-Based Filtering
- **Algorithm**: Cosine similarity
- **Features**: Drug descriptions, condition information, sentiment features
- **Enhancement**: Sentiment scores are integrated into the similarity calculation

## Evaluation Metrics

### Sentiment Analysis Evaluation
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for positive sentiment class
- **Recall**: Recall for positive sentiment class
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve


### Recommender System Evaluation

Performance comparison of different ablation settings was conducted on five conditions under two relevance thresholds (θ = 5 and θ = 8). The following metrics are reported:

- **Hit@10**: Proportion of cases where at least one relevant item appears in the top-10 recommendations.
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10, measuring ranking quality.
- **Mean Rating (Top-10)**: Average rating of the top-10 recommended items.





## Implementation Details

### Software Stack
- **Python**: 3.8+
- **Deep Learning**: PyTorch, Transformers
- **ML**: Scikit-learn, Surprise
- **NLP**: NLTK, spaCy, Gensim
- **Visualization**: Matplotlib, Seaborn, Plotly

### Hardware Requirements
- **GPU**: CUDA-compatible GPU (recommended)
- **RAM**: 16GB+ (recommended)
- **Storage**: 50GB+ for models and data

### Reproducibility
- **Random Seeds**: Fixed across all experiments
- **Environment**: Docker container with exact dependencies
- **Version Control**: Git with tagged releases
- **Experiment Tracking**: MLflow for experiment management

## Ethical Considerations

### Data Privacy
- **Anonymization**: User identifiers removed
- **Consent**: Data used in accordance with terms of service
- **Security**: Secure storage and transmission protocols

### Bias and Fairness
- **Bias Detection**: Analyze for demographic bias
- **Fairness Metrics**: Evaluate across different user groups
- **Transparency**: Document potential biases and limitations

### Medical Disclaimer
- **Not Medical Advice**: Recommendations are not medical advice
- **Professional Consultation**: Users should consult healthcare professionals
- **Limitations**: Acknowledge limitations of automated systems

## Future Work

### Extensions
1. **Multi-modal Features**: Incorporate drug images and molecular structures
2. **Temporal Dynamics**: Model temporal patterns in drug reviews
3. **Personalization**: User-specific sentiment modeling
4. **Interpretability**: Explainable AI for recommendation decisions

### Applications
1. **Clinical Decision Support**: Assist healthcare professionals
2. **Patient Education**: Improve patient understanding
3. **Drug Safety**: Early detection of adverse effects
4. **Market Analysis**: Pharmaceutical industry insights

## References

1. Conneau, A., et al. (2017). "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data"
2. Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
3. Joulin, A., et al. (2016). "FastText.zip: Compressing text classification models"
4. Pennington, J., et al. (2014). "GloVe: Global Vectors for Word Representation"
5. Koren, Y., et al. (2009). "Matrix Factorization Techniques for Recommender Systems" 