# 🚀 Enhancing Sentiment-driven Recommender Systems with LLM-Based Feature Engineering: A Case Study in Drug Review Analysis 💊


## 📄 Paper

Find the paper associated with this project here: [Link to the paper](https://doi.org/10.1109/ACCESS.2025.3590326)


## Context
This project aims to explore and improve recommendation systems by leveraging recent advances in Large Language Models (LLM) for feature engineering, applied to sentiment analysis in drug reviews, with a particular focus on specifically negative opinions. The goal is to demonstrate how integrating advanced language representations can enrich the understanding of user opinions (especially negative feedback) and improve recommendation relevance.

## 🎯 Objectives
- 🧠 Extract rich semantic features from drug reviews using LLMs (e.g., Llama, InferSent, FastText, GloVe).
- 🔍 Compare different feature extraction and sentiment analysis methods.
- 🤖 Develop and evaluate a recommendation engine based on these new representations.

## 🗂️Project Structure
```
├── data/                        # Raw datasets and resources
├── experiments/                 # Experiment configurations
│   └── configs/
├── notebooks/                   # Jupyter notebooks for analysis and development
│   ├── exploratory_analysis/    # Exploratory analyses
│   ├── model_development/      # Model development
│   │   ├── 01 Sentences Embeddings/
│   │   ├── 02 Sentiment Analysis/
│   │   ├── 03 Feature Engineering/
│   │   └── 04 Recommender Engine/
├── requirements.txt             # Python dependencies
```

## Installation
1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```
2. **Create a virtual environment (optional but recommended)**
```bash
python3 -m venv venv
source venv/bin/activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage
- The Jupyter notebooks in `notebooks/` cover the entire pipeline:
  - Exploratory analysis
  - Feature extraction (embeddings)
  - Sentiment analysis
  - Advanced feature engineering
  - Recommendation engine development and evaluation
- Adapt file paths as needed according to your environment.

## Cite this paper
@ARTICLE{11083619,
  author={Kangoni, Samuel Matia and Tshipata, Obed Tshimanga and Nzakuna, Pierre Sedi and Paciello, Vincenzo and Mboma, Jean Gilbert Mbula and Makulo, Jean Robert and Kyamakya, Kyandoghere},
  journal={IEEE Access}, 
  title={Enhancing Sentiment-driven Recommender Systems with LLM-Based Feature Engineering: A Case Study in Drug review Analysis}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Reviews;Drugs;Sentiment analysis;Recommender systems;Feature extraction;Analytical models;Accuracy;Measurement;Training;Standards;Cosine similarity ranking;Drug recommendation systems;Large Language Models;Real-world feedback in healthcare;Sentiment analysis},
  doi={10.1109/ACCESS.2025.3590326}}



## References
- [1] Embedding repositories used in this project:
    - [llm2vec (McGill-NLP)](https://github.com/McGill-NLP/llm2vec)
    - [InferSent (Facebook Research)](https://github.com/facebookresearch/InferSent)
    - [fastText (Facebook Research)](https://github.com/facebookresearch/fastText)
    - [GloVe (Stanford NLP)](https://github.com/stanfordnlp/GloVe)
- [2] Methodological documentation in `docs/methodology/`

---
For contributions or to report issues, please contact me at samuel.matia@unikin.ac.cd.
