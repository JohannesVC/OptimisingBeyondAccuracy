# Optimising Beyond Accuracy 
## Tuning for Diversity and Novelty in Attention-based News Recommenders

This project implements and evaluates neural recommendation systems for news, with a focus on **beyond-accuracy metrics** such as *diversity* and *novelty*. Using the MIND dataset, it compares traditional and attention-based models, incorporating pre-trained language models (GloVe and BERT), and applies custom regularisation layers to encourage diverse and novel recommendations.

## Key Features

- Custom-built bi-encoder architectures
- Integration of GloVe and fine-tuned BERT embeddings
- Training pipeline with hard-negative sampling
- Regularisation for Intra-List Diversity and Novelty
- Evaluation on both accuracy and beyond-accuracy metrics
- Statistical validation with t-tests and visualisations


## Evaluation Metrics

| Type              | Metrics                             |
|-------------------|--------------------------------------|
| Accuracy          | AUC, MRR, nDCG@5, nDCG@10            |
| Beyond-Accuracy   | ILD, Surprisal, EPC, coverage      |

## Installation Instructions

### 1. Clone the repository

```bash
git clone https://github.com/johannesvc/OptimisingBeyondAccuracy.git
cd OptimisingBeyondAccuracy
```

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the MIND-small and/or MIND-large datasets from [msnews.github.io](https://msnews.github.io/), and place them in the `.data/` directory.

### 4. Recreate Embedding Layers

To ensure reproducibility:

- Use the `embeddings.ipynb` to:
  - Build the GloVe lookup matrix
  - Fine-tune the BERT embeddings (`bert-base-uncased`) on titles
- Save the results to `.npy` format in the `.data/` directory

## Run Notebooks

All models are run from notebooks. Simply follow the steps and press *run*.

## Results and Evaluation

After training, results (metrics and hyperparameters) are saved to `results.json`. Evaluation scripts are provided in the `evaluation.ipynb` notebook. These include:

- Accuracy vs diversity trade-offs
- Metric comparisons across model types
- All plots used in the main report

---

## Project Structure

```bash
.
├── baselines.ipynb     # Main baselines script
├── bi-bert.ipynb       # Documented training script using GloVe embeddings
├── bi-glove.ipynb      # Documented training script using BERT embeddings
├── bi-encoder.ipynb    # Main training script for hyperparameter tuning
├── CB.ipynb            # Content-based filtering baseline script
├── EDA.ipynb           # Exporatory Data Analysis and plotting
├── embeddings.ipynb    # Fine-tuning BERT + saving embeddings
├── evaluation.ipynb    # Evaluation scripts and plotting
├── README.md           # This file
├── model.png           # Diagram of model
├── recs
│   ├── bi_encoder.py   # Main training script
│   ├── data_loader.py  # Preprocessing and sequence generation
│   ├── __init__.py
│   ├── metrics.py      # Evaluation and plotting
│   ├── subclasses.py   # Model architectures and custom layers
│   └── utils.py        # Utils for logging
├── requirements.txt
└── results.json        # Output results for all models
```

---

## License

MIT License.

---

## Acknowledgements

- [MIND dataset](https://msnews.github.io/) by Microsoft Research
- GloVe embeddings by Stanford NLP
- BERT models via HuggingFace
- Inspired by Microsoft Recommenders and Transformers4Rec (Nvidia)
