# TransE Implementation & Reproduction

Code that directly implements the algorithm from *Translating Embeddings for Modeling Multi-relational Data* (NeurIPS, 2013)
in PyTorch and reproduces the performance of the Link prediction experiment

## Implementation Details

- **Initialization**: Initialize entity/relation embeddings with $Unif(-6/\sqrt{k}, 6/\sqrt{k})$, normalize relations right after initialization
- **Training loop**: Every epoch, entity normalize → mini-batch sampling → negative sampling (corrupted triplet) → margin-based ranking loss
- **Evaluation**: Link prediction (raw / filtered settings), Mean Rank, Hits@10

## Dataset

[FB15k](https://huggingface.co/datasets/VLyb/FB15k) — a standard KGE benchmark based on Freebase

### FB15k
A subset reduced to a scale suitable for experiments, out of the entire Freebase (approx. 1.2 billion triples, over 80 million entities)

- Only entities that also exist in the Wikilinks database are selected
- Only entities/relations that appear at least 100 times within Freebase are selected
- Duplicate relations that are just the reverse of another (head and tail swapped), like `!/people/person/nationality` being the reverse of `/people/person/nationality`, are removed

**Statistics**

| Item | Count |
|------|------|
| Triplets | 592,213 |
| Entities | 14,951 |
| Relationships | 1,345 |

## How to Run

### Run directly in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0hseunghwan/TransE/blob/main/TransE.ipynb)


### Run locally

1. Git Clone
```bash
   git clone https://github.com/0hseunghwan/TransE.git
   cd TransE
```

2. Install Dependencies
```bash
   pip install -r requirements.txt
```

3. Run the Notebook
```bash
   jupyter notebook TransE.ipynb
```
   ※ The FB15k dataset is automatically downloaded/loaded when the notebook runs (no separate setup needed)

## Key Hyperparameters

| Parameter | Paper | This Implementation |
|---|:---:|:---:|
| Embedding dimension $k$ | 50 | 50 |
| Margin $\gamma$ | 1.0 | 1.0 |
| Distance function | L1 | L1 |
| Learning rate | 0.01 | 0.001 |
| Optimizer | SGD | Adam |
| Batch size | - | 1,024 |
| Epoch | 1,000 | 200 |

> **Note on hyperparameter changes:**
> When training with the paper's settings (SGD, 1000 epochs), the training error kept decreasing continuously even up to 1000 epochs, which was judged to be an underfitting state
> After changing the optimizer to Adam and reducing the number of epochs to 200 and retraining, convergence was reached much faster, and the performance evaluated at this state was similar to the performance reported in the paper

## Reproduction Results

| Setting | Metric | Paper result | Reproduction result |
|---|---|:---:|:---:|
| Raw | Mean Rank | 243 | 227.8 |
| Filtered | Mean Rank | 125 | 110.6 |
| Raw | Hits@10 (%) | 34.9 | 37.1 |
| Filtered | Hits@10 (%) | 47.1 | 49.4 |

> This is the result from Table 3 (Link prediction results) of the paper (Bordes et al., 2013), and this implementation shows similar performance

## References
[Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. *NeurIPS*.](https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
