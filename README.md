# TransE Implementation & Reproduction

*Translating Embeddings for Modeling Multi-relational Data* (NeurIPS, 2013)
의 알고리즘을 PyTorch로 직접 구현하고, Link Prediction 실험의 성능을 재현한 코드

## 구현 내용

- **초기화**: $Unif(-6/\sqrt{k}, 6/\sqrt{k})$ 로 entity/relation 임베딩 초기화, relation은 초기화 직후 정규화
- **학습 루프**: 매 epoch마다 entity normalize → mini-batch 샘플링 → negative sampling (corrupted triplet) → margin-based ranking loss
- **평가**: Link prediction (raw / filtered 두 세팅), Mean Rank, Hits@10

## 데이터셋

[FB15k](https://huggingface.co/datasets/VLyb/FB15k) — Freebase 기반 표준 KGE 벤치마크.

### FB15k
Freebase 전체(약 12억 개의 트리플, 8천만 개 이상의 엔티티) 중, 실험에 적합한 규모로 축소한 서브셋

- Wikilinks 데이터베이스에도 존재하는 엔티티만 선택
- Freebase 내에서 최소 100회 이상 등장하는 엔티티/관계만 선택
- `/people/person/nationality`의 역방향 관계인 `!/people/person/nationality`처럼, head와 tail만 뒤바꾼 중복 관계는 제거

**통계**

| 항목 | 개수 |
|------|------|
| Triplets | 592,213 |
| Entities | 14,951 |
| Relationships | 1,345 |

## 실행 방법

### Colab에서 바로 실행
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0hseunghwan/TransE_implementation/blob/main/TransE.ipynb)


### 로컬에서 실행

1. Git Clone
```bash
   git clone https://github.com/0hseunghwan/TransE_implementation.git
   cd TransE_implementation
```

2. Install Dependencies
```bash
   pip install -r requirements.txt
```

3. Run the Notebook
```bash
   jupyter notebook TransE.ipynb
```
   ※ 노트북 실행 시 FB15k 데이터셋을 자동으로 다운로드/로드 (별도 준비 불필요)

## 주요 하이퍼파라미터

| Parameter | Paper | This Implementation |
|---|:---:|:---:|
| Embedding dimension $k$ | 50 | 50 |
| Margin $\gamma$ | 1.0 | 1.0 |
| Distance function | L1 | L1 |
| Learning rate | 0.01 | 0.001 |
| Optimizer | SGD | Adam |
| Batch size | - | 1024 |
| Epoch | 1000 | 200 |

> **Note on hyperparameter changes:**
> 논문 세팅(SGD, 1000 epoch)으로 학습했을 때 training error가 1000 epoch까지도 지속적으로 감소하는 것을 확인했고, 이는 underfitting 상태로 판단
> 이에 optimizer를 Adam으로 변경하고 epoch 수를 200으로 줄여 재학습한 결과, 더 빠르게 충분한 수렴에 도달했으며, 이 상태에서 평가한 성능이 논문에서 보고된 성능과 유사 (자세한 내용은 [Results](##Reproduction 결과) 참고).

## Reproduction 결과

| Setting | Metric | Paper result | Reproduction result |
|---|---|:---:|:---:|
| Raw | Mean Rank | 243 | 231.2 |
| Raw | Hits@10 (%) | 34.9 | 36.9 |
| Filtered | Mean Rank | 125 | 113.8 |
| Filtered | Hits@10 (%) | 47.1 | 49.4 |

> 논문(Bordes et al., 2013) Table 3 (Link prediction results) 결과이며, 본 구현은 이와 유사한 성능 보임


## 참고문헌

Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013).
Translating Embeddings for Modeling Multi-relational Data. *NeurIPS*.
