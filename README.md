# TransE from Scratch

*Translating Embeddings for Modeling Multi-relational Data* (NeurIPS, 2013)
의 알고리즘을 PyTorch로 직접 구현한 코드입니다.

## 구현 내용

- **초기화**: $Unif(-6/\sqrt{k}, 6/\sqrt{k})$ 로 entity/relation 임베딩 초기화, relation은 초기화 직후 정규화
- **학습 루프**: 매 epoch마다 entity normalize → mini-batch 샘플링 → negative sampling (corrupted triplet) → margin-based ranking loss
- **평가**: Link prediction (raw / filtered 두 세팅), Mean Rank, MRR, Hits@10

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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0hSeunghwan/TransE_implementation/blob/main/TransE.ipynb)


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
|---|---|---|
| Embedding dimension $k$ | 50 | 50 |
| Margin $\gamma$ | 1.0 | 1.0 |
| Distance function | L1 | L1 |
| Learning rate | 0.01 | 0.01 |
| Optimizer | SGD | Adam |
| Batch size | - | 1024 |
| Epoch | 1000 | 200|

## 참고문헌

Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013).
Translating Embeddings for Modeling Multi-relational Data. *NeurIPS*.
