# Query-Document-Relevence-Ranking-model

Kaggle [Crowdflower Search Results Relevance data](https://www.kaggle.com/c/crowdflower-search-relevance)를 이용한 E-Commerce 사용자 검색 시스템 랭킹 모델

## Description
DRMM, PACRR, DRMM-PACRR 

**Medium** 
+ [Query-Document Relevance Ranking model](https://medium.com/@tnsgh0101/query-document-relevence-ranking-model-596c8571b84)
+ [Query-Document Relevance Ranking model(2)](https://medium.com/@tnsgh0101/query-document-relevence-ranking-model-2-b50af71b2ca7)


## utility
모델에 사용된 함수

## model
keras model

## Usage
```
python main.py --model=drmm --bert=True --lr=0.1 --batch=256 --epoch=100
```

## 참고자료 
[A Deep Relevance Matching Model for Ad-hoc Retrieval](https://arxiv.org/abs/1711.08611)
[PACRR: A Position-Aware Neural IR Model for Relevance Matching](https://arxiv.org/abs/1704.03940)
[Deep Relevance Ranking Using Enhanced Document-Query Interactions](https://arxiv.org/abs/1809.01682)
[CEDR: Contextualized Embeddings for Document Ranking](https://arxiv.org/abs/1904.07094)













