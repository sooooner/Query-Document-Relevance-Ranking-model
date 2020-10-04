# Query-Document-Relevence-Ranking-model

Kaggle [Crowdflower Search Results Relevance data](https://www.kaggle.com/c/crowdflower-search-relevance)를 이용한 E-Commerce 사용자 검색 시스템 랭킹 모델

## Description
프로젝트 과정 설명(링크는 **Medium**글) 

**Description.ipynb**
+ DRMM, PACRR, DRMM_PACRR  요약 및 구현 설명
+ [Query-Document Relevance Ranking model(DRMM)](https://medium.com/@tnsgh0101/query-document-relevence-ranking-model-596c8571b84)
+ [Query-Document Relevance Ranking model(PACRR, PACRR-DRMM)](https://medium.com/@tnsgh0101/query-document-relevence-ranking-model-2-b50af71b2ca7)

**Description_bert.ipynb**
+ CEDR 요약 및 구현 설명
+ [Query-Document Relevance Ranking model(CEDR)](https://medium.com/@tnsgh0101/query-document-relevance-ranking-model-3-9305028cf44)

## model implement

**DRMM, PACRR, PACRR_DRMM.ipynb**
+ 각 모델 구현 과정 

**TOTAL, TOTAL-bert.ipynb**
+ DRMM, PACRR, PACRR_DRMM 모델들 비교 
+ CEDR 모델 비교

**BERT, BERT_functional.ipynb**
+ keras functional API를 통한 bert 모델링, 미세 조정
  

## data processing
**data processing.ipynb**
+ DRMM, PACRR, DRMM_PACRR 모델을 위한 data preprocessing

**data processing for bert.ipynb**
+ CEDR 모델을 위한 data preprocessing

## utility
모델에 사용된 함수  
utility/[README.md](https://github.com/sooooner/Query-Document-Relevance-Ranking-model/blob/master/utility/README.md) 참고

## model
모델 구현에 필요한 함수  
model/[README.md](https://github.com/sooooner/Query-Document-Relevance-Ranking-model/blob/master/model/README.md) 참고

## Usage
1. BERT.ipynb로 미세조정을 합니다.
2. data processing.ipynb 와 data processing for bert.ipynb를 실행하여 preprocessed data를 생성한 뒤 main.py로 모델을 훈련 합니다. 
3. 아래 커맨드로 모델을 훈련시킵니다.

```
python main.py --model=drmm --bert=True --lr=0.1 --batch=256 --epoch=100
```
--model : 모델 선택(drmm, pacrr, pacrr_drmm)  
--bert : CEDR 여부 선택  
--lr : learning rate  
--batch : batch size  
--epoch : number of epoch  

## 참고자료 
[A Deep Relevance Matching Model for Ad-hoc Retrieval](https://arxiv.org/abs/1711.08611)  
[PACRR: A Position-Aware Neural IR Model for Relevance Matching](https://arxiv.org/abs/1704.03940)  
[Deep Relevance Ranking Using Enhanced Document-Query Interactions](https://arxiv.org/abs/1809.01682)  
[CEDR: Contextualized Embeddings for Document Ranking](https://arxiv.org/abs/1904.07094)  
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1608.03983)  
[SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)  









