#-*- coding:utf-8 -*-
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

def ndcg(rel_pred, p=None, form="linear"):
    if p==None:
        p = len(rel_pred)
    if p > len(rel_pred):
        rel_pred = np.append(rel_pred, [0]*(p - len(rel_pred)))
    
    rel_true = np.sort(rel_pred)[::-1]
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")
    
    return dcg / idcg

def mAP_score(rel_pred, p=None):
    if p:
        rel_pred = rel_pred[:p]
    if sum(rel_pred) == 0:
        return 1
    count = 0
    precision_list = []
    for i, rel in enumerate(rel_pred):
        if rel == 1:
            count +=1
        precision = count/(i+1)

        if rel == 1:
            precision_list.append(precision)
            
        recall = count/sum(rel_pred)
        if recall == 1:
            break
    return np.mean(precision_list)

def generate_pairwise_dataset(df):
    columns = ['query', 
               'product_title_P',
               'product_title_N',
               'median_relevance_P',
               'median_relevance_N',
               'drmm_hist_P', 
               'drmm_hist_N', 
               'sim_matrix_P',
               'sim_matrix_N',
               'query_idf_P',
               'idf_softmax_P']
    
    new_df = pd.DataFrame(columns=columns)
    for query in tqdm(df['query'].unique()):
        # 만족도 (4 - 3, 2, 1), (3 - 2, 1), (2 - 1) 6개 쌍으로 진행
        for positive in [4, 3, 2]:
            try:
                P_temp = df[df['query']==query].groupby('median_relevance').get_group(positive)
                for negative in range(positive)[:0:-1]:
                    try:
                        N_temp = df[df['query']==query].groupby('median_relevance').get_group(negative)
                        temp = pd.merge(P_temp, N_temp, how='inner', on='query',  suffixes=('_P', '_N'))[columns]
                        new_df = pd.concat((new_df, temp))
                    except:
                        # 만족도가 없는구간 pass
                        pass
            except:
                # 만족도가 없는구간 pass
                pass
    new_df.rename(columns={'query_idf_P':'query_idf', 'idf_softmax_P':'idf_softmax'}, inplace=True)
    return new_df

def history_plot(model_history, model_metric, batch_size, df, save=False):
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    xx = [i*(math.ceil(len(df)/(batch_size))) for i in range(len(model_history.history['val_loss']))]

    loss_ax.plot(model_history.history['loss'], 'y', label='train loss')
    loss_ax.plot(xx, model_history.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(xx, model_metric.history['val_nDCG'], 'b', label='val nDCG')
    acc_ax.plot(xx, model_metric.history['val_mAP'], 'g', label='val mAP')

    loss_ax.set_xlabel('batch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('metric')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    if save:
        plt.savefig('fig1.png')
    plt.show()


def highlight(s):
    if s.median_relevance == 1:
        return ['background-color: #cb4146']*6
    elif s.median_relevance == 2:
        return ['background-color: #be7c80']*6
    elif s.median_relevance == 3:
        return ['background-color: #9badbd']*6
    else:
        return ['background-color: #0ddbff']*6