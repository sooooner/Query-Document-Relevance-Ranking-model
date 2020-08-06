#-*- coding:utf-8 -*-
import numpy as np

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





