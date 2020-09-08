import random
import numpy as np
import pandas as pd
import tensorflow as tf

from ast import literal_eval
from model.loss import Pairwise_ranking_loss
from model.drmm import Gen_DRMM_Model
from model.pacrr import Gen_PACRR_Model
from model.pacrr_drmm import Gen_PACRR_DRMM_Model
from model.callback import LossHistory, _metric
from utility.utility import generate_pairwise_dataset

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, type=str, help="drmm, pacrr, pacrr_drmm")
ap.add_argument("--bert", required=False, type=bool, help="bert")
ap.add_argument("--lr", required=False, type=float, help="learning rate")
ap.add_argument("--batch", required=False, type=float, help="batch size")
ap.add_argument("--epoch", required=False, type=int, help="total epoch count")
args = ap.parse_args()

def load_data(bert=False):
    if not bert:
        df = pd.read_csv('./data/paccr_drmm_all.csv', converters={"positive_hist"      : literal_eval,
                                                                  "negative_hist"      : literal_eval,
                                                                  "query_idf"          : literal_eval,
                                                                  "negative_sim_matrix": literal_eval,
                                                                  "positive_sim_matrix": literal_eval,
                                                                  "idf_softmax"        : literal_eval})


        df = df[['query_preprocessed', 'positive_hist', 'negative_hist', 'query_idf', 'negative_sim_matrix', 'positive_sim_matrix', 'idf_softmax']]
        
        test = pd.read_csv('./data/paccr_drmm_all_test.csv', converters={"hist"       : literal_eval,
                                                                         "query_idf"  : literal_eval,
                                                                         "sim_matrix" : literal_eval,
                                                                         "idf_softmax": literal_eval})
                                                                     
        test['binary_relevance'] = test['median_relevance'].apply(lambda x: 0 if x <= 2 else 1) 
        dev_q = set(random.sample(list(df['query_preprocessed'].unique()), 40))
        train_q = set(df['query_preprocessed'].unique()) - dev_q
        train = df[df['query_preprocessed'].isin(train_q)]
        dev = df[df['query_preprocessed'].isin(dev_q)]
        
        metadata = {'negative_sim_matrix': tf.constant(train['negative_sim_matrix'].tolist(), dtype=tf.float32), 
                    'positive_sim_matrix': tf.constant(train['positive_sim_matrix'].tolist(), dtype=tf.float32),
                    'idf_softmax'        : tf.constant(train['idf_softmax'].tolist(), dtype=tf.float32),
                    'query_idf'          : tf.ragged.constant(train['query_idf'], dtype=tf.float32, ragged_rank=1),
                    'positive_hist'      : tf.ragged.constant(train['positive_hist'], dtype=tf.float32, ragged_rank=1),
                    'negative_hist'      : tf.ragged.constant(train['negative_hist'], dtype=tf.float32, ragged_rank=1)}
                
        metadata_dev = {'negative_sim_matrix': tf.constant(dev['negative_sim_matrix'].tolist(), dtype=tf.float32),
                        'positive_sim_matrix': tf.constant(dev['positive_sim_matrix'].tolist(), dtype=tf.float32),
                        'idf_softmax'        : tf.constant(dev['idf_softmax'].tolist(), dtype=tf.float32),
                        'query_idf'          : tf.ragged.constant(dev['query_idf'], dtype=tf.float32, ragged_rank=1),
                        'positive_hist'      : tf.ragged.constant(dev['positive_hist'], dtype=tf.float32, ragged_rank=1),
                        'negative_hist'      : tf.ragged.constant(dev['negative_hist'], dtype=tf.float32, ragged_rank=1)}
        
        
    else:
        test = pd.read_csv('./data/paccr_drmm_bert_test_all.csv', converters={"query_idf"          : literal_eval,
                                                                              "idf_softmax"        : literal_eval,
                                                                              "sim_matrix"         : literal_eval,
                                                                              "query_token"        : literal_eval,
                                                                              "product_title_token": literal_eval,
                                                                              "token_ids"          : literal_eval,
                                                                              "drmm_hist"          : literal_eval,
                                                                              'token'              : literal_eval})
        
        df = generate_pairwise_dataset(test)
        df.reset_index(inplace=True, drop=True)
        
        dev_q = set(random.sample(list(df['query'].unique()), 40))
        train_q = set(df['query'].unique()) - dev_q
    
        train = df[df['query'].isin(train_q)]
        dev = df[df['query'].isin(dev_q)]
        test = test[test['query'].isin(dev_q)]
        test.rename(columns={'drmm_hist':'hist'}, inplace=True)
    
        metadata = {'negative_sim_matrix': tf.constant(train['sim_matrix_N'].tolist(), dtype=tf.float32, name='negative_sim_matrix'), 
                    'positive_sim_matrix': tf.constant(train['sim_matrix_P'].tolist(), dtype=tf.float32, name='positive_sim_matrix'),
                    'idf_softmax'        : tf.constant(train['idf_softmax'].tolist(), dtype=tf.float32, name='idf_softmax'),
                    'query_idf'          : tf.ragged.constant(train['query_idf'], dtype=tf.float32, ragged_rank=1, name='query_idf'),
                    'positive_hist'      : tf.ragged.constant(train['drmm_hist_P'], dtype=tf.float32, ragged_rank=1, name='positive_hist'),
                    'negative_hist'      : tf.ragged.constant(train['drmm_hist_N'], dtype=tf.float32, ragged_rank=1, name='negative_hist')}
                    
        metadata_dev = {'negative_sim_matrix': tf.constant(dev['sim_matrix_N'].tolist(), dtype=tf.float32, name='negative_sim_matrix'),
                        'positive_sim_matrix': tf.constant(dev['sim_matrix_P'].tolist(), dtype=tf.float32, name='positive_sim_matrix'),
                        'idf_softmax'        : tf.constant(dev['idf_softmax'].tolist(), dtype=tf.float32, name='idf_softmax'),
                        'query_idf'          : tf.ragged.constant(dev['query_idf'], dtype=tf.float32, ragged_rank=1, name='query_idf'),
                        'positive_hist'      : tf.ragged.constant(dev['drmm_hist_P'], dtype=tf.float32, ragged_rank=1, name='positive_hist'),
                        'negative_hist'      : tf.ragged.constant(dev['drmm_hist_N'], dtype=tf.float32, ragged_rank=1, name='negative_hist')}
    
    return metadata, metadata_dev, train, dev, test
    
if __name__ == "__main__":
    
    if args.bert:
        firstk = 13
        lq = 8
    else:
        firstk = 8
        lq = 6
    lg = 5
    nf = 32
    ns = 2

    if args.model == 'drmm':
        model = Gen_DRMM_Model(bert=args.bert)
    elif args.model == 'pacrr':
        model = Gen_PACRR_Model(firstk, lq, lg, nf, ns, bert=args.bert)
    elif args.model == 'pacrr_drmm':
        model = Gen_PACRR_DRMM_Model(firstk, lq, lg, nf, ns, bert=args.bert)
        
    batch_size = 128
    if args.batch:
        batch_size = args.batch
    
    total_epoch_count = 100
    if args.epoch:
        total_epoch_count = args.epoch
    
    lr = 0.1
    if args.lr:
        lr = args.lr

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=args.lr), loss=Pairwise_ranking_loss)
    metadata, metadata_dev, train, dev, test = load_data(bert=args.bert)
    model_history = LossHistory()
    model_metric = _metric(test)

    model.fit(x=metadata, 
              y=tf.constant([0.]*len(train)),
              validation_data=(metadata_dev, tf.constant([0.]*len(dev))),
              shuffle=True,
              epochs=total_epoch_count,
              batch_size=batch_size,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                         model_history,
                         model_metric])
    
    history_plot(model_history, model_metric, batch_size, df=train, save=False)
    
    
    
    
    
    
    
    
    
    
