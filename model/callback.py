import tensorflow as tf
from utility.utility import mAP_score, ndcg
from model.loss import Pairwise_ranking_loss

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, val):
        self.val = val
    
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[]}
        pred = self.model.predict(self.val)
        val_loss = Pairwise_ranking_loss(y_true=None, y_pred=pred)
        self.history['val_loss'].append(val_loss)
        
    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.history['val_loss'].append(logs.get('val_loss'))
        
class _metric(tf.keras.callbacks.Callback):
    def __init__(self, test):
        self.test = test
        self.query = test['query'].unique()
        self.history = {'val_nDCG':[],'val_mAP':[]}
        
    def metric_calculator(self):
        ndcg_sum = 0
        mAP_sum = 0
        for q in self.query:
            ndcg_test = self.test[self.test['query'] == q]
            metadata_for_ndcg = {'negative_sim_matrix': tf.constant(ndcg_test['sim_matrix'].tolist(), dtype=tf.float32, name='negative_sim_matrix'),
                                 'positive_sim_matrix': tf.constant(ndcg_test['sim_matrix'].tolist(), dtype=tf.float32, name='positive_sim_matrix'),
                                 'idf_softmax': tf.constant(ndcg_test['idf_softmax'].tolist(), dtype=tf.float32, name='idf_softmax'),
                                 'query_idf': tf.ragged.constant(ndcg_test['query_idf'], dtype=tf.float32, ragged_rank=1, name='query_idf'),
                                 'positive_hist': tf.ragged.constant(ndcg_test['hist'], dtype=tf.float32, ragged_rank=1, name='positive_hist'),
                                 'negative_hist': tf.ragged.constant(ndcg_test['hist'], dtype=tf.float32, ragged_rank=1, name='negative_hist')}
            ndcg_test['rel'] = self.model.predict(metadata_for_ndcg)[:len(ndcg_test)]
            ndcg_test.sort_values(by=['rel'], axis=0, ascending=False, inplace=True)
            
            rel_pred = list(ndcg_test['median_relevance']-1)
            rel_true = ndcg_test['binary_relevance']
            
            mAP_sum += mAP_score(rel_true, p=20)
            ndcg_sum += ndcg(rel_pred, p=20, form="exp")
            
        mAP = mAP_sum/len(self.query)
        ndcg_score = ndcg_sum/len(self.query)
        return mAP, ndcg_score
        
    def on_train_begin(self, logs={}):
        mAP, ndcg_score = self.metric_calculator()
        self.history['val_nDCG'].append(ndcg_score)
        self.history['val_mAP'].append(mAP)

    def on_epoch_end(self, epoch, logs={}):
        mAP, ndcg_score = self.metric_calculator()
        self.history['val_nDCG'].append(ndcg_score)
        self.history['val_mAP'].append(mAP)
        print(' - val_ndcg: %.4f - val_mAP: %.4f'%(ndcg_score, mAP))