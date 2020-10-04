#-*- coding:utf-8 -*-
from model.layers import Dense, Score_Aggregation, Gating_Network, Word_Matching_Network
import tensorflow as tf

class DRMM(tf.keras.Model):
    def __init__(self):
        super(DRMM, self).__init__(name='DRMM')
        self._supports_ragged_inputs = True
        self.word_matching_network = Word_Matching_Network()
        self.gating_network = Gating_Network()
        self.score_aggregation = Score_Aggregation()

    def call(self, inputs, idf):
        Z = self.word_matching_network(inputs)
        G = self.gating_network(idf)
        score = self.score_aggregation(Z, G)
        return score
        
class Pairwise_DRMM(tf.keras.Model):
    def __init__(self):
        super(Pairwise_DRMM, self).__init__(name='Pairwise_DRMM')
        self._supports_ragged_inputs = True     
        self.drmm = DRMM()
        
    def call(self, inputs):
        positive_hist = inputs['positive_hist']
        negative_hist = inputs['negative_hist']
        query_idf = inputs['query_idf']
        
        positive = self.drmm(positive_hist, query_idf)
        negative = self.drmm(negative_hist, query_idf)
        
        return tf.concat([positive, negative], axis=0) 
        
        
def Gen_DRMM_Model(bert=False):
    if not bert:
        lq=6
        firstk=8
        inputs = {'negative_sim_matrix': tf.keras.Input(shape=(lq, firstk), name='negative_sim_matrix'), 
                  'positive_sim_matrix': tf.keras.Input(shape=(lq, firstk), name='positive_sim_matrix'),
                  'idf_softmax'        : tf.keras.Input(shape=(lq), name='idf_softmax'),
                  'query_idf'          : tf.keras.Input(shape=(None,), ragged=True, name='query_idf'),
                  'positive_hist'      : tf.keras.Input(shape=(None, 30), ragged=True, name='positive_hist'),
                  'negative_hist'      : tf.keras.Input(shape=(None, 30), ragged=True, name='negative_hist')}
    else:
        lq=8
        firstk=13
        inputs = {'negative_sim_matrix': tf.keras.Input(shape=(4, lq, firstk), name='negative_sim_matrix'), 
                  'positive_sim_matrix': tf.keras.Input(shape=(4, lq, firstk), name='positive_sim_matrix'),
                  'idf_softmax'        : tf.keras.Input(shape=(lq), name='idf_softmax'),
                  'query_idf'          : tf.keras.Input(shape=(None,), ragged=True, name='query_idf'),
                  'positive_hist'      : tf.keras.Input(shape=(None, 30), ragged=True, name='positive_hist'),
                  'negative_hist'      : tf.keras.Input(shape=(None, 30), ragged=True, name='negative_hist')}

    output = Pairwise_DRMM()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model