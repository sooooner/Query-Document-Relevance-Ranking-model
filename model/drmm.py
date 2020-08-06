#-*- coding:utf-8 -*-
from model.layers import Dense, Score_Aggregation, Gating_Network, Word_Matching_Network
import tensorflow as tf

class DRMM(tf.keras.Model):
    def __init__(self):
        super(DRMM, self).__init__(name='DRMM')
        self._supports_ragged_inputs = True
        self.Word_Matching_Network = Word_Matching_Network()
        self.Gating_Network = Gating_Network()
        self.Score_Aggregation = Score_Aggregation()

    def call(self, inputs, idf):
        Z = self.Word_Matching_Network(inputs)
        G = self.Gating_Network(idf)
        score = self.Score_Aggregation(Z, G)
        return score
        
class Pairwise_DRMM(tf.keras.Model):
    def __init__(self):
        super(Pairwise_DRMM, self).__init__(name='Pairwise_DRMM')
        self.drmm = DRMM()
        
    def call(self, inputs):
        positive_hist = inputs['positive_hist']
        negative_hist = inputs['negative_hist']
        query_idf = inputs['query_idf']
        
        positive = self.drmm(positive_hist, query_idf)
        negative = self.drmm(negative_hist, query_idf)
        
        return tf.concat([positive, negative], axis=0) 
    
    def predict(self, inputs):
        hist = inputs['hist']
        query_idf = inputs['query_idf']
        score = self.drmm(hist, query_idf)
        return score