#-*- coding:utf-8 -*-
from model.layers import Conv_stack, Dim_wise_max_pooling, Row_wise_max_pooling, Idf_concat, Recurrent_Layer
import tensorflow as tf

class PACRR(tf.keras.Model):
    def __init__(self, firstk, lq, lg, nf, ns):
        super(PACRR, self).__init__(name='PACRR')
        self.firstk = firstk
        self.lq = lq
        self.lg = lg
        self.nf = nf
        self.ns = ns
        
        self.conv_stack = Conv_stack(lg=self.lg, nf=self.nf)
        self.dim_wise_max_pooling = Dim_wise_max_pooling(lg=self.lg, nf=self.nf)
        self.row_wise_max_pooling = Row_wise_max_pooling(lg=self.lg, ns=self.ns, firstk=self.firstk)
        self.idf_concat = Idf_concat()
        self.recurrent_Layer = Recurrent_Layer(lq=self.lq, ns=self.ns, lg=self.lg)

    def call(self, inputs, idf):
        x = self.conv_stack(inputs)
        x = self.dim_wise_max_pooling(x)
        x = self.row_wise_max_pooling(x)
        x = self.idf_concat(x, idf)
        x = self.recurrent_Layer(x)
        return x
        
class Pairwise_PACRR(tf.keras.Model):
    def __init__(self, firstk, lq, lg, nf, ns):
        super(Pairwise_PACRR, self).__init__(name='Pairwise_PACRR')
        self.Pacrr = PACRR(firstk, lq, lg, nf, ns)
        
    def call(self, inputs):
        positive_sim = inputs['positive_sim_matrix']
        negative_sim = inputs['negative_sim_matrix']
        idf = inputs['idf_softmax']
        
        positive = self.Pacrr(positive_sim, idf)
        negative = self.Pacrr(negative_sim, idf)
        
        return tf.concat([positive, negative], axis=0) 
    
    def predict(self, inputs):
        sim_matrix = inputs['sim_matrix']
        idf_softmax = inputs['idf_softmax']
        rel = self.Pacrr(sim_matrix, idf_softmax)
        return rel
        
        
        
        