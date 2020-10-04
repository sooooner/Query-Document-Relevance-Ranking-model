#-*- coding:utf-8 -*-
from model.layers import Conv_stack, Dim_wise_max_pooling, Row_wise_max_pooling, Idf_concat, Recurrent_Layer
import tensorflow as tf

class PACRR(tf.keras.Model):
    def __init__(self, lq, lg, nf, ns):
        super(PACRR, self).__init__(name='PACRR')
        self._supports_ragged_inputs = True     
        self.lq = lq
        self.lg = lg
        self.nf = nf
        self.ns = ns
        
        self.conv_stack = Conv_stack(lg=self.lg, nf=self.nf)
        self.dim_wise_max_pooling = Dim_wise_max_pooling(lg=self.lg, nf=self.nf)
        self.row_wise_max_pooling = Row_wise_max_pooling(lg=self.lg, ns=self.ns)
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
    def __init__(self, lq, lg, nf, ns):
        super(Pairwise_PACRR, self).__init__(name='Pairwise_PACRR')
        self._supports_ragged_inputs = True     
        self.Pacrr = PACRR(lq, lg, nf, ns)
        
    def call(self, inputs):
        positive_sim = inputs['positive_sim_matrix']
        negative_sim = inputs['negative_sim_matrix']
        idf = inputs['idf_softmax']
        
        positive = self.Pacrr(positive_sim, idf)
        negative = self.Pacrr(negative_sim, idf)
        
        return tf.concat([positive, negative], axis=0) 
        
        
def Gen_PACRR_Model(firstk, lq, lg, nf, ns, bert=False):
    if not bert:
        inputs = {'negative_sim_matrix': tf.keras.Input(shape=(lq, firstk), name='negative_sim_matrix'), 
                  'positive_sim_matrix': tf.keras.Input(shape=(lq, firstk), name='positive_sim_matrix'),
                  'idf_softmax'        : tf.keras.Input(shape=(lq), name='idf_softmax'),
                  'query_idf'          : tf.keras.Input(shape=(None,), ragged=True, name='query_idf'),
                  'positive_hist'      : tf.keras.Input(shape=(None, 30), ragged=True, name='positive_hist'),
                  'negative_hist'      : tf.keras.Input(shape=(None, 30), ragged=True, name='negative_hist')}
    else:
        inputs = {'negative_sim_matrix': tf.keras.Input(shape=(4, lq, firstk), name='negative_sim_matrix'), 
                  'positive_sim_matrix': tf.keras.Input(shape=(4, lq, firstk), name='positive_sim_matrix'),
                  'idf_softmax'        : tf.keras.Input(shape=(lq), name='idf_softmax'),
                  'query_idf'          : tf.keras.Input(shape=(None,), ragged=True, name='query_idf'),
                  'positive_hist'      : tf.keras.Input(shape=(None, 30), ragged=True, name='positive_hist'),
                  'negative_hist'      : tf.keras.Input(shape=(None, 30), ragged=True, name='negative_hist')}

    output = Pairwise_PACRR(lq, lg, nf, ns)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model
    