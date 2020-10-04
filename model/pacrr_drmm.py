#-*- coding:utf-8 -*-
from model.layers import Conv_stack, Dim_wise_max_pooling, Row_wise_max_pooling, Idf_concat
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

    def call(self, inputs, idf):
        x = self.conv_stack(inputs)
        x = self.dim_wise_max_pooling(x)
        x = self.row_wise_max_pooling(x)
        x = self.idf_concat(x, idf)
        return x
        
class DRMM(tf.keras.Model):
    def __init__(self, lq):
        super(DRMM, self).__init__(name='DRMM')
        self._supports_ragged_inputs = True     
        self.lq = lq
        # initializer = tf.keras.initializers.he_normal()
        # initializer = tf.keras.initializers.GlorotNormal()
        self.dense1 = tf.keras.layers.Dense(5, activation=tf.keras.activations.elu, kernel_initializer=tf.keras.initializers.he_normal())
        self.dense2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.elu, kernel_initializer=tf.keras.initializers.he_normal())
        self.dense3 = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh, kernel_initializer=tf.keras.initializers.GlorotNormal())

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = tf.squeeze(x, axis=-1)
        x = self.dense3(x)
        return x
        
class PACRR_DRMM(tf.keras.Model):
    def __init__(self, lq, lg, nf, ns):
        super(PACRR_DRMM, self).__init__(name='PACRR_DRMM')
        self._supports_ragged_inputs = True     
        self.lq = lq
        self.lg = lg
        self.nf = nf
        self.ns = ns
        
        self.pacrr = PACRR(lq=self.lq, lg=self.lg, nf=self.nf, ns=self.ns)
        self.drmm = DRMM(self.lq)
        
    def call(self, inputs, idf):
        x = self.pacrr(inputs, idf)
        x = self.drmm(x)
        return x
        
class Pairwise_PACRR_DRMM(tf.keras.Model):
    def __init__(self, lq, lg, nf, ns):
        super(Pairwise_PACRR_DRMM, self).__init__(name='Pairwise_PACRR_DRMM')
        self.Pacrr_Drmm = PACRR_DRMM(lq, lg, nf, ns)
        
    def call(self, inputs):
        positive_sim = inputs['positive_sim_matrix']
        negative_sim = inputs['negative_sim_matrix']
        idf_softmax = inputs['idf_softmax']
        
        positive = self.Pacrr_Drmm(positive_sim, idf_softmax)
        negative = self.Pacrr_Drmm(negative_sim, idf_softmax)
        
        return tf.concat([positive, negative], axis=0) 

        
def Gen_PACRR_DRMM_Model(firstk, lq, lg, nf, ns, bert=False):
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

    output = Pairwise_PACRR_DRMM(lq, lg, nf, ns)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model