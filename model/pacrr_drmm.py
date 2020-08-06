#-*- coding:utf-8 -*-
from model.layers import Conv_stack, Dim_wise_max_pooling, Row_wise_max_pooling, Idf_concat
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

    def call(self, inputs, idf):
        x = self.conv_stack(inputs)
        x = self.dim_wise_max_pooling(x)
        x = self.row_wise_max_pooling(x)
        x = self.idf_concat(x, idf)
        return x
        
class DRMM(tf.keras.Model):
    def __init__(self):
        super(DRMM, self).__init__(name='DRMM')
        initializer = tf.keras.initializers.he_normal()
        self.dense1 = tf.keras.layers.Dense(5, activation='relu', kernel_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(1, activation='relu', kernel_initializer=initializer)
        self.dense3 = tf.keras.layers.Dense(1, activation='relu', kernel_initializer=initializer)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = tf.squeeze(x)
        x = self.dense3(x)
        
        return x
        
class PACRR_DRMM(tf.keras.Model):
    def __init__(self, firstk, lq, lg, nf, ns):
        super(PACRR_DRMM, self).__init__(name='PACRR_DRMM')
        self.firstk = firstk
        self.lq = lq
        self.lg = lg
        self.nf = nf
        self.ns = ns
        
        self.pacrr = PACRR(firstk=self.firstk, lq=self.lq, lg=self.lg, nf=self.nf, ns=self.ns)
        self.drmm = DRMM()
        
    def call(self, inputs, idf):
        x = self.pacrr(inputs, idf)
        x = self.drmm(x)
        return x
        
class Pairwise_PACRR_DRMM(tf.keras.Model):
    def __init__(self, firstk, lq, lg, nf, ns):
        super(Pairwise_PACRR_DRMM, self).__init__(name='Pairwise_PACRR_DRMM')
        self.Pacrr_Drmm = PACRR_DRMM(firstk, lq, lg, nf, ns)
        
    def call(self, inputs):
        positive_sim = inputs['positive_sim_matrix']
        negative_sim = inputs['negative_sim_matrix']
        idf_softmax = inputs['idf_softmax']
        
        positive = self.Pacrr_Drmm(positive_sim, idf_softmax)
        negative = self.Pacrr_Drmm(negative_sim, idf_softmax)
        
        return tf.concat([positive, negative], axis=0) 
    
    def predict(self, inputs):
        sim_matrix = inputs['sim_matrix']
        idf_softmax = inputs['idf_softmax']
        rel = self.Pacrr_Drmm(sim_matrix, idf_softmax)
        return rel