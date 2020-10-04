#-*- coding:utf-8 -*-
import tensorflow as tf

def Pairwise_ranking_loss(y_true, y_pred):
    '''
    ignore y_true
    '''
    positive_score = tf.keras.layers.Lambda(lambda x: x[:len(x)//2], output_shape= (1,))(y_pred)
    negative_score = tf.keras.layers.Lambda(lambda x: x[len(x)//2:], output_shape= (1,))(y_pred)

    return tf.keras.backend.mean(tf.math.maximum(0., 1 - positive_score + negative_score))