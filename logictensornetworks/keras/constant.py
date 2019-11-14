"""
:Date: Nov 14, 2019
:Version: 0.0.1
"""

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as KL
from tensorflow.keras import backend

class Constants(KL.Layer):
    def __init__(self,
                 constants,
                 number_of_features,
                 *kwargs):
        super(Constants, self).__init__()
        # constants can be either a list of or a string
        # is it's a string every string's char will denotate a constant
        self.constants = {c: None for c in constants}
        self.number_of_features = number_of_features

    def build(self, inputs):
        min_val = [0.] * self.number_of_features
        max_val = [1.] * self.number_of_features

        for const in self.constants.keys():
            w = self.add_weight(name='const_' + const,
                                shape=(1, self.number_of_features),
                                initializer=K.initializers.RandomUniform(minval=min_val,
                                                                         maxval=max_val),
                                trainable=True)
            w.doms = []
            self.constants[const] = w

        super(Constants, self).build(inputs)

    def call(self, inputs=None):
        return [tf.identity(c) for c in self.constants.values()]
