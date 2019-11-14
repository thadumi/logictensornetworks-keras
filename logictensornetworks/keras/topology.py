"""
:Date: Nov 12, 2019
:Version: 0.0.1
"""

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import backend

from logictensornetworks import backend as be


# BIAS_factor = 1e-7
# BIAS = tf.Variable(0.0, trainable=False)


class Predicate(K.layers.Layer):
    def __init__(self,
                 label,
                 number_of_features=None,
                 pred_definition=None,
                 layers=None,
                 *kwargs):
        super(Predicate, self).__init__(*kwargs)

        self.label = label
        self.number_of_features = None

        self.pred_definition = pred_definition
        self.layers = layers or 4

        if type(number_of_features) is list:
            self.number_of_features = sum([int(v.shape[1]) for v in number_of_features])
        elif tf.is_tensor(number_of_features):
            self.number_of_features = int(number_of_features.shape[1])
        else:
            self.number_of_features = number_of_features

        assert type(self.number_of_features) is int

        # weights
        self.w = None
        self.u = None

    def build(self, inputs):
        # TODO: check if is passing a layer or a
        # https://github.com/keras-team/keras/issues/8131

        if self.pred_definition is None:
            # if there is not a custom predicate model defined create one using the default structure
            self.w = self.add_weight(name=self.label + "_W",
                                     shape=(self.layers, self.number_of_features + 1, self.number_of_features + 1),
                                     initializer=K.initializers.RandomNormal(mean=0, stddev=1, seed=None),
                                     trainable=True)

            self.u = self.add_weight(name=self.label + "_u",
                                     shape=(self.layers, 1),
                                     initializer=K.initializers.Ones(),
                                     trainable=True)

        super(Predicate, self).build(True)

    def call(self, *inputs):
        crossed_args, list_of_args_in_crossed_args = be.cross_args(inputs)
        result = self._call_default_model(*list_of_args_in_crossed_args)

        if crossed_args.doms:
            result = tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1], [1]], axis=0))
        else:
            result = tf.reshape(result, (1,))
        result.doms = crossed_args.doms

        # BIAS.assign(tf.divide(BIAS + .5 - tf.reduce_mean(result), 2) * BIAS_factor)
        return result

    def _call_default_model(self, *inputs):
        app_label = self.label + "/" + "_".join([arg.name.split(":")[0] for arg in inputs]) + "/"
        tensor_args = tf.concat(inputs, axis=1)

        W = tf.linalg.band_part(self.w, 0, -1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)), tensor_args], 1)

        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.layers, 1, 1]), W)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])

        gX = tf.matmul(tf.tanh(XWX), self.u)

        return tf.sigmoid(gX, name=app_label)


# K.Input(shape=shape, name='constant_' + label)
def constant(label, shape):
    c = tf.Variable()
    c.doms = []

def variable(label, shape):
    pass

def predicate(label, features):
    return lambda *inputs: Predicate(label, number_of_features=features)(*inputs)


embedding_size = 10  # each constant is interperted in a vector of this size

# create on constant for each individual a,b,... i,j, ...
constants = {l: constant(l, embedding_size) for l in 'abcdefghijklmn'}

friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
           ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]

Friends = predicate('Friends', embedding_size * 2)
Smokers = predicate('Smokers', embedding_size)
Cancer = predicate('Cancer', embedding_size)

facts = [Friends(constants[x], constants[y]) for (x, y) in friends]
out = K.layers.Concatenate()(facts)

model = K.Model(inputs=constants, outputs=out)
