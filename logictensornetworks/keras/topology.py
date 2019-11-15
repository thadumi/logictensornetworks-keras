"""
:Date: Nov 15, 2019
:Version: 0.0.2
"""

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as KL
from tensorflow.keras import backend
from constant import *
from variable import *

from logictensornetworks import backend as be


class Predicate(KL.Layer):
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

    def build(self, inputs_shape):
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

        super(Predicate, self).build(inputs_shape)

    def call(self, inputs):
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
        # app_label = self.label + "/" + "_".join([arg.name.split(":")[0] for arg in inputs]) + "/"
        tensor_args = tf.concat(inputs, axis=1)

        W = tf.linalg.band_part(self.w, 0, -1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)), tensor_args], 1)

        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.layers, 1, 1]), W)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])

        gX = tf.matmul(tf.tanh(XWX), self.u)

        return tf.sigmoid(gX)


# K.Input(shape=shape, name='constant_' + label)
def __constant(label, shape):
    # c = K.Input(shape=embedding_size, name=label)
    c = tf.Variable(tf.random.uniform(
        shape=(1, shape),
        minval=[0.] * shape,
        maxval=[1.] * shape),
        name=label)
    c.doms = []
    return c


def predicate(label, features):
    return lambda *inputs: Predicate(label, number_of_features=features)(*inputs)


if __name__ == '__main__':


    embedding_size = 10  # each constant is interperted in a vector of this size

    # create on constant for each individual a,b,... i,j, ...

    g1 = {l: Constant(l, min_value=[0] * embedding_size, max_value=[1.] * embedding_size) for l in 'abcdefgh'}
    g2 = {l: Constant(l, min_value=[0] * embedding_size, max_value=[1.] * embedding_size) for l in 'ijklmn'}
    g = {**g1, **g2}

    friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
               ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
    smokes = ['a', 'e', 'f', 'g', 'j', 'n']
    cancer = ['a', 'e']

    p = Variable(label='p', tensor=KL.Concatenate()(list(g.values())))
    q = Variable(label='q', tensor=KL.Concatenate()(list(g.values())))
    p1 = Variable(label='p1', tensor=KL.Concatenate()(list(g1.values())))
    q1 = Variable(label='q1', tensor=KL.Concatenate()(list(g1.values())))
    p2 = Variable(label='p2', tensor=KL.Concatenate()(list(g2.values())))
    q2 = Variable(label='q2', tensor=KL.Concatenate()(list(g2.values())))


    Friends = predicate('Friends', embedding_size * 2)
    Smokers = predicate('Smokers', embedding_size)
    Cancer = predicate('Cancer', embedding_size)

    facts = [Friends([g[x], g[y]]) for (x, y) in friends]

    '''
    a = Constant('a', min_value=[0] * embedding_size, max_value=[1.] * embedding_size)
    b = Constant('b', min_value=[0] * embedding_size, max_value=[1.] * embedding_size)
    
    Friends = predicate('Friends', embedding_size * 2)
    
    out = Friends([a, b])
    model = K.Model(inputs=[a, b], outputs=out)
    '''
