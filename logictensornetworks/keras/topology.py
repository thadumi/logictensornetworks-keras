"""
:Date: Nov 12, 2019
:Version: 0.0.1
"""

import tensorflow as tf
from tensorflow import keras as K

from logictensornetworks import backend as be

BIAS_factor = 1e-7
BIAS = tf.Variable(0.0, trainable=False)

'''
x = K.Input(shape=(32,))

CONSTANTS = {}
VARIABLE = {}


def constant(label, shape):
    CONSTANTS[label] = shape


def variable(shape, label):
    VARIABLE[label] = shape


size = 20
for l in 'abcdefghijklmn':
    constant(l, size)

for l in 'pq':
    variable((14, 20), l)

for l in ('p1', 'q1'):
    variable((8, 20), l)

for l in ('p2', 'q2'):
    variable((6, 20), l)


def total_shape_constants():
    return tuple(CONSTANTS.values())


print(len(total_shape_constants()))
'''


def _variable(label, features=None, tensor=None, const=None):
    """

    :param label: the name of the tensor
    :param features: a scalar representing the number of features of the variable has to be passed if a tensor is not specified
                     Will create a K.Input tensor of the shape (None, features)
    :param tensor: a tf.Tensor which represent initial value of the variable
    :return: a tensor representing the variable
    """
    if features is not None and type(features) is int:
        var = K.Input(dtype=tf.float32, shape=(None, features), name=label)
    elif tensor is not None and tf.is_tensor(tensor):
        var = tf.identity(tensor, name=label)
    else:
        var = tf.constant(const, name=label)

    var.doms = [label]
    return var

def constant(label, value=None, min_value=None, max_value=None):
    if value is not None:
        const = tf.constant(value, name=label)
    else:
        const = tf.constant(tf.random.uniform(
            shape=(1, len(min_value)),
            minval=min_value,
            maxval=max_value),
            name=label)

    const.doms = []
    return const


class Predicate(K.layers.Layer):

    def __init__(self,
                 label,
                 number_of_features_or_vars=None,
                 pred_definition=None,
                 layers=None,
                 *kwargs):
        super(Predicate, self).__init__(*kwargs)

        self.label = label
        self.number_of_features = None

        self.pred_definition = pred_definition
        self.layers = layers or 4

        if type(number_of_features_or_vars) is list:
            self.number_of_features = sum([int(v.shape[1]) for v in number_of_features_or_vars])
        elif tf.is_tensor(number_of_features_or_vars):
            self.number_of_features = int(number_of_features_or_vars.shape[1])
        else:
            self.number_of_features = number_of_features_or_vars

        assert type(self.number_of_features) is int

        # weights
        self.w = None
        self.u = None

    def build(self):
        # TODO: check if is passing a layer or a

        if self.pred_definition is None:
            # if there is not a custom predicate model defined create one using the default structure

            self.w = self.add_weight(name=self.label + "_W",
                                     shape=(self.layers, self.number_of_features + 1, self.number_of_features + 1),
                                     initializer=K.initializers.RandomNormal(mean=0, stddev=1, seed=None))

            self.u = self.add_weight(name=self.label + "_u",
                                     shape=(self.layers, 1),
                                     initializer=K.initializers.Ones())
        super(Predicate, self).build(True)

    def call(self, inputs):
        crossed_args, list_of_args_in_crossed_args = be.cross_args(inputs)
        result = self._call_default_model(*list_of_args_in_crossed_args)

        if crossed_args.doms:
            result = tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1], [1]], axis=0))
        else:
            result = tf.reshape(result, (1,))
        result.doms = crossed_args.doms

        BIAS.assign(tf.divide(BIAS + .5 - tf.reduce_mean(result), 2) * BIAS_factor)
        return result

    def _call_default_model(self, inputs):
        app_label = self.label + "/" + "_".join([arg.name.split(":")[0] for arg in inputs]) + "/"
        tensor_args = tf.concat(inputs, axis=1)

        W = tf.linalg.band_part(self.w, 0, -1)
        X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)), tensor_args], 1)

        XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [self.layers, 1, 1]), W)
        XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])

        gX = tf.matmul(tf.tanh(XWX), self.u)

        return tf.sigmoid(gX, name=app_label)


'''
g1 = {l: constant(l, min_value=[0.] * 20, max_value=[1.] * 20) for l in 'abcdefgh'}
g2 = {l: constant(l, min_value=[0.] * 20, max_value=[1.] * 20) for l in 'ijklmn'}
g = {**g1, **g2}

friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
           ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
smokes = ['a', 'e', 'f', 'g', 'j', 'n']
cancer = ['a', 'e']

p = _variable('p', tf.concat(list(g.values()), axis=0))
'''