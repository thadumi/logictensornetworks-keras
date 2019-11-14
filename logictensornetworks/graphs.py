from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import backend

_And = None
F_Or = None
F_Implies = None
F_Equiv = None
F_Not = None
F_Forall = None
F_Exists = None


def set_tnorm(tnorm):
    assert tnorm in ['min', 'luk', 'prod', 'mean', '']
    global F_And, F_Or, F_Implies, F_Not, F_Equiv, F_Forall

    if tnorm == "luk":
        def F_And(wffs):
            with GRAPH.as_default():
                a = tf.math.reduce_sum(wffs, axis=-1, keepdims=True)
                b = tf.cast(tf.shape(wffs)[-1], dtype=tf.dtypes.float32)
                c = tf.cast(tf.shape(wffs)[-1], dtype=tf.dtypes.float32)

                ret = tf.math.maximum(0.0, tf.math.reduce_sum(wff, axis=-1, keepdims=True) + 1 - c)

            return ret

        def F_Or(wffs):
            with GRAPH.as_default():
                ret = tf.math.minimum(tf.math.reduce_sum(wffs, axis=-1, keepdims=True), 1.0)

            return ret

        def F_Implies(wff1, wff2):
            with GRAPH.as_default():
                ret = tf.math.minimum(1., 1 - wff1 + wff2)
            return ret

        def F_Not(wff):
            with GRAPH.as_default():
                ret = 1 - wff
            return ret

        def F_Equiv(wff1, wff2):
            with GRAPH.as_default():
                ret = 1 - tf.math.abs(wff1 - wff2)

            return ret

def set_universal_aggreg(aggreg):
    assert aggreg in ['hmean', 'min', 'mean']
    global F_Forall
    if aggreg == "hmean":
        def F_Forall(axis, wff):
            with GRAPH.as_default():
                agg = 1 / tf.math.reduce_mean(1 / (wff + 1e-10), axis=axis)
            return agg

    if aggreg == "min":
        def F_Forall(axis, wff):
            with GRAPH.as_default():
                agg = tf.math.reduce_min(wff, axis=axis)
            return agg

    if aggreg == "mean":
        def F_Forall(axis, wff):
            with GRAPH.as_default():
                agg = tf.math.reduce_mean(wff, axis=axis)
            return agg

def set_existential_aggregator(aggreg):
    assert aggreg in ['max']
    global F_Exists
    if aggreg == "max":
        def F_Exists(axis, wff):
            with GRAPH.as_default():
                e = tf.math.reduce_max(wff, axis=axis)
            return e


set_tnorm("luk")
set_universal_aggreg("hmean")
set_existential_aggregator("max")

GRAPH = tf.Graph()

def Not(wff):
    with GRAPH.as_default():
        result = F_Not(wff)
        label = "NOT_" + wff.name.split(":")[0]
        result = tf.identity(result, name=label)
        result.doms = wff.doms
        return result

def Forall(args, wff):
    if type(args) is not tuple:
        args = (args,)

    result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in args]]
    quantif_axis = [wff.doms.index(var.doms[0]) for var in args]

    with GRAPH.as_default():
        not_empty_vars = tf.cast(tf.math.reduce_prod(tf.stack([tf.size(var) for var in args])),
                                 dtype=tf.dtypes.bool)

        ones = tf.ones((1,) * (len(result_doms) + 1))
        result = tf.cond(not_empty_vars, lambda: F_Forall(quantif_axis, wff), lambda: ones)
    result.doms = result_doms

    return result


def constant(label, value=None, min_value=None, max_value=None):
    label = 'ltn_constant_' + label
    const = None
    with GRAPH.as_default():
        if value is not None:
            const = tf.constant(value, name=label)
        else:
            const = tf.Variable(tf.random.uniform(shape=(1, len(min_value)),
                                                  minval=min_value,
                                                  maxval=max_value, name=label),
                                name=label)
    const.doms = []
    return const


def variable(label, feed, number_of_features=None):
    var = None
    with GRAPH.as_default():
        if tf.is_tensor(feed):
            var = tf.identity(feed, name=label)
        elif type(feed) is list:
            var = tf.concat(feed, axis=0)
        else:
            var = tf.constant(feed, name=label)

    var.doms = [label]
    return var


def predicate(label, number_of_features_or_vars, pred_definition=None, layers=4):
    if type(number_of_features_or_vars) is list:
        number_of_features = sum([int(v.shape[1]) for v in number_of_features_or_vars])
    elif tf.is_tensor(number_of_features_or_vars):
        number_of_features = int(number_of_features_or_vars.shape[1])
    else:
        number_of_features = number_of_features_or_vars

    def _create_predicate_model():
        W = None
        u = None

        with GRAPH.as_default():
            W = tf.linalg.band_part(
                tf.Variable(
                    tf.random.normal([layers,
                                      number_of_features + 1,
                                      number_of_features + 1],
                                     mean=0,
                                     stddev=1),
                    name="W" + label),
                0, -1)

            u = tf.Variable(tf.ones([layers, 1]), name="u" + label)

        def _apply_pred(*args):
            with GRAPH.as_default():
                app_label = label + "/" + "_".join([arg.name.split(":")[0] for arg in args]) + "/"
                tensor_args = tf.concat(args, axis=1)
                X = tf.concat([tf.ones((tf.shape(tensor_args)[0], 1)),
                               tensor_args], 1)
                XW = tf.matmul(tf.tile(tf.expand_dims(X, 0), [layers, 1, 1]), W)
                XWX = tf.squeeze(tf.matmul(tf.expand_dims(X, 1), tf.transpose(XW, [1, 2, 0])), axis=[1])
                gX = tf.matmul(tf.tanh(XWX), u)
                result = tf.sigmoid(gX, name=app_label)
                return result

        return _apply_pred

    if pred_definition is None:
        apply_pred = _create_predicate_model()
    else:
        def apply_pred(*args):
            return pred_definition(*args)

    def _predicate(*args):
        with GRAPH.as_default():
            crossed_args, list_of_args_in_crossed_args = cross_args(args)
            result = apply_pred(*list_of_args_in_crossed_args)
            if crossed_args.doms != []:
                result = tf.reshape(result, tf.concat([tf.shape(crossed_args)[:-1], [1]], axis=0))
            else:
                result = tf.reshape(result, (1,))
            result.doms = crossed_args.doms
        return result

    return _predicate


def cross_args(args):
    result = args[0]

    for arg in args[1:]:
        result, _ = cross_2args(result, arg)

    result_flat = tf.reshape(result,
                             (tf.math.reduce_prod(tf.shape(result)[:-1]),
                              tf.shape(result)[-1]))

    result_args = tf.split(result_flat, [tf.shape(arg)[-1] for arg in args], 1)
    return result, result_args


def cross_2args(X, Y):
    if X.doms == [] and Y.doms == []:
        result = tf.concat([X, Y], axis=-1)
        result.doms = []
        return result, [X, Y]

    X_Y = set(X.doms) - set(Y.doms)
    Y_X = set(Y.doms) - set(X.doms)

    eX = X
    eX_doms = [x for x in X.doms]
    for y in Y_X:
        eX = tf.expand_dims(eX, 0)
        eX_doms = [y] + eX_doms

    eY = Y
    eY_doms = [y for y in Y.doms]
    for x in X_Y:
        eY = tf.expand_dims(eY, -2)
        eY_doms.append(x)

    perm_eY = []
    for y in eY_doms:
        perm_eY.append(eX_doms.index(y))

    eY = tf.transpose(eY, perm=perm_eY + [len(perm_eY)])
    mult_eX = [1] * (len(eX_doms) + 1)
    mult_eY = [1] * (len(eY_doms) + 1)

    for i in range(len(mult_eX) - 1):
        mult_eX[i] = tf.math.maximum(1, tf.math.floordiv(tf.shape(eY)[i], tf.shape(eX)[i]))
        mult_eY[i] = tf.math.maximum(1, tf.math.floordiv(tf.shape(eX)[i], tf.shape(eY)[i]))

    result1 = tf.tile(eX, mult_eX)
    result2 = tf.tile(eY, mult_eY)
    result = tf.concat([result1, result2], axis=-1)

    result1.doms = eX_doms
    result2.doms = eX_doms
    result.doms = eX_doms

    return result, [result1, result2]


size = 20
g1 = {l: constant(label=l, min_value=[0.] * size, max_value=[1.] * size) for l in 'abcdefgh'}
g2 = {l: constant(label=l, min_value=[0.] * size, max_value=[1.] * size) for l in 'ijklmn'}
g = {**g1, **g2}

friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
           ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
smokes = ['a', 'e', 'f', 'g', 'j', 'n']
cancer = ['a', 'e']

p = variable('p', list(g.values()))
q = variable('q', list(g.values()))
p1 = variable('p1', list(g1.values()))
q1 = variable('q1', list(g1.values()))
p2 = variable('p2', list(g2.values()))
q2 = variable('q2', list(g2.values()))

Friends = predicate('Friends', size * 2)
Smokes = predicate('Smokes', size)
Cancer = predicate('Cancer', size)