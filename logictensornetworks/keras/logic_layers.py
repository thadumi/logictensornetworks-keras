"""
:Date: Nov 15, 2019
:Version: 0.0.1
"""

import tensorflow as tf
from tensorflow.keras import layers as KL

import logictensornetworks.backend as be


class NotLayer(KL.Layer):

    def __init__(self, *kwargs):
        super(NotLayer, self).__init__(*kwargs)

    def call(self, input):
        #  TODO: assert input is only one tensor

        result = be.F_Not(input)
        label = 'NOT_' + input.name.split(":")[0] if not tf.executing_eagerly() else 'NOT_'
        result = tf.identity(result, name=label)
        result.doms = input.doms

        self.doms = result.doms
        return result


class AndLayer(KL.Layer):

    def __init__(self, *kwargs):
        super(AndLayer, self).__init__(*kwargs)

    def call(self, inputs):
        #  TODO: assert inputs

        if len(inputs) == 0:
            result = tf.constant(1.0)
            result.doms = []
        else:
            cross_inputs, _ = be.cross_args(inputs)
            label = '_AND_'.join([wff.name.split(":")[0] for wff in inputs])
            result = tf.identity(be.F_And(cross_inputs), name=label)
            result.doms = cross_inputs.doms
        self.doms = result.doms
        return result


class OrLayer(KL.Layer):

    def __init__(self, *kwargs):
        super(ImpliesLayer, self).__init__(*kwargs)

    def call(self, inputs):
        #  TODO: assert inputs

        if len(inputs) == 0:
            result = tf.constant(0.0)
            result.doms = []
        else:
            cross_inputs, _ = be.cross_args(inputs)
            label = "_OR_".join([wff.name.split(":")[0] for wff in inputs])
            result = tf.identity(be.F_Or(cross_inputs), name=label)
            result.doms = cross_inputs.doms
        self.doms = result.doms
        return result


class ImpliesLayer(KL.Layer):

    def __init__(self, *kwargs):
        super(ImpliesLayer, self).__init__(*kwargs)

    def call(self, inputs):
        # TODO: assert inputs
        wff1 = inputs[0]
        wff2 = inputs[1]

        _, cross_wffs = be.cross_2args(wff1, wff2)
        label = wff1.name.split(":")[0] + '_IMP_' + wff2.name.split(":")[0]
        result = be.F_Implies(cross_wffs[0], cross_wffs[1])
        result = tf.identity(result, name=label)
        result.doms = cross_wffs[0].doms
        self.doms = result.doms
        return result


class EquivalenceLayer(KL.Layer):

    def __init__(self, *kwargs):
        super(EquivalenceLayer, self).__init__(*kwargs)

    def call(self, inputs):
        # TODO: assert inputs

        wff1 = inputs[0]
        wff2 = inputs[1]

        _, cross_wffs = be.cross_2args(wff1, wff2)
        label = wff1.name.split(":")[0] + "_IFF_" + wff2.name.split(":")[0]

        result = be.F_Equiv(cross_wffs[0], cross_wffs[1])
        result.doms = cross_wffs[0].doms
        self.doms = result.doms
        return result


class ForallLayer(KL.Layer):

    def __init__(self, *kwargs):
        super(ForallLayer, self).__init__(*kwargs)

    def call(self, inputs):
        # TODO: assert inputs

        vars = inputs[:-1]
        wff = inputs[-1]

        result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in vars]]
        quantif_axis = [wff.doms.index(var.doms[0]) for var in vars]
        not_empty_vars = tf.cast(tf.reduce_prod(tf.stack([tf.size(var) for var in vars])), tf.bool)
        ones = tf.ones((1,) * (len(result_doms) + 1))
        result = tf.cond(not_empty_vars, lambda: be.F_ForAll(quantif_axis, wff), lambda: ones)
        result.doms = result_doms
        self.doms = result.doms
        return result


class ExistsLayer(KL.Layer):

    def __init__(self, *kwargs):
        super(ExistsLayer, self).__init__(*kwargs)

    def call(self, inputs):
        # TODO: assert inputs

        vars = inputs[:-1]
        wff = inputs[-1]

        result_doms = [x for x in wff.doms if x not in [var.doms[0] for var in vars]]
        quantif_axis = [wff.doms.index(var.doms[0]) for var in vars]
        not_empty_vars = tf.cast(tf.reduce_prod(tf.stack([tf.size(var) for var in vars])), tf.bool)
        zeros = tf.zeros((1,) * (len(result_doms) + 1))
        result = tf.cond(not_empty_vars, lambda: be.F_Exists(quantif_axis, wff), lambda: zeros)
        result.doms = result_doms
        self.doms = result.doms
        return result


def Not(tensor):
    layer = NotLayer()
    tensor = layer(tensor)
    tensor.doms = layer.doms
    return tensor


def And(*tensors):
    layer = AndLayer()
    tensor = layer(tensors)
    tensor.doms = layer.doms
    return tensor


def Or(*tensors):
    layer = OrLayer()
    tensor = layer(tensors)
    tensor.doms = layer.doms
    return tensor


def Implies(*tensors):
    layer = ImpliesLayer()
    tensor = layer(tensors)
    tensor.doms = layer.doms
    return tensor


def Equiv(*tensors):
    layer = EquivalenceLayer()
    tensor = layer(tensors)
    tensor.doms = layer.doms
    return tensor


def Exists(*tensors):
    layer = ExistsLayer()
    tensor = layer(tensors)
    tensor.doms = layer.doms
    return tensor


def Forall(*tensors):
    layer = ForallLayer()
    tensor = layer(tensors)
    tensor.doms = layer.doms
    return tensor
