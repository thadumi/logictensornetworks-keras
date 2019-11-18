"""
:Date: Nov 18, 2019
:Version: 0.1.1
"""

import tensorflow as tf

import logictensornetworks.backend as be
import logictensornetworks.keras as KLTN


class NotLayer(KLTN.LtnLayer):

    def __init__(self, *args, **kwargs):
        super(NotLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        result = be.F_Not(inputs)
        # label = 'NOT_' + input.name.split(":")[0] if not tf.executing_eagerly() else 'NOT_'
        return result

    def compute_doms(self, inputs, **kwargs):
        return inputs._ltn_doms


class AndLayer(KLTN.LtnLayer):

    def __init__(self, *kwargs):
        super(AndLayer, self).__init__(*kwargs)

        self._computed_doms = []

    def call(self, inputs):
        if len(inputs) == 0:
            result = tf.constant(1.0)
        else:
            cross_inputs, _ = be.cross_args(inputs)
            # label = '_AND_'.join([wff.name.split(":")[0] for wff in inputs])
            result = be.F_And(cross_inputs)
        self._computed_doms = cross_inputs._ltn_doms

        return result

    def compute_doms(self, inputs, **kwargs):
        return self._computed_doms


class OrLayer(KLTN.LtnLayer):

    def __init__(self, *kwargs):
        super(OrLayer, self).__init__(*kwargs)
        self._computed_doms = []

    def call(self, inputs):
        if len(inputs) == 0:
            result = tf.constant(0.0)
            result._ltn_doms = []
        else:
            cross_inputs, _ = be.cross_args(inputs)
            # label = "_OR_".join([wff.name.split(":")[0] for wff in inputs])
            result = be.F_Or(cross_inputs)
            result._ltn_doms = cross_inputs._ltn_doms

        self._computed_doms = result._ltn_doms

        return result

    def compute_doms(self, inputs, **kwargs):
        return self._computed_doms


class ImpliesLayer(KLTN.LtnLayer):

    def __init__(self, *kwargs):
        super(ImpliesLayer, self).__init__(*kwargs)

    def call(self, inputs):
        wff1 = inputs[0]
        wff2 = inputs[1]

        _, cross_wffs = be.cross_2args(wff1, wff2)
        # label = wff1.name.split(":")[0] + '_IMP_' + wff2.name.split(":")[0]

        result = be.F_Implies(cross_wffs[0], cross_wffs[1])
        result._ltn_doms = cross_wffs[0]._ltn_doms

        self._computed_doms = result._ltn_doms
        return result

    def compute_doms(self, inputs, **kwargs):
        return self._computed_doms


class EquivalenceLayer(KLTN.LtnLayer):

    def __init__(self, *kwargs):
        super(EquivalenceLayer, self).__init__(*kwargs)

    def call(self, inputs):
        wff1 = inputs[0]
        wff2 = inputs[1]

        _, cross_wffs = be.cross_2args(wff1, wff2)
        label = wff1.name.split(":")[0] + "_IFF_" + wff2.name.split(":")[0]

        result = be.F_Equiv(cross_wffs[0], cross_wffs[1])
        result._ltn_doms = cross_wffs[0]._ltn_doms
        self._computed_doms = result._ltn_doms
        return result

    def compute_doms(self, inputs, **kwargs):
        return self._computed_doms


class ForallLayer(KLTN.LtnLayer):

    def __init__(self, *kwargs):
        super(ForallLayer, self).__init__(*kwargs)

    def call(self, inputs):
        vars = inputs[:-1]
        wff = inputs[-1]

        result_doms = [x for x in wff._ltn_doms if x not in [var._ltn_doms[0] for var in vars]]
        quantif_axis = [wff._ltn_doms.index(var._ltn_doms[0]) for var in vars]
        not_empty_vars = tf.cast(tf.reduce_prod(tf.stack([tf.size(var) for var in vars])), tf.bool)
        ones = tf.ones((1,) * (len(result_doms) + 1))
        result = tf.cond(not_empty_vars, lambda: be.F_ForAll(quantif_axis, wff), lambda: ones)
        result._ltn_doms = result_doms
        self._computed_doms = result._ltn_doms
        return result

    def compute_doms(self, inputs, **kwargs):
        return self._computed_doms


class ExistsLayer(KLTN.LtnLayer):

    def __init__(self, *kwargs):
        super(ExistsLayer, self).__init__(*kwargs)

    def call(self, inputs):
        vars = inputs[:-1]
        wff = inputs[-1]

        result_doms = [x for x in wff._ltn_doms if x not in [var._ltn_doms[0] for var in vars]]
        quantif_axis = [wff._ltn_doms.index(var._ltn_doms[0]) for var in vars]
        not_empty_vars = tf.cast(tf.reduce_prod(tf.stack([tf.size(var) for var in vars])), tf.bool)
        zeros = tf.zeros((1,) * (len(result_doms) + 1))
        result = tf.cond(not_empty_vars, lambda: be.F_Exists(quantif_axis, wff), lambda: zeros)
        result._ltn_doms = result_doms
        self._computed_doms = result._ltn_doms
        return result

    def compute_doms(self, inputs, **kwargs):
        return self._computed_doms


def Not(tensor):
    return NotLayer()(tensor)


def And(*tensors):
    return AndLayer()(tensors)


def Or(*tensors):
    return OrLayer()(tensors)


def Implies(*tensors):
    return ImpliesLayer()(tensors)


def Equiv(*tensors):
    return EquivalenceLayer()(tensors)


def Exists(vars, tensor):
    if type(vars) is not tuple:
        vars = (vars,)

    return ExistsLayer()([*vars, tensor])


def Forall(vars, tensor):
    if type(vars) is not tuple:
        vars = (vars,)

    return ForallLayer()([*vars, tensor])
