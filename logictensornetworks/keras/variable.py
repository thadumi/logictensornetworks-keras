"""
A variable aka InputLayer (placeholder) or a combination of constants

:Date: Nov 18, 2019
:Version: 0.1.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras as K


class VariableLayer(K.layers.InputLayer):

    def __init__(self,
                 input_shape=None,
                 batch_size=None,
                 dtype=None,
                 input_tensor=None,
                 sparse=False,
                 name=None,
                 ragged=False,
                 **kwargs):
        super(VariableLayer, self).__init__(input_shape=input_shape,
                                            batch_size=batch_size,
                                            dtype=dtype,
                                            input_tensor=input_tensor,
                                            sparse=sparse,
                                            name=name,
                                            ragged=sparse)


def Variable(label=None,
             shape=None,
             tensor=None,
             sparse=False,
             ragged=False,
             **kwargs):
    if label is None:
        raise ValueError('Please provide to Variable the label value.')

    if sparse and ragged:
        raise ValueError('Cannot set both sparse and ragged to True in a Keras input.')

    if shape is None and tensor is None:
        raise ValueError('Please provide to Variable either a `shape`'
                         ' or a `tensor` argument. Note that '
                         '`shape` does not include the batch '
                         'dimension.')

    if tensor is None:  # a variable is a placeholder
        variable_layer = VariableLayer(name=label,
                                       input_shape=shape,
                                       input_tensor=tensor,
                                       sparse=sparse,
                                       ragged=ragged)

        # Return tensor including `_keras_history` and `doms`.
        # Note that in this case train_output and test_output are the same pointer.
        outputs = variable_layer._inbound_nodes[0].output_tensors
    else:
        # a variable is a combination of constants therefor is not a placeholder
        # TODO(thadumi) should provide the API for defining just the set of variables and then
        #               combine theme here with a KL.Constants(axis=0)
        outputs = (tensor,)

    if len(outputs) == 1:
        out = outputs[0]
        out._ltn_doms = [label]
    else:
        out = outputs
        for o in outputs:
            o._ltn_doms = [label]

    return out
