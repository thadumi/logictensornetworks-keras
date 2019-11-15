"""
A variable aka InputLayer

:Date: Nov 15, 2019
:Version: 0.0.1
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

    variable_layer = VariableLayer(name=label,
                                   input_shape=shape,
                                   input_tensor=tensor,
                                   sparse=sparse,
                                   ragged=ragged)

    # Return tensor including `_keras_history` and `doms`.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = variable_layer._inbound_nodes[0].output_tensors

    if len(outputs) == 1:
        out = outputs[0]
        out.doms = [label]
    else:
        out = outputs
        for o in outputs:
            o.doms = [label]

    return out
