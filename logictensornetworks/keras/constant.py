"""
A constant aka InputLayer with one weight ano no placeholders

:Date: Nov 15, 2019
:Version: 0.1.2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras as K
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import node as node_module


class ConstantLayer(base_layer.Layer):

    def __init__(self,
                 label=None,
                 min_value=None,
                 max_value=None,
                 **kwargs):
        if not label:
            prefix = 'C'
            name = prefix + '_' + str(backend.get_uid(prefix))
        else:
            name = label

        dtype = backend.floatx()

        super(ConstantLayer, self).__init__(dtype=dtype, name=name)

        # backend.random_uniform_variable(shape=(20,), low=[0.]*20, high=[1.]*20)
        self._c = self.add_weight(name=label,
                                  shape=(1, len(min_value)),
                                  initializer=K.initializers.RandomUniform(minval=min_value,
                                                                           maxval=max_value),
                                  trainable=True)
        self.is_placeholder = False
        self.built = True

        graph = backend.get_graph()
        with graph.as_default():
            fake_input_tensor = backend.constant(backend.get_value(self._c),
                                                 )
        # Create an input node to add to self.outbound_node
        # and set output_tensors' _keras_history.

        fake_input_tensor._keras_history = base_layer.KerasHistory(self, 0, 0)
        fake_input_tensor._keras_mask = None
        fake_input_tensor.doms = []

        node_module.Node(
            self,
            inbound_layers=[],
            node_indices=[],
            tensor_indices=[],
            input_tensors=[fake_input_tensor],
            output_tensors=[fake_input_tensor])


def Constant(label=None,
             min_value=None,
             max_value=None,
             **kwargs):
    if label is None:
        raise ValueError('Please provide to Variable the label value.')

    if min_value is None and max_value is None:
        raise ValueError('Please provide to Constant either a `min_value`'
                         ' or a `max_value` argument. Note that '
                         'you can also provide both.')

    if min_value is None:
        min_value = [0.] * len(max_value)
    elif max_value is None:
        max_value = [1.] * len(min_value)

    constant_layer = ConstantLayer(label=label,
                                   min_value=min_value,
                                   max_value=max_value)

    # Return tensor including `_keras_history`.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = constant_layer._inbound_nodes[0].output_tensors
    return outputs[0] if len(outputs) == 1 else outputs
