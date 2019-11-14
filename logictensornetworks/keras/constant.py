"""
A constant aka InputLayer with one weight ano no placeholders

:Date: Nov 14, 2019
:Version: 0.0.3
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras as K
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.InputLayer')
class ConstantLayer(base_layer.Layer):
    """Layer to be used as an entry point into a Network (a graph of layers).

  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create a placeholder tensor (pass arguments `input_shape`, and
  optionally, `dtype`).

  It is generally recommend to use the functional layer API via `Input`,
  (which creates an `InputLayer`) without directly using `InputLayer`.

  This class can create placeholders for tf.Tensors, tf.SparseTensors, and
  tf.RaggedTensors by choosing 'sparse=True' or 'ragged=True'.

  Arguments:
      input_shape: Shape tuple (not including the batch axis), or `TensorShape`
        instance (not including the batch axis).
      batch_size: Optional input batch size (integer or None).
      dtype: Datatype of the input.
      input_tensor: Optional tensor to use as layer input
          instead of creating a placeholder.
      sparse: Boolean, whether the placeholder created is meant to be sparse.
      ragged: Boolean, whether the placeholder created is meant to be ragged.
          In this case, values of 'None' in the 'shape' argument represent
          ragged dimensions. For more information about RaggedTensors, see
          https://www.tensorflow.org/guide/ragged_tensors.
      name: Name of the layer (string).
  """

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
        self.built = True
        # with graph.as_default():

        # Create an input node to add to self.outbound_node
        # and set output_tensors' _keras_history.
        fake_input_tensor = backend.constant(backend.get_value(self._c), name=name)
        fake_input_tensor._keras_history = base_layer.KerasHistory(self, 0, 0)
        fake_input_tensor._keras_mask = None
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
    '''
    TODO: check args
    if shape is None and tensor is None:
        raise ValueError('Please provide to Input either a `shape`'
                         ' or a `tensor` argument. Note that '
                         '`shape` does not include the batch '
                         'dimension.')
    '''

    constant_layer = ConstantLayer(label=label,
                                   min_value=min_value,
                                   max_value=max_value)

    # Return tensor including `_keras_history`.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = constant_layer._inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
