"""
:Date: Nov 18, 2019
:Version: 0.1.0
"""

from tensorflow.python.keras import layers as KL
from tensorflow.python.util import nest


class LtnLayer(KL.Layer):

    def __init__(self, *args, **kwargs):
        doms = kwargs.get('doms')

        super(LtnLayer, self).__init__(*args, **kwargs)

        self._ltn_doms = doms if doms is not None else []

    def __call__(self, inputs, *args, **kwargs):
        outputs = super(LtnLayer, self).__call__(inputs, *args, **kwargs)
        flat_outputs = nest.flatten(outputs)

        self._ltn_doms = self.compute_doms(inputs, **kwargs)

        for output in flat_outputs:
            output._ltn_doms = tuple(self._ltn_doms)

        return outputs

    def compute_doms(self, inputs, **kwargs):
        return self.doms

    @property
    def doms(self):
        return self._ltn_doms
