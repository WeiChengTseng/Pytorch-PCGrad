import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers, nonlin=F.relu):
        nn.Module.__init__(self)
        self._layers = nn.ModuleList()
        input_dim = int(np.prod(input_dim))
        last_dim = input_dim
        self._nonlin = nonlin
        for hidden_layer_dim in layers:
            self._layers.append(nn.Linear(last_dim, hidden_layer_dim))
            last_dim = hidden_layer_dim
        self._layers.append(nn.Linear(last_dim, output_dim))

    def forward(self, arg, *args):
        out = arg
        for i in range(len(self._layers) - 1):
            layer = self._layers[i]
            evaluated = layer(out)
            out = self._nonlin(evaluated)
        out = self._layers[-1](out)
        return out

    __call__ = forward


class SampleMetaInformation(object):
    """
    Class SampleMetaInformation should be used to store metainformation for each sample.
    """
    def __init__(self, task=None):
        self.task = task
        self.steps = []

    def append(self, attr_name, obj, new_step=False):
        if new_step:
            self.steps.append({})
        else:
            assert len(
                self.steps
            ) > 0, 'initialize a new step first by calling this function with new_step=True'
        self.steps[-1][attr_name] = obj

    def finalize(self):
        """
        This method finalizes a trajectory, by translating the stored sar tuples into attributes of this class
        :return:
        """
        res = {}
        for step in self.steps:
            for key in step.keys():
                res[key] = []
        for i, step in enumerate(self.steps):
            for key in res.keys():
                if key not in step:
                    res[key].append(None)
                else:
                    res[key].append(step[key])
        for key, val in res.items():
            setattr(self, key, val)