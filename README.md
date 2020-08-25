# PCGrad

This repository provide code of reimplementation for [Gradient Surgery for Multi-Task Learning](https://arxiv.org/pdf/2001.06782.pdf) in PyTorch 1.6.0.


## Usage

```python
import torch
import torch.nn as nn
import torch.optim as optim
from pcgrad import PCGrad

# wrap your favorite optimizer
optimizer = PCGrad(optim.Adam(net.parameters())) 
losses = # a list of per-task losses
assert len(losses) == num_tasks
optimizer.pc_backward(losses) # calculate the gradient can apply gradient modification
optimizer.step()  # apply gradient step
```

## Reference

Please cite as:

```
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={arXiv preprint arXiv:2001.06782},
  year={2020}
}
```