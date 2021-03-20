import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pcgrad import PCGrad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


def example(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = TestNet().to(rank)
    # model = nn.Linear(3, 4).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    optimizer = PCGrad(optim.SGD(ddp_model.parameters(), lr=0.001))

    outputs = ddp_model(torch.randn(2, 3).to(rank))
    labels = torch.randn(2, 4).to(rank)

    loss1, loss2 = loss1_fn(outputs, labels), loss2_fn(outputs, labels)
    losses =  [loss1, loss2]
    optimizer.pc_backward(losses)
    optimizer.step()



if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)

    world_size = 2
    mp.spawn(example, args=(world_size, ), nprocs=world_size, join=True)
