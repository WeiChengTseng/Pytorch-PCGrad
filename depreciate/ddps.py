import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pcgrad import PCGrad


class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self._data = torch.rand(100, 3)
        self._label = torch.rand(100, 4)

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return self._data[idx], self._label[idx]


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


def example(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    model = TestNet().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    optimizer = PCGrad(optim.SGD(ddp_model.parameters(), lr=0.001))

    dataset = Dataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         sampler=sampler,
                                         batch_size=10,
                                         drop_last=True,
                                         num_workers=2,
                                         pin_memory=True)

    for ep in range(100):
        for data, label in loader:
            outputs = ddp_model(data.to(rank))
            label = label.to(rank)

            loss1, loss2 = loss1_fn(outputs, label), loss2_fn(outputs, label)
            losses = [loss1, loss2]
            optimizer.pc_backward(losses)
            optimizer.step()
    print('workder {}: toy training processing ends'.format(rank))


if __name__ == '__main__':
    torch.manual_seed(4)
    world_size = 2
    mp.spawn(example, args=(world_size, ), nprocs=world_size, join=True)
