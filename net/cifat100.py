import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = nn.Sequential(nn.Conv2d(3, 160, 3, 3), nn.ReLU(),
                                  nn.Conv2d(160, 160, 3, 3), nn.ReLU(),
                                  nn.Conv2d(160, 160, 3, 3), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(100, 300), nn.ReLU(),
                                  nn.Linear(300, 100))
        return

    def forward(self, img):
        self._net(img)
        return self._net(img)