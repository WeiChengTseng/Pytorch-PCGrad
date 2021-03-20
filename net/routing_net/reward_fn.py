import abc
import torch
import torch.nn as nn
from collections import deque


class BaseReward(nn.Module, metaclass=abc.ABCMeta):
    """
    Class BaseReward defines the base function for all final reward functions.
    """
    def __init__(self, scale=1.):
        nn.Module.__init__(self)
        self._scale = scale

    @abc.abstractmethod
    def forward(self, loss, yest, ytrue):
        pass


class NegLossReward(BaseReward):
    """
    Class NegLossReward defines the simplest reward function, expressed as the negative loss.
    """
    def __init__(self, *args, **kwargs):
        BaseReward.__init__(self, *args, **kwargs)

    def forward(self, loss, yest, ytrue):
        with torch.no_grad():
            reward = -loss.squeeze()
        return reward


class PerActionBaseReward(object, metaclass=abc.ABCMeta):
    """
    Class BaseReward defines the base class for per-action rewards.
    """
    def __init__(self, history_window=256, *args, **kwargs):
        self._hist_len = history_window
        self._dists = deque(maxlen=history_window)
        self._actions = deque(maxlen=history_window)
        self._precomp = None

    def register(self, dist, action):
        self._dists.append(dist.detach())
        self._actions.append(action.detach())

    def clear(self):
        self._dists = deque(maxlen=self._hist_len)
        self._actions = deque(maxlen=self._hist_len)
        self._precomp = None

    def get_reward(self, dist, action):
        return torch.FloatTensor([0.]).to(action.device)


class CollaborationReward(PerActionBaseReward):
    """
    Class CollaborationReward defines a collaboration reward measured by the average probability
    of taking the action taken by an agent.
    """
    def __init__(self, reward_ratio=0.1, num_actions=None, history_len=256):
        PerActionBaseReward.__init__(self, history_len)
        self._reward_ratio = reward_ratio
        self._num_actions = num_actions

    def get_reward(self, dist, action):
        action_count = torch.zeros(len(self._actions),
                                   self._num_actions).to(dist.device)
        action_count = action_count.scatter(
            1,
            torch.stack(list(self._actions), 0).unsqueeze(1), 1.)
        action_count = torch.sum(action_count, dim=0) / len(self._actions)
        self._precomp = action_count
        self._precomp = self._reward_ratio * self._precomp
        return self._precomp[action] * self._reward_ratio


class CorrectClassifiedReward(BaseReward):
    """
    Class CorrectClassifiedReward defines the +1 reward for correct classification, and -1 otherwise.
    """
    def __init__(self, *args, **kwargs):
        BaseReward.__init__(self, *args, **kwargs)

    def forward(self, loss, yest, ytrue):
        # input checking - onehot vs indices
        if yest.numel() == yest.size(0):
            y_ind = yest
        else:
            _, y_ind = yest.max(dim=1)
        if ytrue.numel() == ytrue.size(0):
            yt_ind = ytrue
        else:
            _, yt_ind = ytrue.max(dim=1)
        return -1. + 2. * (y_ind.squeeze() == yt_ind.squeeze()).float()