import torch
import torch.nn as nn

import abc

from collections import defaultdict

from .helpers import SampleMetaInformation
from .reward_fn import CorrectClassifiedReward


class Initialization(nn.Module):
    """
    The initialization class defines a thin layer that initializes the meta-information and actions - composing
    the pytorch-routing information triplet.
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, xs, tasks=()):
        if len(tasks) > 0:
            mxs = [SampleMetaInformation(task=t) for t in tasks]
        else:
            mxs = [SampleMetaInformation() for _ in xs]
        return xs, mxs, None


class Selection(nn.Module):
    """
    Class RoutingWrapperModule defines a wrapper around a regular pytorch module that computes the actual routing
    given a list of modules to choose from, and a list of actions to select a module for each sample in a batch.
    """
    def __init__(self, *modules, name='', store_module_pointers=False):
        nn.Module.__init__(self)
        self.name = name
        # self._threads = threads
        self._submodules = nn.ModuleList(modules)
        self._selection_log = []
        self._logging_selections = False
        self.__output_dim = None
        self._store_module_pointers = store_module_pointers

    def forward(self, xs, mxs, actions, mask=None):
        """
        This method takes a list of samples - a batch - and calls _forward_sample on each. Samples are
        a tensor where the first dimension is the batch dimension.
        :param xs:
        :param mxs:
        :param actions:
        :param mask: a torch.ByteTensor that determines if the trajectory is active. if it is not, no action
                               will be executed
        :return:
        """
        assert len(xs) == len(mxs)
        batch_size = xs.size(0)
        # capture the special case of just one submodule - and skip all computation
        if len(self._submodules) == 1:
            return self._submodules[0](xs), mxs, actions
        # retrieving output dim for output instantiation
        if self.__output_dim is None:
            self.__output_dim = self._submodules[0](
                xs[0].unsqueeze(0)).shape[1:]
        # initializing the "termination" mask
        mask = torch.ones(batch_size, dtype=torch.uint8, device=xs.device) \
            if mask is None else mask
        # parallelizing this loop does not work. however, we can split the batch by the actions
        # creating the target variable
        ys = torch.zeros((batch_size, *self.__output_dim),
                         dtype=torch.float,
                         device=xs.device)
        for i in torch.arange(actions.max() + 1, device=xs.device):
            if i not in actions:
                continue
            # computing the mask as the currently active action on the active trajectories
            m = ((actions == i) * mask)
            if not any(m):
                continue
            ys[m] = self._submodules[i](xs[m])
        if self._logging_selections:
            self._selection_log += actions.reshape(-1).cpu().tolist()
        if self._store_module_pointers:
            for mx, a in zip(mxs, actions):
                mx.append('selected_modules', self._submodules[a])
        return ys, mxs, actions

    def start_logging_selections(self):
        self._logging_selections = True

    def stop_logging_and_get_selections(self, add_to_old=False):
        self._logging_selections = False
        logs = list(set([int(s) for s in self._selection_log]))
        del self._selection_log[:]
        self.last_selection_freeze = logs + self.last_selection_freeze if add_to_old else logs
        return self.last_selection_freeze


class Loss(nn.Module, metaclass=abc.ABCMeta):
    """
    This function defines the combined module/decision loss functions. It performs four steps that will result in
    separate losses for the modules and the decision makers:
    1. it computes the module losses
    2. it translates these module losses into per-sample reinforcement learning rewards
    3. it uses these final rewards to compute the full rl-trajectories for each sample
    4. it uses the decision-making specific loss functions to compute the total decision making loss
    """
    def __init__(self,
                 pytorch_loss_func,
                 routing_reward_func,
                 discounting=1.,
                 clear=False,
                 normalize_per_action_rewards=True):
        nn.Module.__init__(self)
        self._discounting = discounting
        self._loss_func = pytorch_loss_func
        self._clear = clear
        try:
            self._loss_func.reduction = 'none'
        except AttributeError:
            pass
        self._reward_func = routing_reward_func
        self._npar = normalize_per_action_rewards

    def _get_rl_loss_tuple_map(self, mys, device):
        rl_loss_tuple_map = defaultdict(lambda: defaultdict(list))
        reward_functions = set()
        for traj_counter, my in zip(
                torch.arange(len(mys), device=device).unsqueeze(1), mys):
            my.finalize(
            )  # translates the trajectory from a list of obj into lists
            my.add_rewards = [ar if ar is not None else 0. for ar in my.add_rewards] \
                if hasattr(my, 'add_rewards') else [0.] * len(my.actions)
            assert len(my.actions) == len(my.states) == len(
                my.add_rewards) == len(my.reward_func)
            rewards = []
            # computing the rewards
            for state, action, reward_func, add_r in zip(
                    my.states, my.actions, my.reward_func, my.add_rewards):
                # normalize the per-action reward to the entire sequence length
                per_action_reward = (reward_func.get_reward(state, action) +
                                     add_r) / len(my.actions)
                # normalize to the final reward, s.t. it will be interpreted as a fraction thereof
                per_action_reward = per_action_reward * torch.abs(
                    my.final_reward) if self._npar else per_action_reward
                rewards.append(per_action_reward)
                reward_functions.add(reward_func)
            rewards[-1] += my.final_reward
            returns = [0.]
            # computing the returns
            for i, rew in enumerate(reversed(rewards)):
                returns.insert(0, rew + returns[0])
            returns = returns[:-1]
            # creating the tensors to compute the loss from the SARSA tuple
            for lf, s, a, rew, ret, pa, ns, na in zip(
                    my.loss_funcs, my.states, my.actions, rewards, returns,
                ([None] + my.actions)[:-1], (my.states + [None])[1:],
                (my.actions + [None])[1:]):
                is_terminal = ns is None or s.numel() != ns.numel()
                rl_loss_tuple_map[lf]['indices'].append(traj_counter)
                rl_loss_tuple_map[lf]['is_terminal'].append(
                    torch.tensor([is_terminal],
                                 dtype=torch.uint8,
                                 device=device))
                rl_loss_tuple_map[lf]['states'].append(s)
                rl_loss_tuple_map[lf]['actions'].append(a.view(-1))
                rl_loss_tuple_map[lf]['rewards'].append(rew.view(-1))
                rl_loss_tuple_map[lf]['returns'].append(ret.view(-1))
                rl_loss_tuple_map[lf]['final_reward'].append(
                    my.final_reward.view(-1))
                rl_loss_tuple_map[lf]['prev_actions'].append(
                    a.new_zeros(1) if pa is None else pa.view(-1))
                rl_loss_tuple_map[lf]['next_states'].append(
                    s if is_terminal else ns)
                rl_loss_tuple_map[lf]['next_actions'].append(
                    a.new_zeros(1) if is_terminal else na.view(-1))
        # concatenating the retrieved values into tensors
        for k0, v0 in rl_loss_tuple_map.items():
            for k1, v1 in v0.items():
                v0[k1] = torch.cat(v1, dim=0)
        if self._clear:
            for rf in reward_functions:
                rf.clear()
        return rl_loss_tuple_map

    def forward(self,
                ysest,
                mys,
                ystrue=None,
                external_losses=None,
                reduce=True):
        assert not(ystrue is None and external_losses is None), \
            'Must provide ystrue and possibly external_losses (or both).'
        batch_size = ysest.size(0)
        if external_losses is not None:
            # first case: external losses are provided externally
            assert external_losses.size()[0] == len(
                mys), 'One loss value per sample is required.'
            module_loss = external_losses.view(external_losses.size()[0],
                                               -1).sum(dim=1)
        else:
            # second case: they are not, so we need to compute them
            module_loss = self._loss_func(ysest, ystrue)
            if len(module_loss.size()) > 1:
                module_loss = module_loss.sum(dim=1).reshape(-1)
        if ystrue is None:
            # more input checking
            assert not isinstance(self._reward_func, CorrectClassifiedReward), \
                'Must provide ystrue when using CorrectClassifiedReward'
            ystrue = ysest.new_zeros(batch_size)
        assert len(module_loss) == len(mys) == len(ysest) == len(ystrue), \
            'Losses, metas, predictions and targets need to have the same length ({}, {}, {}, {})'.format(
                len(module_loss), len(mys), len(ysest), len(ystrue))
        # add the final reward, as we can only compute them now that we have the external feedback
        for l, my, yest, ytrue in zip(module_loss.split(1, dim=0), mys,
                                      ysest.split(1, dim=0),
                                      ystrue.split(1, dim=0)):
            my.final_reward = self._reward_func(l, yest, ytrue)
        # retrieve the SARSA pairs to compute the respective decision making losses
        rl_loss_tuple_map = self._get_rl_loss_tuple_map(mys,
                                                        device=ysest.device)
        # initialize the rl loss
        routing_loss = torch.zeros(batch_size,
                                   dtype=torch.float,
                                   device=ysest.device)
        for loss_func, rl_dict in rl_loss_tuple_map.items():
            # batch the RL loss by loss function, if possible
            rl_losses = loss_func(rl_dict['is_terminal'], rl_dict['states'],
                                  rl_dict['next_states'], rl_dict['actions'],
                                  rl_dict['next_actions'], rl_dict['rewards'],
                                  rl_dict['returns'], rl_dict['final_reward'])
            for i in torch.arange(batch_size, device=ysest.device):
                # map the losses back onto the sample indices
                routing_loss[i] = routing_loss[i] + torch.sum(
                    rl_losses[rl_dict['indices'] == i])
        if reduce:
            module_loss = module_loss.mean()
            routing_loss = routing_loss.mean()
        return module_loss, routing_loss