import copy
import torch
import torch.nn.functional as F

from .decision import Decision


class REINFORCE(Decision):
    """
    REINFORCE (likelihood ratio policy gradient) based decision making.
    """
    def _loss(self, is_terminal, state, next_state, action, next_action,
              reward, cum_return, final_reward):
        return -state[:, :, 0].gather(index=action.unsqueeze(1),
                                      dim=1).view(-1) * cum_return.detach()

    def set_exploration(self, exploration):
        if self._set_pg_temp:
            self._pg_temperature = 0.1 + 9.9 * exploration

    def _forward(self, xs, agent):
        policy = self._policy[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy /
                                                       self._pg_temperature)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        return xs, actions, self._eval_stochastic_are_exp(
            actions, distribution.logits), distribution.logits


class ActorCritic(REINFORCE):
    """
    ActorCritic based decision making.
    """
    def __init__(self, *args, qvalue_net=None, **kwargs):
        REINFORCE.__init__(self, *args, **kwargs)
        if qvalue_net is None and 'policy_net' in kwargs:
            qvalue_net = copy.deepcopy(kwargs['policy_net'])
        self._qvalue_mem = self._construct_policy_storage(
            self._num_selections, self._pol_type, qvalue_net,
            self._pol_hidden_dims)

    def _loss(self, is_terminal, state, next_state, action, next_action,
              reward, cum_return, final_reward):
        normalized_return = (cum_return - state[:, :, 1].gather(
            index=action.unsqueeze(1), dim=1).view(-1)).detach()
        act_loss = -state[:, :, 0].gather(index=action.unsqueeze(1),
                                          dim=1).view(-1) * normalized_return
        value_target = torch.where(is_terminal, final_reward,
                                   next_state[:, :, 1].max(dim=1)[0] -
                                   reward).detach()
        val_loss = F.mse_loss(state[:, :, 1].gather(index=action.unsqueeze(1),
                                                    dim=1).view(-1),
                              value_target,
                              reduction='none').view(-1)
        return act_loss + val_loss

    def _forward(self, xs, agent):
        policy = self._policy[agent](xs)
        values = self._qvalue_mem[agent](xs)
        distribution = torch.distributions.Categorical(logits=policy /
                                                       self._pg_temperature)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        state = torch.stack([distribution.logits, values], 2)
        return xs, actions, self._eval_stochastic_are_exp(actions,
                                                          state), state


class WPL(ActorCritic):
    """
    Weighted Policy Learner (WPL) Multi-Agent Reinforcement Learning based decision making.
    """
    def _loss(self, is_terminal, state, next_state, action, next_action,
              reward, cum_return, final_reward):
        grad_est = cum_return - state[:, :, 1].gather(
            index=action.unsqueeze(1), dim=1).view(-1)
        grad_projected = torch.where(grad_est < 0, 1. + grad_est,
                                     2. - grad_est)
        prob_taken = state[:, :, 0].gather(index=action.unsqueeze(1),
                                           dim=1).view(-1)
        prob_target = (prob_taken * grad_projected).detach()
        act_loss = F.mse_loss(prob_taken, prob_target, reduction='none')
        ret_loss = F.mse_loss(state[:, :, 1].gather(index=action.unsqueeze(1),
                                                    dim=1).view(-1),
                              cum_return.detach(),
                              reduction='none').view(-1)
        return act_loss + ret_loss

    def _forward(self, xs, agent):
        policy = self._policy[agent](xs)
        # policy = F.relu(policy) - F.relu(policy - 1.) + 1e-6
        policy = (policy.transpose(0, 1) - policy.min(dim=1)[0]).transpose(
            0, 1) + 1e-6
        # policy = policy/policy.sum(dim=1)
        values = self._qvalue_mem[agent](xs)
        distribution = torch.distributions.Categorical(probs=policy)
        if self.training:
            actions = distribution.sample()
        else:
            actions = distribution.logits.max(dim=1)[1]
        state = torch.stack([distribution.logits, values], 2)
        return xs, actions, self._eval_stochastic_are_exp(actions,
                                                          state), state
