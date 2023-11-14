from collections import deque

import numpy as np
import random


class PrioritizedGroupReplayBuffer(object):
    def __init__(self, size, num_agents, alpha=0.6, beta=0.4, beta_annealing_steps=1000):
        self.storage = [[] for _ in range(num_agents)]
        self.priorities = [deque(maxlen=size) for _ in range(num_agents)]
        self.maxsize = int(size)
        self.next_idx = [0] * num_agents
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.steps = 0

    def __len__(self):
        return min(len(agent_buffer) for agent_buffer in self.storage)

    def clear(self):
        self.storage = [[] for _ in range(len(self.storage))]
        self.priorities = [deque(maxlen=self.maxsize) for _ in range(len(self.storage))]
        self.next_idx = [0] * len(self.storage)

    def add(self, o, a, r, o_, agent_idx):
        data = (o, a, r, o_)
        priority = 1.0  # Initial priority
        if len(self.priorities[agent_idx]) > 0:
            priority = max(self.priorities[agent_idx])
        self.priorities[agent_idx].append(priority)

        if self.next_idx[agent_idx] >= len(self.storage[agent_idx]):
            self.storage[agent_idx].append(data)
        else:
            self.storage[agent_idx][self.next_idx[agent_idx]] = data
        self.next_idx[agent_idx] = (self.next_idx[agent_idx] + 1) % self.maxsize

    def update_priority(self, errors, agent_idx):
        for i, error in enumerate(errors):
            self.priorities[agent_idx][i] = abs(error) + 1e-6  # Avoid zero priority

    def calculate_weights(self):
        total_priorities = np.sum([np.array(agent_priorities) ** self.alpha for agent_priorities in self.priorities],
                                  axis=0)
        weights = ((1 / total_priorities) ** self.beta).tolist()
        weights /= np.max(weights)  # Normalize weights
        return weights

    def encode_sample(self, idxes, agent_dix):
        observations, actions, rewards, observations_ = [], [], [], []
        for i in idxes:
            data = self.storage[agent_dix][i]
            obs, act, rew, obs_ = data
            observations.append(np.concatenate(obs[:]))
            actions.append(act)
            rewards.append(rew)
            observations_.append(np.concatenate(obs_[:]))
        return np.array(observations), np.array(actions), np.array(rewards), np.array(observations_)

    def make_index(self, batch_size):
        priorities = [np.array(agent_priorities) ** self.alpha for agent_priorities in self.priorities]
        total_priorities = np.sum(priorities, axis=0)
        probabilities = total_priorities / np.sum(total_priorities)
        idxes = np.random.choice(len(self.storage[0]), size=batch_size, p=probabilities)
        return idxes

    def sample(self, batch_size, agent_dix):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, min(len(self.storage[agent_dix]), self.maxsize))

        weights = self.calculate_weights()
        sampled_weights = [weights[i] for i in idxes]

        # Update beta parameter for prioritized experience replay
        self.beta = min(1.0, self.beta + (1.0 - self.beta) / self.beta_annealing_steps * self.steps)
        self.steps += 1

        return self.encode_sample(idxes, agent_dix), idxes, sampled_weights









class ReplayBuffer(object):
    def __init__(self, size):
        self.storage = []
        self.maxsize = int(size)
        self.next_idx = 0

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage = []
        self.next_idx = 0

    def add(self, o, a, r, o_):
        data = (o, a, r, o_)

        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.maxsize

    def encode_sample(self, idxes):
        observations, actions, rewards, observations_ = [], [], [], []
        for i in idxes:
            data = self.storage[i]
            obs, act, rew, obs_ = data
            observations.append(np.concatenate(obs[:]))
            actions.append(act)
            rewards.append(rew)
            observations_.append(np.concatenate(obs_[:]))
        return np.array(observations), np.array(actions), np.array(rewards), np.array(observations_)

    def make_index(self, batch_size):
        return [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]

    def sample(self, batch_size):

        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self.storage))
        return self.encode_sample(idxes)\




