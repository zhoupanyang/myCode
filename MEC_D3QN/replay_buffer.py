import numpy as np
import random



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

