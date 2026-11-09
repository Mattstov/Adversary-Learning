import numpy as np

from .base import Policy, real_rows

NOOP, MONITOR, RESTORE = 0, 1, 2


class NoDefense(Policy):
    def act(self, obs, mask, deterministic=False):
        return np.array([0, NOOP])


class RandomDefender(Policy):
    def act(self, obs, mask, deterministic=False):
        return np.array([np.random.choice(real_rows(mask)), np.random.randint(3)])


class SweepDefender(Policy):
    def __init__(self):
        self.reset()

    def reset(self, info=None):
        self.next = 0

    def act(self, obs, mask, deterministic=False):
        alerted = np.flatnonzero(obs[:, 2])
        if len(alerted):
            return np.array([alerted[0], RESTORE])
        sensitive = np.flatnonzero(obs[:, 1]) if obs[:, 1].any() else real_rows(mask)
        row = sensitive[self.next % len(sensitive)]
        self.next += 1
        return np.array([row, MONITOR])
