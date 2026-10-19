import numpy as np


class Policy:
    def reset(self, info=None):
        pass

    def act(self, obs, mask, deterministic=False):
        raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass


def real_rows(mask):
    return np.flatnonzero(mask)
