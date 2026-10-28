import numpy as np

from nasimemu.nasim.envs.action import Exploit, OSScan, PrivilegeEscalation, ProcessScan, ServiceScan, SubnetScan
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from .base import Policy, real_rows


class RandomAttacker(Policy):
    def reset(self, info=None):
        self.n_kinds = len(info["action_list"])

    def act(self, obs, mask, deterministic=False):
        return np.array([np.random.choice(real_rows(mask)), np.random.randint(self.n_kinds)])


class GreedyAttacker(Policy):
    def reset(self, info=None):
        self.action_list = info["action_list"]
        self.kind_of = {}
        for kind, (cls, _) in enumerate(self.action_list):
            self.kind_of.setdefault(cls, kind)
        self.tried = set()

    def act(self, obs, mask, deterministic=False):
        rows = real_rows(mask)
        hosts = sorted(rows, key=lambda r: -HostVector(obs[r]).value)
        for row in hosts:
            for kind in self._candidates(HostVector(obs[row])):
                key = (HostVector(obs[row]).address, kind)
                if key not in self.tried:
                    self.tried.add(key)
                    return np.array([row, kind])
        return np.array([np.random.choice(rows), np.random.randint(len(self.action_list))])

    def _candidates(self, host):
        scan = self.kind_of
        if host.access >= AccessLevel.ROOT:
            return [scan[SubnetScan]]
        if host.compromised:
            return [scan[SubnetScan], scan[ProcessScan]] + self._matching(host, PrivilegeEscalation)
        if not host.reachable:
            return []
        return [scan[ServiceScan], scan[OSScan]] + self._matching(host, Exploit)

    def _matching(self, host, cls):
        os, services, processes = host.os, host.services, host.processes
        kinds = []
        for kind, (c, params) in enumerate(self.action_list):
            if c is not cls:
                continue
            if params.get("os") is not None and not os.get(params["os"]):
                continue
            if cls is Exploit and not services.get(params["service"]):
                continue
            if cls is PrivilegeEscalation and params.get("process") is not None \
                    and not processes.get(params["process"]):
                continue
            kinds.append(kind)
        return kinds
