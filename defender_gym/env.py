import gym
import numpy as np
from gym import spaces

from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.action import NoOp
from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from policies import RandomAttacker, SweepDefender

NOOP, MONITOR, RESTORE = 0, 1, 2
N_FEATURES = 4
DEFENDER_COSTS = {NOOP: 0.0, MONITOR: 0.01, RESTORE: 0.1}
ATTACKER_COST = 0.01


class DefenderEnv(gym.Env):
    def __init__(self, scenario_path, attacker=None, max_hosts=50, max_steps=100,
                 defender_costs=None, attacker_cost=ATTACKER_COST):
        self.max_hosts = max_hosts
        self.max_steps = max_steps
        self.defender_costs = defender_costs if defender_costs is not None else DEFENDER_COSTS
        self.attacker_cost = attacker_cost
        self.attacker = attacker if attacker is not None else RandomAttacker()
        self.sim = NASimEmuEnv(scenario_path, step_limit=None)

        self.observation_space = spaces.Box(0.0, 1.0, (max_hosts, N_FEATURES), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([max_hosts, 3])
        self.reset()

    def reset(self):
        self.new_game()
        return self._obs()

    def new_game(self):
        self.attacker_obs = self.sim.reset()
        self.hosts = [tuple(int(x) for x in a) for a in self.sim.host_index]
        assert len(self.hosts) <= self.max_hosts, "raise max_hosts"

        self.alert = np.zeros(self.max_hosts, dtype=np.float32)
        self.restored = np.zeros(self.max_hosts, dtype=np.float32)
        self.steps = 0
        self.last_attack = None
        self.attacker_view()
        self.attacker.reset({"action_list": self.sim.action_list})

    def step(self, action):
        att_obs, att_mask = self.attacker_view()
        def_r, _, done, info = self.play(action, self.attacker.act(att_obs, att_mask))
        return self._obs(), def_r, done, info

    def play(self, def_action, att_action):
        row, kind = int(def_action[0]), int(def_action[1])

        if row < len(self.hosts):
            host = self._host(self.hosts[row])
            if kind == MONITOR:
                self.alert[row] = float(host.compromised)
            elif kind == RESTORE:
                host.compromised = False
                host.access = 0
                self.alert[row] = 0.0
                self.restored[row] = 1.0

        att_row, att_kind = int(att_action[0]), int(att_action[1])
        att_cost = 0.0
        if att_row < len(self.attacker_rows):
            self.last_attack = (self.attacker_rows[att_row], att_kind)
            self.attacker_obs, _, _, _ = self.sim.step(self.last_attack)
            att_cost = self.attacker_cost * self.sim.action_list[att_kind][1]["cost"]
        else:
            self.last_attack = None
            self.attacker_obs, _, _, _ = self.sim.step(NoOp())

        exposure = self._exposure()
        def_r = -exposure - self.defender_costs[kind]
        att_r = exposure - att_cost
        self.steps += 1
        done = self.steps >= self.max_steps
        return def_r, att_r, done, {"breached": len(self._sensitive_compromised()),
                                    "exposure": exposure}

    def attacker_view(self):
        rows = [r for r in self.attacker_obs[:-1] if HostVector(r).address != (0, 0)]
        self.attacker_rows = [tuple(int(x) for x in HostVector(r).address) for r in rows]
        assert len(rows) <= self.max_hosts, "raise max_hosts"
        obs = np.zeros((self.max_hosts, self.attacker_obs.shape[1]), dtype=np.float32)
        obs[:len(rows)] = rows
        return obs, np.arange(self.max_hosts) < len(rows)

    def host_mask(self):
        return np.arange(self.max_hosts) < len(self.hosts)

    def action_masks(self):
        return np.concatenate([self.host_mask(), np.ones(3, dtype=bool)])

    def _host(self, addr):
        return self.sim.env.current_state.get_host(addr)

    def _sensitive_compromised(self):
        return {a for a in self.hosts if self._host(a).value > 0 and self._host(a).compromised}

    def _exposure(self):
        values = [(self._host(a).value, self._host(a).access) for a in self.hosts]
        total = sum(v for v, _ in values if v > 0)
        if total == 0:
            return 0.0
        return float(sum(v * acc / AccessLevel.ROOT for v, acc in values if v > 0) / total)

    def _obs(self):
        obs = np.zeros((self.max_hosts, N_FEATURES), dtype=np.float32)
        for i, addr in enumerate(self.hosts):
            obs[i, 0] = 1.0
            obs[i, 1] = float(self._host(addr).value > 0)
        obs[:, 2] = self.alert
        obs[:, 3] = self.restored
        return obs


class AttackerEnv(gym.Env):
    def __init__(self, scenario_path, defender=None, max_hosts=50, max_steps=100,
                 defender_costs=None, attacker_cost=ATTACKER_COST):
        self.defender = defender if defender is not None else SweepDefender()
        self.game = DefenderEnv(scenario_path, max_hosts=max_hosts, max_steps=max_steps,
                                defender_costs=defender_costs, attacker_cost=attacker_cost)
        self.reset()
        n_features = self.attacker_obs.shape[1]
        self.observation_space = spaces.Box(0.0, np.inf, (max_hosts, n_features), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([max_hosts, len(self.game.sim.action_list)])

    @property
    def sim(self):
        return self.game.sim

    def reset(self):
        self.game.new_game()
        self.defender.reset()
        self.attacker_obs, self.mask = self.game.attacker_view()
        return self.attacker_obs

    def step(self, action):
        def_action = self.defender.act(self.game._obs(), self.game.host_mask())
        _, att_r, done, info = self.game.play(def_action, action)
        self.attacker_obs, self.mask = self.game.attacker_view()
        return self.attacker_obs, att_r, done, info

    def host_mask(self):
        return self.mask

    def action_masks(self):
        return np.concatenate([self.mask, np.ones(self.action_space.nvec[1], dtype=bool)])


if __name__ == "__main__":
    import sys
    from gym.utils.env_checker import check_env

    folder = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    small = f"{folder}/sm_entry_dmz_one_subnet.v2.yaml"
    medium = f"{folder}/md_entry_dmz_one_subnet.v2.yaml"

    for name, make in [("DefenderEnv", DefenderEnv), ("AttackerEnv", AttackerEnv)]:
        env = make(small)
        check_env(env)
        print(f"{name} 1. gym checker passed")

        obs = env.reset()
        assert env.observation_space.contains(obs)
        print(f"{name} 2. reset passed")

        for _ in range(100):
            action = env.action_space.sample()
            action[0] = np.random.choice(np.flatnonzero(env.host_mask()))
            obs, r, done, info = env.step(action)
            assert env.observation_space.contains(obs)
            if done:
                obs = env.reset()
        print(f"{name} 3. 100 random steps passed")

        widths = {make(p).reset().shape[1] for p in (small, medium)}
        assert len(widths) == 1
        print(f"{name} 4. per-host feature width matches across sizes:", widths)
