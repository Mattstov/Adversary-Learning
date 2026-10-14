"""Minimal defender gym wrapper around NASimEmu (gym 0.21 API).

Each step: the defender acts first, then a placeholder attacker acts.
The attacker is a random stub for now; swap in the frozen PPO attacker later.

Observation: (max_hosts, 4) per-host features, zero-padded.
    0 present   1 if this row is a real host
    1 sensitive 1 if the host holds a sensitive service (value > 0)
    2 alert     1 if the last Monitor on this host found it compromised
    3 restored  1 if the defender has restored this host this episode
Action: (host_row, kind) with kind 0 = NoOp, 1 = Monitor, 2 = Restore.
Actions on padded rows do nothing.
Reward (placeholder, raw breach): -1 per sensitive host newly compromised.
"""

import random

import gym
import numpy as np
from gym import spaces

from nasimemu.env import NASimEmuEnv
from nasimemu.nasim.envs.host_vector import HostVector

# The three defender action kinds (the second number in an action)
NOOP, MONITOR, RESTORE = 0, 1, 2
# Number of features per host in the observation
N_FEATURES = 4


class DefenderEnv(gym.Env):
    def __init__(self, scenario_path, max_hosts=50, max_steps=100):
        self.max_hosts = max_hosts  # observation is padded to this many rows
        self.max_steps = max_steps  # episode length
        # The underlying NASimEmu simulator. It is built from the scenario file
        # and is rebuilt on every reset, so host counts can change between episodes.
        self.sim = NASimEmuEnv(scenario_path, step_limit=None)

        # Observation: one row per host, one column per feature, all values 0 or 1.
        self.observation_space = spaces.Box(0.0, 1.0, (max_hosts, N_FEATURES), dtype=np.float32)
        # Action: [which host row, which kind (NoOp / Monitor / Restore)].
        self.action_space = spaces.MultiDiscrete([max_hosts, 3])
        self.reset()  # gym 0.21's checker may call step() before reset()

    def reset(self):
        # Start a new episode: rebuild the network and get the attacker's first view.
        self.attacker_obs = self.sim.reset()
        # Host addresses (subnet, host) in a fixed order. Row i of the observation is hosts[i].
        self.hosts = [tuple(int(x) for x in a) for a in self.sim.host_index]
        assert len(self.hosts) <= self.max_hosts, "raise max_hosts"

        # Defender memory: what Monitor found, and which hosts were restored.
        self.alert = np.zeros(self.max_hosts, dtype=np.float32)
        self.restored = np.zeros(self.max_hosts, dtype=np.float32)
        self.steps = 0
        return self._obs()

    def step(self, action):
        row, kind = int(action[0]), int(action[1])

        # 1. Defender acts first.
        if row < len(self.hosts):  # actions on padded (empty) rows do nothing
            host = self._host(self.hosts[row])
            if kind == MONITOR:
                # Look at the host: the alert shows whether it is compromised right now.
                self.alert[row] = float(host.compromised)
            elif kind == RESTORE:
                # Clean the host: remove the attacker's foothold and access level.
                host.compromised = False
                host.access = 0
                self.alert[row] = 0.0
                self.restored[row] = 1.0

        # 2. Attacker acts (random placeholder). Compare sensitive hosts
        #    compromised before and after to see if this move caused a breach.
        before = self._sensitive_compromised()
        self.attacker_obs, _, _, _ = self.sim.step(self._random_attack())
        after = self._sensitive_compromised()

        # 3. Reward is -1 for each sensitive host newly breached this step.
        reward = -float(len(after - before))
        self.steps += 1
        done = self.steps >= self.max_steps
        # info reports how many sensitive hosts are compromised right now
        return self._obs(), reward, done, {"breached": len(after)}

    def _host(self, addr):
        # Look up one host's live state in the simulator by its (subnet, host) address.
        return self.sim.env.current_state.get_host(addr)

    def _sensitive_compromised(self):
        # Set of sensitive hosts (value > 0) that are currently compromised.
        # Used only for the reward; the defender never sees this directly.
        return {a for a in self.hosts if self._host(a).value > 0 and self._host(a).compromised}

    def _random_attack(self):
        # Pick a random attacker action on a host the attacker has already seen.
        # Each row of attacker_obs is a host vector; the last row is the previous action result.
        seen = [HostVector(r).address for r in self.attacker_obs[:-1]]
        seen = [a for a in seen if a != (0, 0)] or [self.hosts[0]]  # (0, 0) marks an empty entry
        return (random.choice(seen), random.randrange(len(self.sim.action_list)))

    def _obs(self):
        # Build the defender's observation: one row of features per host, zero-padded.
        obs = np.zeros((self.max_hosts, N_FEATURES), dtype=np.float32)
        for i, addr in enumerate(self.hosts):
            obs[i, 0] = 1.0                                   # present: this row is a real host
            obs[i, 1] = float(self._host(addr).value > 0)     # sensitive: host holds something valuable
        obs[:, 2] = self.alert                                # alert: last Monitor found it compromised
        obs[:, 3] = self.restored                             # restored: defender cleaned it this episode
        return obs


if __name__ == "__main__":
    # The four checks. Run with the scenarios folder path as the argument.
    import sys
    from gym.utils.env_checker import check_env

    folder = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    small = f"{folder}/sm_entry_dmz_one_subnet.v2.yaml"
    medium = f"{folder}/md_entry_dmz_one_subnet.v2.yaml"

    # Check 1: gym's built-in checker confirms the env follows the gym rules.
    env = DefenderEnv(small)
    check_env(env)
    print("1. gym checker passed")

    # Check 2: reset returns an observation that fits the observation space.
    obs = env.reset()
    assert env.observation_space.contains(obs)
    print("2. reset passed")

    # Check 3: 100 random defender actions run without crashing and stay in the space.
    for _ in range(100):
        obs, r, done, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if done:
            obs = env.reset()
    print("3. 100 random steps passed")

    # Check 4: the per-host feature width is the same on small and medium scenarios.
    widths = {DefenderEnv(p).reset().shape[1] for p in (small, medium)}
    assert widths == {N_FEATURES}
    print("4. per-host feature width matches across sizes:", widths)
