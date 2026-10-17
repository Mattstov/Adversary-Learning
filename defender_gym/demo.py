"""Progress demo for the defender gym.

Part 1 prints a step-by-step trace of one episode.
Part 2 compares three simple defenders over many episodes and saves a chart.

Run with the scenarios folder path as the argument:
    python demo.py ../NASimEmu-master/scenarios/ [scenario_name] [episodes]

The default scenario has sensitive hosts. The *_one_subnet scenarios used by
env.py's checks have none, so every reward there is 0.
"""

import sys

import matplotlib
matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt
import numpy as np

from env import DefenderEnv, MONITOR, NOOP, RESTORE

KIND_NAMES = {NOOP: "NoOp", MONITOR: "Monitor", RESTORE: "Restore"}


# ---- Three baseline defenders. Each takes (env, obs) and returns an action. ----

def no_defense(env, obs):
    return np.array([0, NOOP])


def random_defense(env, obs):
    return env.action_space.sample()


class SweepDefense:
    """Restore any host with an alert; otherwise Monitor sensitive hosts in turn
    (all hosts if the network has no sensitive ones)."""

    def __init__(self):
        self.next = 0

    def __call__(self, env, obs):
        alerted = np.flatnonzero(obs[:, 2])
        if len(alerted):
            return np.array([alerted[0], RESTORE])
        sensitive = np.flatnonzero(obs[:, 1]) if obs[:, 1].any() else np.flatnonzero(obs[:, 0])
        row = sensitive[self.next % len(sensitive)]
        self.next += 1
        return np.array([row, MONITOR])


def attack_name(env, a_idx):
    # e.g. "Exploit e_drupal" or "SubnetScan"
    cls, params = env.sim.action_list[a_idx]
    return f"{cls.__name__} {params.get('name', '')}".strip()




def trace_episode(env, policy):
    # Print one full episode, showing only the steps where something happened:
    # a sensitive host was breached, Monitor raised an alert, or a Restore ran.
    # Wrap the attacker so we can see which move it picked.
    pick_attack = env._random_attack
    def recorded():
        env.last_attack = pick_attack()
        return env.last_attack
    env._random_attack = recorded

    obs = env.reset()
    sensitive = [env.hosts[i] for i in np.flatnonzero(obs[:, 1])]
    print(f"Network: {len(env.hosts)} hosts {env.hosts}")
    print(f"Sensitive hosts: {sensitive}\n")
    print(f"{'step':>4}  {'defender':<18} {'attacker':<40} {'reward':>6}  compromised")
    done, t, quiet = False, 0, 0
    while not done:
        action = policy(env, obs)
        row, kind = int(action[0]), int(action[1])
        target = env.hosts[row] if row < len(env.hosts) else "(empty)"
        prev_breached = info["breached"] if t else 0
        obs, r, done, info = env.step(action)
        t += 1
        alert = kind == MONITOR and obs[row, 2] > 0
        if r or kind == RESTORE or alert or info["breached"] != prev_breached:
            if quiet:
                print(f"{'':>4}  ... {quiet} quiet steps ...")
                quiet = 0
            (s, h), a_idx = env.last_attack
            note = "  <- ALERT" if alert else ""
            print(f"{t:>4}  {KIND_NAMES[kind] + ' ' + str(target):<18} "
                  f"{attack_name(env, a_idx) + f' on ({s}, {h})':<40} {r + 0.0:>6.0f}  {info['breached']}{note}")
        else:
            quiet += 1
    if quiet:
        print(f"{'':>4}  ... {quiet} quiet steps ...")
    env._random_attack = pick_attack


def evaluate(env, policy, episodes):
    # Run episodes. Record per step how many sensitive hosts are compromised,
    # and per episode the total reward.
    curves, totals = [], []
    for _ in range(episodes):
        obs, done, curve, total = env.reset(), False, [], 0.0
        while not done:
            obs, r, done, info = env.step(policy(env, obs))
            curve.append(info["breached"])
            total += r
        curves.append(curve)
        totals.append(total)
    curves = np.array(curves)
    # exposure = sensitive-host-steps spent compromised per episode
    return curves.mean(axis=0), np.array(totals), curves.sum(axis=1)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    scenario_name = sys.argv[2] if len(sys.argv) > 2 else "sm_entry_dmz_two_subnets"
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    env = DefenderEnv(f"{folder}/{scenario_name}.v2.yaml")

    print("=" * 84)
    print(f"PART 1: one episode on {scenario_name}, Sweep defender vs random attacker")
    print("=" * 84)
    trace_episode(env, SweepDefense())

    print("\n" + "=" * 84)
    print(f"PART 2: {episodes} episodes x {env.max_steps} steps per defender")
    print("=" * 84)
    policies = {"No defense": no_defense, "Random defender": random_defense,
                "Sweep defender": SweepDefense()}
    colors = ["#c44", "#888", "#48c"]
    print(f"{'defender':<18} {'reward (mean +- std)':>22} {'steps compromised':>19}")
    curves, rewards, exposure = {}, {}, {}
    for label, policy in policies.items():
        curves[label], rewards[label], exposure[label] = evaluate(env, policy, episodes)
        print(f"{label:<18} {rewards[label].mean():>13.2f} +- {rewards[label].std():<5.2f}"
              f"{exposure[label].mean():>19.1f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))
    for (label, curve), c in zip(curves.items(), colors):
        ax1.plot(curve, label=label, color=c)
    ax1.set_title("Sensitive hosts compromised (mean per step)")
    ax1.set_xlabel("step")
    ax1.set_ylabel("sensitive hosts compromised")
    ax1.legend()

    def bars(ax, data, title, ylabel):
        ax.bar(data.keys(), [d.mean() for d in data.values()],
               yerr=[d.std() for d in data.values()], capsize=6, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    bars(ax2, exposure, "Time compromised (lower is better)", "sensitive-host-steps compromised")
    bars(ax3, rewards, "Current env reward (higher is better)", "total reward")
    fig.suptitle(f"Defender gym on {scenario_name}: baseline defenders vs random attacker, "
                 f"{episodes} episodes")
    fig.tight_layout()
    fig.savefig("defender_baselines.png", dpi=150)
    print("\nSaved chart to defender_baselines.png")
