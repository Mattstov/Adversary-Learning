import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import DefenderEnv, MONITOR, NOOP, RESTORE
from policies import NoDefense, RandomDefender, SweepDefender

KIND_NAMES = {NOOP: "NoOp", MONITOR: "Monitor", RESTORE: "Restore"}


def attack_name(env, a_idx):
    cls, params = env.sim.action_list[a_idx]
    return f"{cls.__name__} {params.get('name', '')}".strip()


def trace_episode(env, policy):
    obs = env.reset()
    policy.reset()
    sensitive = [env.hosts[i] for i in np.flatnonzero(obs[:, 1])]
    print(f"Network: {len(env.hosts)} hosts {env.hosts}")
    print(f"Sensitive hosts: {sensitive}\n")
    print(f"{'step':>4}  {'defender':<18} {'attacker':<40} {'reward':>6}  compromised")
    done, t, quiet = False, 0, 0
    while not done:
        action = policy.act(obs, env.host_mask())
        row, kind = int(action[0]), int(action[1])
        target = env.hosts[row] if row < len(env.hosts) else "(empty)"
        prev_breached = info["breached"] if t else 0
        obs, r, done, info = env.step(action)
        t += 1
        alert = kind == MONITOR and obs[row, 2] > 0
        if kind == RESTORE or alert or info["breached"] != prev_breached:
            if quiet:
                print(f"{'':>4}  ... {quiet} quiet steps ...")
                quiet = 0
            if env.last_attack:
                (s, h), a_idx = env.last_attack
                attack = attack_name(env, a_idx) + f" on ({s}, {h})"
            else:
                attack = "NoOp"
            note = "  <- ALERT" if alert else ""
            print(f"{t:>4}  {KIND_NAMES[kind] + ' ' + str(target):<18} "
                  f"{attack:<40} {r + 0.0:>6.2f}  {info['breached']}{note}")
        else:
            quiet += 1
    if quiet:
        print(f"{'':>4}  ... {quiet} quiet steps ...")


def evaluate(env, policy, episodes):
    curves, totals = [], []
    for _ in range(episodes):
        obs, done, curve, total = env.reset(), False, [], 0.0
        policy.reset()
        while not done:
            obs, r, done, info = env.step(policy.act(obs, env.host_mask()))
            curve.append(info["breached"])
            total += r
        curves.append(curve)
        totals.append(total)
    curves = np.array(curves)
    return curves.mean(axis=0), np.array(totals), curves.sum(axis=1)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    scenario_name = sys.argv[2] if len(sys.argv) > 2 else "sm_entry_dmz_two_subnets"
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    env = DefenderEnv(f"{folder}/{scenario_name}.v2.yaml")

    print("=" * 84)
    print(f"PART 1: one episode on {scenario_name}, Sweep defender vs random attacker")
    print("=" * 84)
    trace_episode(env, SweepDefender())

    print("\n" + "=" * 84)
    print(f"PART 2: {episodes} episodes x {env.max_steps} steps per defender")
    print("=" * 84)
    policies = {"No defense": NoDefense(), "Random defender": RandomDefender(),
                "Sweep defender": SweepDefender()}
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
