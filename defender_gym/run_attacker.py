"""Isolated test of the attacker policies (Planned Work: "run isolated tests").

Plays each attacker policy for N episodes against a fixed defender, on the
small and the medium scenario, prints the average results, and saves a
chart to attacker_baselines.png.

Usage (from inside defender_gym/, with the venv active):
    python run_attacker.py ../NASimEmu-master/scenarios [episodes]

What a healthy result looks like:
    - greedy clearly beats random (it actually plans its attacks)
    - learned (untrained) scores about the same as random
    - learned never picks a padded (non-existent) host row
    - greedy does not collapse on medium compared to small

"success %" is the share of episodes where the attacker breached at least one
sensitive host. Against NoDefense it equals 100 minus the defender win rate
used in the proposal, so this run doubles as the proposal's NoOp baseline.
"""
import sys

import matplotlib
matplotlib.use("Agg")  # save to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

from env import AttackerEnv
from policies import GreedyAttacker, NoDefense, RandomAttacker

# The learned policy needs torch. If torch isn't installed, skip it
# instead of crashing, so the scripted attackers can still be tested.
try:
    from policies.learned import LearnedPolicy
except ImportError:
    LearnedPolicy = None

# Use the two_subnets scenarios: the one_subnet ones have no sensitive hosts,
# so exposure (and the attacker's reward) would always be 0.
SCENARIOS = {
    "small": "sm_entry_dmz_two_subnets",
    "medium": "md_entry_dmz_two_subnets",
}


def run_episodes(env, policy, episodes):
    """Play `episodes` full games with `policy` as the attacker.

    Returns:
      totals   - total attacker reward per episode
      breached - sensitive hosts compromised at the end of each episode
      padded   - how many times the policy picked a padded host row, per episode
      curve    - mean exposure at each step, averaged over all episodes
    """
    totals, breached, padded, curves = [], [], [], []
    for _ in range(episodes):
        obs = env.reset()
        # AttackerEnv.reset() does NOT reset the attacker policy we pass in,
        # so we do it here. GreedyAttacker needs the action list to know
        # which actions exist, and it clears its "already tried" memory.
        policy.reset({"action_list": env.sim.action_list})

        done, total, bad_picks, curve = False, 0.0, 0, []
        while not done:
            mask = env.host_mask()  # True for real hosts, False for padding
            action = policy.act(obs, mask)
            # action = [host row, action kind]. A row past the real hosts
            # is padding: the env turns it into a NoOp, but a working
            # policy should never choose it.
            if not mask[int(action[0])]:
                bad_picks += 1
            obs, r, done, info = env.step(action)
            total += r
            curve.append(info["exposure"])  # 0 = nothing taken, 1 = all sensitive hosts rooted

        totals.append(total)
        breached.append(info["breached"])
        padded.append(bad_picks)
        curves.append(curve)
    return (np.array(totals), np.array(breached), np.array(padded),
            np.array(curves).mean(axis=0))


def make_attackers(env):
    """Build the attackers to test. A new set is made per scenario."""
    attackers = {"random": RandomAttacker(), "greedy": GreedyAttacker()}
    if LearnedPolicy is not None:
        # Untrained network, sized from the env's observation/action shapes.
        attackers["learned"] = LearnedPolicy.for_env(env)
    return attackers


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    if LearnedPolicy is None:
        print("torch not installed: skipping the learned policy\n")

    # results[size][label] = (totals, breached, padded, curve), used for the chart
    results = {}
    for size, name in SCENARIOS.items():
        # NoDefense = the defender never acts, so we measure the attacker's
        # strength on its own. Swap in SweepDefender() to test against a
        # simple defender.
        env = AttackerEnv(f"{folder}/{name}.v2.yaml", defender=NoDefense())

        print(f"=== {size} ({name}), {episodes} episodes x {env.game.max_steps} steps")
        # success = % of episodes where the attacker compromised at least one
        # sensitive host. Against NoDefense this is 100 - the defender win rate
        # from the proposal's metrics.
        print(f"{'attacker':<10} {'reward (mean +- std)':>22} {'breached':>10} "
              f"{'success %':>10} {'padded picks':>13}")
        results[size] = {}
        for label, policy in make_attackers(env).items():
            totals, breached, padded, curve = run_episodes(env, policy, episodes)
            results[size][label] = (totals, breached, padded, curve)
            print(f"{label:<10} {totals.mean():>13.2f} +- {totals.std():<6.2f}"
                  f"{breached.mean():>10.2f} {100 * (breached > 0).mean():>10.0f} "
                  f"{padded.sum():>13d}")
        print()

    # Chart: one row per scenario size, three panels per row.
    #   left   - exposure over the episode (how fast each attacker gets in)
    #   middle - sensitive hosts breached at the end (higher = stronger attacker)
    #   right  - total attacker reward
    colors = {"random": "#888", "greedy": "#c44", "learned": "#48c"}
    fig, axes = plt.subplots(len(results), 3, figsize=(16, 4.5 * len(results)), squeeze=False)
    for (size, by_attacker), (ax1, ax2, ax3) in zip(results.items(), axes):
        labels = list(by_attacker)
        bar_colors = [colors[l] for l in labels]

        for label, (_, _, _, curve) in by_attacker.items():
            ax1.plot(curve, label=label, color=colors[label])
        ax1.set_title(f"{size}: exposure per step (mean)")
        ax1.set_xlabel("step")
        ax1.set_ylabel("exposure (0 to 1)")
        ax1.set_ylim(0, 1)
        ax1.legend()

        # Bars show the mean, error bars show one standard deviation.
        breached = [by_attacker[l][1] for l in labels]
        ax2.bar(labels, [b.mean() for b in breached], yerr=[b.std() for b in breached],
                capsize=6, color=bar_colors)
        ax2.set_title(f"{size}: sensitive hosts breached at end")
        ax2.set_ylabel("hosts")

        totals = [by_attacker[l][0] for l in labels]
        ax3.bar(labels, [t.mean() for t in totals], yerr=[t.std() for t in totals],
                capsize=6, color=bar_colors)
        ax3.set_title(f"{size}: total attacker reward")
        ax3.set_ylabel("reward")

    fig.suptitle(f"Attacker policies vs no defense, {episodes} episodes each")
    fig.tight_layout()
    fig.savefig("attacker_baselines.png", dpi=150)
    print("Saved chart to attacker_baselines.png")
