"""Isolated test of the defender policies (same idea as run_attacker.py).

Plays each defender policy for N episodes against a fixed attacker, on the
small and the medium scenario, prints the average results, and saves a
chart to defender_vs_<attacker>.png.

Usage (from inside defender_gym/, with the venv active):
    python run_defender.py ../NASimEmu-master/scenarios [episodes] [greedy|random]

[episodes] sets the small scenario (default 100). Medium always runs
MEDIUM_EPISODES (200); change it below if needed.

What a healthy result looks like:
    - none is the worst (or tied worst) on exposure: it is the baseline
    - sweep beats none and random on exposure, at some action cost
    - learned (untrained) scores about the same as random
    - learned never picks a padded (non-existent) host row

"win %" is the share of episodes that end with no sensitive host breached
(the defender win rate from the proposal). "vs none" is the defender's mean
reward minus NoDefense's on the same size, so positive = better than doing
nothing. "false restores" counts Restore on hosts that were not compromised,
which is the proposal's false positive measure. Note the env's per-action
cost (Monitor 0.01, Restore 0.1) is flat: it charges every Restore the same,
so it does NOT penalize false positives specifically.
"""
import sys

import matplotlib
matplotlib.use("Agg")  # save to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

from env import DefenderEnv, RESTORE
from policies import GreedyAttacker, NoDefense, RandomAttacker, RandomDefender, SweepDefender

# The learned policy needs torch. If torch isn't installed, skip it
# instead of crashing, so the scripted defenders can still be tested.
try:
    from policies.learned import LearnedPolicy
except ImportError:
    LearnedPolicy = None

# Use the two_subnets scenarios: the one_subnet ones have no sensitive hosts,
# so exposure (and the defender's loss) would always be 0.
SCENARIOS = {
    "small": "sm_entry_dmz_two_subnets",
    "medium": "md_entry_dmz_two_subnets",
}
ATTACKERS = {"greedy": GreedyAttacker, "random": RandomAttacker}
# Medium has its own episode count, separate from the [episodes] argument.
MEDIUM_EPISODES = 200


def run_episodes(env, policy, episodes):
    """Play `episodes` full games with `policy` as the defender.

    Returns:
      totals   - total defender reward per episode
      breached - sensitive hosts compromised at the end of each episode
      cost     - total action cost paid by the defender per episode
      restores - Restore actions per episode
      false_r  - Restores on hosts that were NOT compromised, per episode
      padded   - how many times the policy picked a padded host row, per episode
      curve    - mean exposure at each step, averaged over all episodes
    """
    totals, breached, cost, restores, false_r, padded, curves = [], [], [], [], [], [], []
    for _ in range(episodes):
        # DefenderEnv.reset() resets the attacker for us, but NOT the
        # defender policy we pass in, so we do that here (SweepDefender
        # needs it to restart its sweep).
        obs = env.reset()
        policy.reset()

        done, total, paid, n_rest, n_false, bad_picks, curve = False, 0.0, 0.0, 0, 0, 0, []
        while not done:
            mask = env.host_mask()  # True for real hosts, False for padding
            action = policy.act(obs, mask)
            row, kind = int(action[0]), int(action[1])
            if not mask[row]:
                # Padding row: the env ignores the action, but a working
                # policy should never choose it.
                bad_picks += 1
            elif kind == RESTORE:
                n_rest += 1
                # Check BEFORE stepping, since Restore clears the flag.
                if not env._host(env.hosts[row]).compromised:
                    n_false += 1
            paid += env.defender_costs[kind]
            obs, r, done, info = env.step(action)
            total += r
            curve.append(info["exposure"])  # 0 = nothing taken, 1 = all sensitive hosts rooted

        totals.append(total)
        breached.append(info["breached"])
        cost.append(paid)
        restores.append(n_rest)
        false_r.append(n_false)
        padded.append(bad_picks)
        curves.append(curve)
    return dict(totals=np.array(totals), breached=np.array(breached), cost=np.array(cost),
                restores=np.array(restores), false_r=np.array(false_r),
                padded=np.array(padded), curve=np.array(curves).mean(axis=0))


def make_defenders(env):
    """Build the defenders to test. A new set is made per scenario."""
    defenders = {"none": NoDefense(), "random": RandomDefender(), "sweep": SweepDefender()}
    if LearnedPolicy is not None:
        # Untrained network, sized from the env's observation/action shapes
        # (4 features per host, 3 action kinds).
        defenders["learned"] = LearnedPolicy.for_env(env)
    return defenders


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    attacker_name = sys.argv[3] if len(sys.argv) > 3 else "greedy"

    if LearnedPolicy is None:
        print("torch not installed: skipping the learned policy\n")

    # results[size][label] = dict from run_episodes, used for the chart
    results = {}
    n_episodes = {"small": episodes, "medium": MEDIUM_EPISODES}
    for size, name in SCENARIOS.items():
        # The attacker is fixed for the whole run, like NoDefense was fixed
        # in run_attacker.py. Greedy is the strongest scripted attacker.
        env = DefenderEnv(f"{folder}/{name}.v2.yaml", attacker=ATTACKERS[attacker_name]())

        print(f"=== {size} ({name}) vs {attacker_name} attacker, "
              f"{n_episodes[size]} episodes x {env.max_steps} steps")
        print(f"{'defender':<9} {'reward (mean +- std)':>22} {'vs none':>8} {'breached':>9} "
              f"{'win %':>6} {'cost':>6} {'restores':>9} {'false restores':>15} {'padded':>7}")
        results[size] = {}
        for label, policy in make_defenders(env).items():
            res = run_episodes(env, policy, n_episodes[size])
            results[size][label] = res
            t = res["totals"]
            gap = t.mean() - results[size]["none"]["totals"].mean()  # "none" runs first
            print(f"{label:<9} {t.mean():>13.2f} +- {t.std():<6.2f}{gap:>8.2f} "
                  f"{res['breached'].mean():>9.2f} {100 * (res['breached'] == 0).mean():>6.0f} "
                  f"{res['cost'].mean():>6.2f} {res['restores'].mean():>9.1f} "
                  f"{res['false_r'].mean():>15.1f} {res['padded'].sum():>7d}")
        print()

    # Chart: one row per scenario size, four panels per row.
    #   1 - exposure over the episode (lower = better defense)
    #   2 - sensitive hosts breached at the end (lower = better)
    #   3 - total defender reward (higher = better)
    #   4 - action cost paid (what the defense cost to run)
    colors = {"none": "#c44", "random": "#888", "sweep": "#48c", "learned": "#4a4"}
    fig, axes = plt.subplots(len(results), 4, figsize=(20, 4.5 * len(results)), squeeze=False)
    for (size, by_def), (ax1, ax2, ax3, ax4) in zip(results.items(), axes):
        labels = list(by_def)
        bar_colors = [colors[l] for l in labels]

        for label, res in by_def.items():
            ax1.plot(res["curve"], label=label, color=colors[label])
        ax1.set_title(f"{size}: exposure per step (mean, {n_episodes[size]} episodes)")
        ax1.set_xlabel("step")
        ax1.set_ylabel("exposure (0 to 1)")
        ax1.set_ylim(0, 1)
        ax1.legend()

        # Bars show the mean, error bars show one standard deviation.
        for ax, key, title, ylabel in [
                (ax2, "breached", "sensitive hosts breached at end", "hosts"),
                (ax3, "totals", "total defender reward", "reward"),
                (ax4, "cost", "defender action cost", "cost")]:
            data = [by_def[l][key] for l in labels]
            ax.bar(labels, [d.mean() for d in data], yerr=[d.std() for d in data],
                   capsize=6, color=bar_colors)
            ax.set_title(f"{size}: {title}")
            ax.set_ylabel(ylabel)

    fig.suptitle(f"Defender policies vs {attacker_name} attacker "
                 f"({n_episodes['small']} small / {n_episodes['medium']} medium episodes)")
    fig.tight_layout()
    out = f"defender_vs_{attacker_name}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved chart to {out}")
