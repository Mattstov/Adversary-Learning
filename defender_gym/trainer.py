"""Train the attacker and the defender with PPO.

Both agents are LearnedPolicy networks (policies/learned.py). Two modes:

    separate   (default) each agent trains against a fixed scripted opponent:
               defender vs DEFENDER_OPPONENT (greedy) in DefenderEnv,
               attacker vs ATTACKER_OPPONENT (no defense) in AttackerEnv.
               Each episode, each agent plays one game of its own.
    selfplay   the two learned agents play each other in DefenderEnv
               (defender acts, then attacker). Both keep learning.

Every UPDATE_EVERY episodes, each agent gets one PPO update from its own steps.

Usage (from inside defender_gym/, with the venv active):
    python trainer.py ../NASimEmu-master/scenarios [episodes] [small|medium] [separate|selfplay]

Defaults: 1000 episodes, small, separate. Saves:
    checkpoints/defender.pt, checkpoints/attacker.pt   final policies
    checkpoints/*_ep<N>.pt                             a copy every SAVE_EVERY episodes
    training_log.csv                                   one row per episode
    training_curves.png                                rewards and breaches over training

To see what each agent learned, test the checkpoints with run_defender.py and
run_attacker.py (they add a "trained" bar when the checkpoints exist).
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch

from env import AttackerEnv, DefenderEnv
from policies import GreedyAttacker, NoDefense, RandomAttacker, RandomDefender, SweepDefender
from policies.learned import LearnedPolicy

SCENARIOS = {
    "small": "sm_entry_dmz_two_subnets",
    "medium": "md_entry_dmz_two_subnets",
}
# Fixed scripted opponents for "separate" mode. Swap in RandomAttacker,
# RandomDefender or SweepDefender (all imported above) to train against another.
DEFENDER_OPPONENT = GreedyAttacker   # the attacker the defender trains against
ATTACKER_OPPONENT = NoDefense        # the defender the attacker trains against

UPDATE_EVERY = 10   # episodes between PPO updates (10 x 100 = 1000 steps per update)
SAVE_EVERY = 100    # episodes between checkpoint copies
SEED = 0

# PPO settings (standard defaults)
LR = 3e-4           # learning rate
GAMMA = 0.99        # discount: how much future reward counts
LAM = 0.95          # GAE smoothing for the advantages
CLIP = 0.2          # how far one update may move the policy
EPOCHS = 4          # passes over the collected steps per update
MINIBATCH = 256     # steps per gradient step
VALUE_COEF = 0.5    # weight of the critic's loss
ENTROPY_COEF = 0.01 # bonus for keeping some randomness, so it keeps exploring
MAX_GRAD = 0.5      # gradient clipping


class Agent:
    """One learning agent: its policy, its optimizer, and the steps it has
    collected since its last update."""

    def __init__(self, policy):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.net.parameters(), lr=LR)
        self.clear()

    def clear(self):
        self.obs, self.mask, self.actions = [], [], []
        self.logp, self.values, self.rewards, self.dones = [], [], [], []

    @torch.no_grad()
    def act(self, obs, mask):
        """Sample an action and remember what PPO needs about it."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
        mask_t = torch.as_tensor(mask, dtype=torch.bool)[None]
        dist, value = self.policy.distribution(obs_t, mask_t)
        idx = dist.sample()
        # The network picks one (host row, action kind) pair out of a flat list.
        action = np.array(divmod(int(idx), self.policy.net.n_kinds))

        self.obs.append(np.array(obs, dtype=np.float32))
        self.mask.append(np.array(mask, dtype=bool))
        self.actions.append(action)
        self.logp.append(float(dist.log_prob(idx)))
        self.values.append(float(value))
        return action

    def record(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def advantages(self):
        """GAE: how much better each action did than the critic expected.
        Returns (advantages, returns). Updates always happen at the end of an
        episode, so the last stored step is always done."""
        adv = np.zeros(len(self.rewards), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_value, last = 0.0, 0.0  # episode ended: nothing comes after
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + GAMMA * next_value - self.values[t]
            last = delta + GAMMA * LAM * last
            adv[t] = last
        return adv, adv + np.array(self.values, dtype=np.float32)

    def update(self):
        """One PPO update on everything collected since the last one."""
        adv, returns = self.advantages()
        obs = torch.as_tensor(np.array(self.obs))
        mask = torch.as_tensor(np.array(self.mask))
        actions = torch.as_tensor(np.array(self.actions))
        old_logp = torch.as_tensor(self.logp, dtype=torch.float32)
        returns = torch.as_tensor(returns)
        adv = torch.as_tensor(adv)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(EPOCHS):
            for idx in torch.randperm(len(adv)).split(MINIBATCH):
                logp, entropy, value = self.policy.evaluate(obs[idx], mask[idx], actions[idx])
                # Clipped PPO objective: push up good actions, but not too far at once.
                ratio = torch.exp(logp - old_logp[idx])
                policy_loss = -torch.min(ratio * adv[idx],
                                         ratio.clamp(1 - CLIP, 1 + CLIP) * adv[idx]).mean()
                value_loss = ((value - returns[idx]) ** 2).mean()
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.net.parameters(), MAX_GRAD)
                self.opt.step()
        self.clear()


def moving_average(x, n=20):
    x = np.asarray(x, dtype=float)
    if len(x) < n:
        return x
    return np.convolve(x, np.ones(n) / n, mode="valid")


def play_alone(env, agent):
    """One game with `agent` choosing every move; the env's scripted opponent
    moves inside env.step(). Returns (total reward, sensitive hosts breached)."""
    obs, done, total = env.reset(), False, 0.0
    while not done:
        action = agent.act(obs, env.host_mask())
        obs, r, done, info = env.step(action)
        agent.record(r, done)
        total += r
    return total, info["breached"]


def play_each_other(env, defender, attacker):
    """One game between the two learned agents. Returns (defender reward,
    attacker reward, sensitive hosts breached)."""
    env.new_game()  # fresh network, nothing compromised
    def_obs = env._obs()
    done, def_total, att_total = False, 0.0, 0.0
    while not done:
        att_obs, att_mask = env.attacker_view()
        def_action = defender.act(def_obs, env.host_mask())
        att_action = attacker.act(att_obs, att_mask)
        # Defender acts first, then the attacker (same order as DefenderEnv.step).
        def_r, att_r, done, info = env.play(def_action, att_action)
        defender.record(def_r, done)
        attacker.record(att_r, done)
        def_obs = env._obs()
        def_total += def_r
        att_total += att_r
    return def_total, att_total, info["breached"]


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "scenarios"
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    size = sys.argv[3] if len(sys.argv) > 3 else "small"
    mode = sys.argv[4] if len(sys.argv) > 4 else "separate"
    assert mode in ("separate", "selfplay"), "mode must be separate or selfplay"

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    path = f"{folder}/{SCENARIOS[size]}.v2.yaml"
    if mode == "separate":
        def_env = DefenderEnv(path, attacker=DEFENDER_OPPONENT())  # resets the attacker each game
        att_env = AttackerEnv(path, defender=ATTACKER_OPPONENT())  # resets the defender each game
        defender = Agent(LearnedPolicy.for_env(def_env))
        attacker = Agent(LearnedPolicy.for_env(att_env))
        opponents = f"defender vs {DEFENDER_OPPONENT.__name__}, attacker vs {ATTACKER_OPPONENT.__name__}"
    else:
        # We call env.play() with both agents' actions, so the scripted
        # attacker stub inside DefenderEnv is never used.
        env = DefenderEnv(path)
        att_obs, _ = env.attacker_view()
        defender = Agent(LearnedPolicy.for_env(env))  # 4 features per host, 3 action kinds
        attacker = Agent(LearnedPolicy(att_obs.shape[1], len(env.sim.action_list)))
        opponents = "defender vs learned attacker (self-play)"

    print(f"Training on {size} ({SCENARIOS[size]}), {episodes} episodes, mode {mode}, "
          f"PPO update every {UPDATE_EVERY} episodes")
    print(f"  {opponents}")
    print(f"{'episodes':>9} {'defender reward':>16} {'breached':>9} "
          f"{'attacker reward':>16} {'breached':>9}")

    # One row per episode: episode, defender reward, breached in the defender's
    # game, attacker reward, breached in the attacker's game. In selfplay both
    # agents share one game, so the two breach columns are the same.
    log = []
    for ep in range(1, episodes + 1):
        if mode == "separate":
            def_total, def_breached = play_alone(def_env, defender)
            att_total, att_breached = play_alone(att_env, attacker)
        else:
            def_total, att_total, def_breached = play_each_other(env, defender, attacker)
            att_breached = def_breached
        log.append((ep, def_total, def_breached, att_total, att_breached))

        if ep % UPDATE_EVERY == 0:
            defender.update()
            attacker.update()
            recent = np.array(log[-UPDATE_EVERY:])
            print(f"{ep:>9} {recent[:, 1].mean():>16.2f} {recent[:, 2].mean():>9.2f} "
                  f"{recent[:, 3].mean():>16.2f} {recent[:, 4].mean():>9.2f}")

        if ep % SAVE_EVERY == 0:
            defender.policy.save(f"checkpoints/defender_ep{ep}.pt")
            attacker.policy.save(f"checkpoints/attacker_ep{ep}.pt")

    defender.policy.save("checkpoints/defender.pt")
    attacker.policy.save("checkpoints/attacker.pt")
    print("Saved checkpoints/defender.pt and checkpoints/attacker.pt")

    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "defender_reward", "defender_game_breached",
                         "attacker_reward", "attacker_game_breached"])
        writer.writerows(log)
    print("Saved training_log.csv")

    # Chart: one panel per agent, 20-episode moving average. Reward on the left
    # axis, breaches (dashed) on the right. A defender that is learning: reward
    # rises, breaches fall. An attacker that is learning: both rise.
    log = np.array(log)
    def_opp = DEFENDER_OPPONENT.__name__ if mode == "separate" else "learned attacker"
    att_opp = ATTACKER_OPPONENT.__name__ if mode == "separate" else "learned defender"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # ColorBrewer Set1
    for ax, name, opp, r_col, b_col, color in [
            (axes[0], "defender", def_opp, 1, 2, "#377eb8"),
            (axes[1], "attacker", att_opp, 3, 4, "#e41a1c")]:
        ax.plot(moving_average(log[:, r_col]), color=color, label="reward")
        ax.set_title(f"{name} vs {opp} (20-episode average)")
        ax.set_xlabel("episode")
        ax.set_ylabel("reward")
        ax_b = ax.twinx()
        ax_b.plot(moving_average(log[:, b_col]), color="#999999", linestyle="--", label="breached")
        ax_b.set_ylabel("sensitive hosts breached at end")
        ax.legend(loc="upper left")
        ax_b.legend(loc="upper right")
    fig.suptitle(f"Training on {size}, {episodes} episodes, {mode}")
    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=150)
    print("Saved chart to training_curves.png")
