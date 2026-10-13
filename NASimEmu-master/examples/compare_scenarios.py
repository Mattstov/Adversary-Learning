"""Run a random agent on a baseline and a modified scenario with identical seeds and print them side by side.

Usage (from the repo root):
    python examples/compare_scenarios.py
    python examples/compare_scenarios.py --episodes 50 --steps 50
"""
import argparse, os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import gym
import nasimemu, nasimemu.env_utils as env_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', default='scenarios/sm_entry_dmz_one_subnet.v2.yaml')
parser.add_argument('--modified', default='scenarios/custom_entry_dmz_one_subnet.v2.yaml')
parser.add_argument('--episodes', type=int, default=20)
parser.add_argument('--steps', type=int, default=30)
args = parser.parse_args()


def run_episode(env, seed):
    # the env reseeds from entropy in __init__, so seed here, right before the scenario is generated
    random.seed(seed); np.random.seed(seed)
    s = env.reset()
    total, captured = 0., 0
    for _ in range(args.steps):
        action = random.choice(env_utils.get_possible_actions(env, s))
        s, r, done, info = env.step(action)
        total += r
        if r > 0:
            captured += 1
        if done:
            s = info['s_true']
            break
    return total, captured


def evaluate(scenario):
    env = gym.make('NASimEmu-v0', emulate=False, scenario_name=scenario, step_limit=args.steps)
    res = [run_episode(env, seed) for seed in range(args.episodes)]
    rewards = np.array([r for r, _ in res]); caps = np.array([c for _, c in res])
    return rewards, caps


print(f"{args.episodes} episodes x {args.steps} steps, random agent, seeds 0..{args.episodes - 1}\n")
print(f"{'':10s} {'mean reward':>12s} {'std':>7s} {'best':>7s} {'worst':>7s} {'ep. with goal':>14s}")
for name, scenario in (('baseline', args.baseline), ('modified', args.modified)):
    rewards, caps = evaluate(scenario)
    print(f"{name:10s} {rewards.mean():+12.2f} {rewards.std():7.2f} {rewards.max():+7.1f} {rewards.min():+7.1f} "
          f"{(caps > 0).sum():>10d}/{args.episodes}")
