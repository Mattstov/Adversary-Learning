"""Run a random agent in the NASimEmu simulator (no emulation needed).

Usage (from the repo root):
    python examples/run_simulation.py
    python examples/run_simulation.py --scenario scenarios/uni.v2.yaml --steps 50
"""
import argparse, os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import gym
import nasimemu, nasimemu.env_utils as env_utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', default='scenarios/sm_entry_dmz_one_subnet.v2.yaml:scenarios/sm_entry_dmz_two_subnets.v2.yaml',
                    help="one or more scenario files separated by ':'")
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--plot', default='network_trace.html', help="where to save the final observation graph ('' to skip)")
args = parser.parse_args()

# scenario paths are relative to the repo root (absolute Windows paths would clash with the ':' separator)
os.chdir(ROOT)

env = gym.make('NASimEmu-v0', emulate=False, scenario_name=args.scenario, step_limit=args.steps)
if args.seed is not None:  # seed after gym.make (the env reseeds from entropy there) so the network layout is reproducible too
    random.seed(args.seed); np.random.seed(args.seed)
s = env.reset()

total_reward = 0.
action, info = None, {'subnet_graph': env.subnet_graph}
for step in range(1, args.steps + 1):
    actions = env_utils.get_possible_actions(env, s)
    action = random.choice(actions)
    s, r, done, info = env.step(action)
    total_reward += r

    (action_subnet, action_host), action_id = action
    action_class, action_params = env.action_list[action_id]
    action_name = action_class.__name__ + (f"({action_params['name']})" if 'name' in action_params else '')
    print(f"[{step:3d}] {action_name:32s} on subnet={action_subnet} host={action_host} -> reward={r:+.2f}")

    if done:
        s = info['s_true']  # the env auto-resets on done; keep the final observation
        break

print("\nFinal observation:")
env.render(s)

s_graph = env_utils.convert_to_graph(s, info['subnet_graph'])
print(f"\nEpisode finished after {step} steps, total reward {total_reward:+.2f}, done={done}")
print(f"Graph observation: {len(s_graph[0])} nodes discovered")

if args.plot:
    fig = env_utils.plot_network(s, info['subnet_graph'], info['a_raw'])
    fig.write_html(args.plot)
    print(f"Network plot saved to {os.path.abspath(args.plot)}")
