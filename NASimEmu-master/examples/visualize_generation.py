"""Procedurally generate several networks from ONE scenario and draw them side by side.

Usage (from the repo root):
    python examples/visualize_generation.py
    python examples/visualize_generation.py --scenario scenarios/uni.v2.yaml --seeds 6
    python examples/visualize_generation.py --scenario small-gen        # built-in NASim generator
"""
import argparse, os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import gym
import nasimemu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

parser = argparse.ArgumentParser()
parser.add_argument('--scenario', default='scenarios/sm_entry_dmz_one_subnet.v2.yaml',
                    help="a .v2.yaml file, or a built-in generator name (tiny-gen, small-gen, medium-gen-rgoal, ...)")
parser.add_argument('--seeds', type=int, default=4, help="how many networks to generate")
parser.add_argument('--out', default='generated_networks.png')
args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def draw(ax, sc, title):
    """Hosts as dots around a hub per subnet; hubs linked by the scenario topology; hub 0 is the attacker/internet."""
    g, hub_pos, host_pos, colors, sizes = nx.Graph(), {}, {}, {}, {}
    n_sub = len(sc.subnets)
    for s in range(n_sub):
        g.add_node(('hub', s))
    for a in range(n_sub):
        for b in range(a + 1, n_sub):
            if sc.topology[a][b]:
                g.add_edge(('hub', a), ('hub', b), kind='link')
    hub_pos = nx.circular_layout(nx.Graph([(a, b) for (_, a), (_, b) in g.edges()]), scale=1.0)
    for (addr, host) in sc.hosts.items():
        s, h = int(addr[0]), int(addr[1])
        ang = 2 * np.pi * h / sc.subnets[s]
        cx, cy = hub_pos[s]
        pos = (cx + 0.28 * np.cos(ang), cy + 0.28 * np.sin(ang))
        g.add_node(('host', s, h)); host_pos[('host', s, h)] = pos
        g.add_edge(('hub', s), ('host', s, h), kind='member')
    pos = {**{('hub', s): tuple(hub_pos[s]) for s in range(n_sub)}, **host_pos}
    link = [e for e in g.edges if g.edges[e]['kind'] == 'link']
    member = [e for e in g.edges if g.edges[e]['kind'] == 'member']
    nx.draw_networkx_edges(g, pos, edgelist=link, ax=ax, width=2, edge_color='#555')
    nx.draw_networkx_edges(g, pos, edgelist=member, ax=ax, width=0.8, edge_color='#bbb')
    hubs = [('hub', s) for s in range(n_sub)]
    nx.draw_networkx_nodes(g, pos, nodelist=hubs, ax=ax, node_size=[500 if s == 0 else 260 for s in range(n_sub)],
                           node_color=['orange' if s == 0 else '#999' for s in range(n_sub)], node_shape='s')
    nx.draw_networkx_labels(g, pos, {('hub', s): ('Attacker' if s == 0 else f'subnet {s}') for s in range(n_sub)},
                            ax=ax, font_size=8, verticalalignment='bottom')
    for addr, host in sc.hosts.items():
        s, h = int(addr[0]), int(addr[1])
        node = ('host', s, h)
        if s == 0:
            continue
        os_name = 'windows' if host.os.get('windows') else 'linux'  # built-in generators use other OS names, shown as green
        sens = tuple(addr) in sc.sensitive_hosts
        nx.draw_networkx_nodes(g, pos, nodelist=[node], ax=ax, node_size=420 if sens else 160,
                               node_color='#4c8bf5' if os_name == 'windows' else '#3fb27f',
                               node_shape='*' if sens else 'o', edgecolors='gold' if sens else 'none', linewidths=2)
    ax.set_title(title, fontsize=10); ax.axis('off'); ax.margins(0.2)


# most built-in presets ship without address_space_bounds, which NASimEmu needs; borrow the medium preset's (12 subnets, 6 hosts)
from nasimemu.nasim.scenarios.benchmark import AVAIL_GEN_BENCHMARKS
if args.scenario in AVAIL_GEN_BENCHMARKS:
    AVAIL_GEN_BENCHMARKS[args.scenario].setdefault('address_space_bounds', (12, 6))

env = gym.make('NASimEmu-v0', emulate=False, scenario_name=args.scenario).unwrapped  # strip gym wrappers to reach the NASim env
cols = min(args.seeds, 3)
rows = -(-args.seeds // cols)
fig, axes = plt.subplots(rows, cols, figsize=(max(5 * cols, 9), 5 * rows), squeeze=False)

for i in range(args.seeds):
    random.seed(i); np.random.seed(i)   # the env reseeds from entropy in __init__, so seed right before reset()
    env.reset()
    sc = env.env.scenario
    hosts = [int(n) for n in sc.subnets[1:]]
    draw(axes[i // cols][i % cols], sc, f"seed {i}: {sum(hosts)} hosts, subnets {hosts}, {len(sc.sensitive_hosts)} sensitive")
    print(f"seed {i}: subnet sizes {hosts}, sensitive hosts {sorted((int(a), int(b)) for a, b in sc.sensitive_hosts)}")

for j in range(args.seeds, rows * cols):
    axes[j // cols][j % cols].axis('off')

fig.legend(handles=[Line2D([], [], marker='o', ls='', color='#3fb27f', label='Linux host'),
                    Line2D([], [], marker='o', ls='', color='#4c8bf5', label='Windows host'),
                    Line2D([], [], marker='*', ls='', color='#4c8bf5', mec='gold', ms=14, label='Sensitive host (goal)'),
                    Line2D([], [], marker='s', ls='', color='orange', label='Attacker / internet')],
           loc='lower center', ncol=4 if cols > 1 else 2, frameon=False)
fig.suptitle(f"Generated from: {args.scenario}")
fig.tight_layout(rect=(0, 0.08 if cols == 1 else 0.04, 1, 1))
fig.savefig(args.out, dpi=110)
print(f"Saved {os.path.abspath(args.out)}")
