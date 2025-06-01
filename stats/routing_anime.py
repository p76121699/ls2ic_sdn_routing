import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
import config

# ===========================
# Parameter Settings
# ===========================
algo = config["algo"]
bw_r_file = f'../dataset/{config["topo"]}_traffic/bw_r.txt'
pos_file = f'../dataset/{config["topo"]}_traffic/traffic_generator/pos.json'
metric_file = f'../results/{algo}/net_metrics.csv'
path_file = f'../results/{algo}/drl_paths_list.txt'
max_capacity = 200000
interval_ms = config["interval_ms"]  # Animation interval (milliseconds)
source_target = config["source_target"]  # Source-destination pairs to track
color = ['red', 'blue', 'purple', 'orange', 'black']

# ===========================
# Function to draw offset dashed lines
# ===========================
def draw_offset_edge(ax, pos, u, v, offset=0.05, **line_kwargs):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    if length == 0:
        return
    offset_x = -dy / length * offset
    offset_y = dx / length * offset
    new_x1, new_y1 = x1 + offset_x, y1 + offset_y
    new_x2, new_y2 = x2 + offset_x, y2 + offset_y
    ax.plot([new_x1, new_x2], [new_y1, new_y2], **line_kwargs)

# ===========================
# Load topology and node positions
# ===========================
def make_anime():
    G = nx.Graph()
    with open(bw_r_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            node1, node2, delay, bw = map(int, parts)
            G.add_edge(node1, node2, weight=1)

    with open(pos_file, 'r') as f:
        pos = json.load(f)
        pos = {int(k): tuple(v) for k, v in pos.items()}

    # ===========================
    # Load all metrics data and path data
    # ===========================
    metric_df = pd.read_csv(metric_file)
    path_data = []
    with open(path_file, 'r') as f:
        for line in f:
            path_data.append(json.loads(line.strip()))

    # ===========================
    # Initialize plot
    # ===========================
    fig, ax = plt.subplots(figsize=(12, 10))

    labels = {n: str(int(n)) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm, ax=ax, label='(Remaining Bandwidth / Max Capacity)')
    ax.axis('off')

    for idx, (src, dst) in enumerate(source_target):
        y_pos = 0.95 - idx * 0.04

        # Draw dashed line legend
        line = Line2D(
            [0.03, 0.08], [y_pos, y_pos],
            transform=fig.transFigure,
            color=color[idx],
            linestyle='dashed',
            linewidth=2
        )
        fig.lines.append(line)

        # Display corresponding SD pair label
        fig.text(
            0.09, y_pos+0.005,
            f"({src}, {dst})",
            color='black',
            fontsize=10,
            ha='left',
            va='top'
        )

    # ===========================
    # Update function for each frame
    # ===========================
    def update(frame_idx):
        ax.clear()

        # Draw nodes and base topology
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, labels={n: str(int(n)) for n in G.nodes()}, ax=ax)

        step_df = metric_df[metric_df['step'] == frame_idx]
        new_weights = []
        for u, v in G.edges():
            match = step_df[((step_df['node1'] == u) & (step_df['node2'] == v)) |
                            ((step_df['node1'] == v) & (step_df['node2'] == u))]
            if not match.empty:
                free_bw = match.iloc[0]['bwd']
                norm = free_bw / max_capacity
                new_weights.append(norm)
            else:
                new_weights.append(1.0)

        # Draw dashed path
        for j in range(len(source_target)):
            try:
                data = path_data[frame_idx]  # First path
                route = data[str(source_target[j][0])][str(source_target[j][1])][0]
                for i in range(len(route) - 1):
                    draw_offset_edge(
                        ax, pos, route[i], route[i+1],
                        offset=0.03,
                        color=color[j], linestyle='dashed', linewidth=2.0, alpha=0.8
                    )
            except KeyError:
                pass  # This step or pair does not exist

        ax.set_title(f"{algo}: Step {frame_idx+1}")
        ax.axis('off')

    # ===========================
    # Play animation
    # ===========================
    num_frames = len(path_data)
    ani = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, repeat=False)
    fps = int(1000 / interval_ms)
    mp4_writer = FFMpegWriter(fps=fps, metadata=dict(artist='de'), bitrate=1800)
    ani.save("./results/"+algo+"/network_animation.mp4", writer=mp4_writer)
    #ani.save("./"+algo+"/network_animation2.gif", writer='pillow', fps=5)
    #plt.show()

if __name__ == '__main__':
    make_anime()
