import json
from collections import defaultdict
import pandas as pd

# Mapping between algorithms and their file paths
algorithms = {
    "LS2IC": "../results/ls2ic/drl_paths_list.txt",
    "MADQN": "../results/ps_dqn/drl_paths_list.txt",
    "MADQN-PA": "../results/ps_dqn_a/drl_paths_list.txt",
    "MF-Q": "../results/meanfield/drl_paths_list.txt"
}

# Store the statistics for each algorithm
algo_link_usage = {}

# Count link usage frequency for each algorithm
for algo, path_file in algorithms.items():
    link_usage = defaultdict(int)

    with open(path_file, "r") as f:
        for line in f:
            step_data = json.loads(line.strip())
            for str_src in step_data:
                for str_dst in step_data[str_src]:
                    routes = step_data[str_src][str_dst]
                    if not routes:
                        continue
                    path = routes[0]
                    for u, v in zip(path[:-1], path[1:]):
                        edge = tuple(sorted((u, v)))
                        link_usage[edge] += 1

    algo_link_usage[algo] = link_usage

# Find all links that appear in the Top 10 of any algorithm (union)
top_links_set = set()
for usage_dict in algo_link_usage.values():
    top_links = sorted(usage_dict.items(), key=lambda x: -x[1])[:10]
    top_links_set.update([link for link, _ in top_links])

# Create a unified table
all_links = sorted(top_links_set)
table_data = []

for link in all_links:
    row = {"Link": f"{link}"}
    for algo in algorithms.keys():
        usage = algo_link_usage[algo].get(link, 0)
        row[algo] = usage
    table_data.append(row)

# Output as a DataFrame and save as CSV
df = pd.DataFrame(table_data)
df.to_csv("../results/compare/link_usage_comparison.csv", index=False)
