import json
import csv
import networkx as nx
from typing import Dict, List, Tuple


class dijkstraAgent:
    """Per‑period bandwidth‑aware shortest‑path agent.

    • 一次性讀 `bw_r.txt` 取得 link capacity (Mbps)。
    • 每次 `get_action()`：
        1. 讀最新 `net_info.csv` → link 目前 `free_bw` (Mbps)。
        2. 權重 = 1 / max(free_ratio , ε)
        3. 以 NetworkX `shortest_path()` 為所有 (src,dst) 計算 path。
        4. 回傳 nested dict `{src: {dst: [path]}}`   —— 與你示例一致。
    """

    def __init__(self, args):
        self.topo_file = args.bw_file
        self.net_info_file = args.net_info_file
        self.G = self._load_topology()
        self.nodes = sorted(self.G.nodes())  # 方便迭代

    # ---------- file loaders ----------
    def _load_topology(self):
        G = nx.Graph()
        with open(self.topo_file) as f:
            for line in f:
                n1, n2, _, cap = map(int, line.strip().split(','))
                G.add_edge(n1, n2, capacity=cap)
        return G

    def _load_net_info(self) -> Dict[Tuple[int, int], float]:
        """Return map (u,v) -> free_bw(Mbps). 排除 header。"""
        free = {}
        with open(self.net_info_file) as f:
            rdr = csv.reader(f)
            next(rdr)  # skip header
            for n1, n2, bwd_kBps, *_ in rdr:
                u, v = int(n1), int(n2)
                resid = float(bwd_kBps) / 1000  # kB/s → Mb/s
                free_bw = max(resid, 0.0)
                free[(u, v)] = free_bw
                free[(v, u)] = free_bw
        return free

    # ---------- main ----------
    def get_action(self, *_):
        free = self._load_net_info()
        eps = 1e-6
        Gw = nx.Graph()
        for u, v, data in self.G.edges(data=True):
            cap = self.G[u][v]["capacity"] * 2# Mbps
            free_link = free.get((u, v), 0)
            free_ratio = max(free_link / cap, 1e-3)   # 避免 0
            weight = 1.0 / (free_ratio ** 2)              # 放大差異
            Gw.add_edge(u, v, weight=weight)

        actions = {}
        for src in self.nodes:
            actions[src] = {}
            for dst in self.nodes:
                if src == dst:
                    continue
                path = nx.shortest_path(Gw, src, dst, weight='weight')
                actions[src][dst] = [path]
        return actions, {}

    # ----- stub -----
    def append_sample(self, *args, **kwargs):
        pass
    def update(self):
        return {}
    def save_model(self, *args, **kwargs):
        pass
    def load_model(self, *args, **kwargs):
        pass
    def update_target(self):
        pass
