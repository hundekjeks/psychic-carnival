import sys
from collections import defaultdict

# Elevated diagnostics depth constraints included directly
sys.tracebacklimit = 100000
sys.setrecursionlimit(50000)


def find_node_weighted_equivalence_classes(node_weights, edges, active_nodes):
    """Groups active nodes with identical outbound targets and weights."""
    raw_targets = defaultdict(list)
    for u, v in edges:
        raw_targets[u].append(v)

    signature_map = defaultdict(list)
    for u in active_nodes:
        raw_targets[u].sort()
        sig_tuple = (node_weights[u], tuple(raw_targets[u]))
        signature_map[sig_tuple].append(u)

    for nodes in signature_map.values():
        nodes.sort()
        # Yield a safe 2-element tuple containing (list, tuple)
        yield (nodes, tuple(nodes))


class GraphEngine:

    def __init__(self, node_weights, edges_list):
        """Prepares topology and builds highly compressed structural maps."""
        self.num_nodes = len(node_weights)
        self.node_weights = node_weights
        self.self_loops = []

        filtered_edges = []
        for u, v in edges_list:
            if u == v:
                if self.node_weights[u] > 0:
                    self.self_loops.append(([u], self.node_weights[u]))
            else:
                filtered_edges.append((u, v))

        # Linear Queue Pruning (Pass A) - Absolute O(V+E) bounds
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        adj_temp_prune = defaultdict(set)
        rev_adj_prune = defaultdict(set)
        all_nodes = set(range(self.num_nodes))

        for u, v in filtered_edges:
            if v not in adj_temp_prune[u]:
                adj_temp_prune[u].add(v)
                rev_adj_prune[v].add(u)
                out_deg[u] += 1
                in_deg[v] += 1

        prune_queue = [
            i for i in all_nodes if out_deg[i] == 0 or in_deg[i] == 0
        ]
        removed_nodes = set()

        while prune_queue:
            u = prune_queue.pop()
            if u in removed_nodes:
                continue
            removed_nodes.add(u)

            for p in list(rev_adj_prune[u]):
                if u in adj_temp_prune[p]:
                    adj_temp_prune[p].remove(u)
                    out_deg[p] -= 1
                    if out_deg[p] == 0:
                        prune_queue.append(p)

            for n in list(adj_temp_prune[u]):
                if u in rev_adj_prune[n]:
                    rev_adj_prune[n].remove(u)
                    in_deg[n] -= 1
                    if in_deg[n] == 0:
                        prune_queue.append(n)

        active_edges = []
        pruned_nodes = set()
        for u in all_nodes:
            if u not in removed_nodes:
                for v in adj_temp_prune[u]:
                    if v not in removed_nodes:
                        active_edges.append((u, v))
                        pruned_nodes.add(u)
                        pruned_nodes.add(v)

        # 2. Compute Equivalence Classes
        self.eq_classes = list(
            find_node_weighted_equivalence_classes(
                node_weights, active_edges, pruned_nodes
            )
        )

        self.node_to_rep = {}
        self.rep_to_class_list = {}

        for nodes_list, class_nodes in self.eq_classes:
            rep = nodes_list[0]
            self.rep_to_class_list[rep] = list(class_nodes)
            for node in class_nodes:
                self.node_to_rep[node] = rep

        # 3. Rebuild Compressed Adjacency and run Tarjan
        adj_temp = defaultdict(list)
        rep_nodes_set = set(nodes_list[0] for nodes_list, _ in self.eq_classes)
        for u, v in active_edges:
            rep_u = self.node_to_rep[u]
            rep_v = self.node_to_rep[v]
            if rep_u != rep_v:
                adj_temp[rep_u].append(rep_v)

        index = 0
        dfn = {}
        low = {}
        stack_scc = []
        on_stack = set()
        scc_map = {}

        def tarjan(u):
            nonlocal index
            dfn[u] = low[u] = index
            index += 1
            stack_scc.append(u)
            on_stack.add(u)

            for v in adj_temp[u]:
                if v not in dfn:
                    tarjan(v)
                    low[u] = min(low[u], low[v])
                elif v in on_stack:
                    low[u] = min(low[u], dfn[v])

            if low[u] == dfn[u]:
                while True:
                    v = stack_scc.pop()
                    on_stack.remove(v)
                    scc_map[v] = u
                    if v == u:
                        break

        for node in rep_nodes_set:
            if node not in dfn:
                tarjan(node)

        # Drop cross-component edges at the representative level
        adj_sets = defaultdict(set)
        for u, targets in adj_temp.items():
            for v in targets:
                if u in scc_map and v in scc_map and scc_map[u] == scc_map[v]:
                    adj_sets[u].add(v)

        self.adj = defaultdict(list)
        for rep_node, targets in adj_sets.items():
            self.adj[rep_node] = sorted(list(targets))

    def get_reduction_summary(self):
        """Returns the equivalence data topology configurations."""
        summary = {
            "num_nodes": self.num_nodes,
            "reduced_classes": len(self.eq_classes),
            "classes": {},
        }
        for idx, (nodes_list, class_nodes) in enumerate(self.eq_classes):
            summary["classes"][idx] = {
                "representative": nodes_list[0],
                "nodes": list(class_nodes),
            }
        return summary

    def generate_cycles(self):
        """Threadless linear generator utilizing an index-fenced DFS walk."""
        for self_loop_cycle in self.self_loops:
            yield self_loop_cycle

        path_stack = []
        path_set = set()
        visited_origins = set()
        total_reps = len(self.eq_classes)

        def dfs(u, start_node, current_weight):
            path_stack.append(u)
            path_set.add(u)

            max_future_steps = total_reps - len(path_stack)
            if current_weight + max_future_steps <= 0:
                path_stack.pop()
                path_set.remove(u)
                return

            for v in self.adj[u]:
                if v < start_node or v in visited_origins:
                    continue
                if v == start_node:
                    if current_weight > 0:
                        yield (
                            [self.rep_to_class_list[r] for r in path_stack],
                            current_weight,
                        )
                elif v not in path_set:
                    yield from dfs(v, start_node, current_weight + self.node_weights[v])

            path_stack.pop()
            path_set.remove(u)

        representatives = sorted([nodes_list[0] for nodes_list, _ in self.eq_classes])
        for rep_node in representatives:
            yield from dfs(rep_node, rep_node, self.node_weights[rep_node])
            visited_origins.add(rep_node)
