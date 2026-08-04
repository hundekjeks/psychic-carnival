import unittest
from cycles_engine import GraphEngine


class TestExtensiveGraphEngine(unittest.TestCase):

    def test_01_baseline_equivalence_and_cycle(self):
        """Verifies core structural reduction and loop containment."""
        weights = [1, -1, 1, 1]
        edges = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 0)]

        engine = GraphEngine(weights, edges)
        summary = engine.get_reduction_summary()

        self.assertEqual(summary["num_nodes"], 4)
        self.assertEqual(summary["reduced_classes"], 3)

        cycles = list(engine.generate_cycles())
        self.assertEqual(len(cycles), 1)

        # FIXED: Explicitly extract the first element before unpacking tuple
        cycle_path, total_weight = cycles[0]
        self.assertEqual(total_weight, 1)
        self.assertEqual(len(cycle_path), 3)

        normalized_path = set(tuple(sorted(pool)) for pool in cycle_path)
        expected_path = {tuple([0]), tuple([1]), tuple([2, 3])}

        self.assertEqual(normalized_path, expected_path)

    def test_02_iterative_leaf_pruning_cascades(self):
        """Tests heavy deep-tree dead-end elimination sweeps."""
        weights = [1, 1, 1, 0, 1, -1, 1]
        edges = [
            (0, 1), (1, 2), (2, 0),  # Valid Core Loop
            (2, 3), (3, 4), (4, 5),  # Long outgoing path (Dead end)
            (6, 0)                   # Long incoming path (Dead end)
        ]
        engine = GraphEngine(weights, edges)
        summary = engine.get_reduction_summary()
        
        self.assertEqual(summary["reduced_classes"], 3)
        cycles = list(engine.generate_cycles())
        self.assertEqual(len(cycles), 1)

    def test_03_tarjan_cross_component_edge_drops(self):
        """Tests that edges traveling between different SCCs disappear."""
        weights = [1, 1, 1, -1, 1]
        edges = [
            (0, 1), (1, 0),  # SCC A
            (1, 2),          # Cross-SCC Edge (Must be dropped!)
            (2, 3), (3, 4), (4, 2)  # SCC B
        ]
        engine = GraphEngine(weights, edges)
        cycles = list(engine.generate_cycles())
        
        self.assertEqual(len(cycles), 2)

    def test_04_early_weight_bounding_and_pruning(self):
        """Verifies that unviable negative branches drop immediately."""
        weights = [1, -1, -1, 0, 1]
        edges = [
            (0, 1), (1, 2), (2, 0),  # Negative Loop
            (0, 3), (3, 4), (4, 0)   # Positive Loop
        ]
        engine = GraphEngine(weights, edges)
        cycles = list(engine.generate_cycles())
        
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0][1], 2)

    def test_05_combinatorial_explosion_density(self):
        """Verifies zero-buffering streams correctly across fully dense K_5."""
        num_nodes = 5
        weights = [1] * num_nodes
        edges = [(u, v) for u in range(num_nodes) 
                        for v in range(num_nodes) if u != v]

        engine = GraphEngine(weights, edges)
        cycles = list(engine.generate_cycles())
        
        self.assertEqual(len(cycles), 84)
        for path, weight in cycles:
            self.assertEqual(weight, len(path))

    def test_06_alternating_sign_loops(self):
        """Checks calculation stability over alternating negative steps."""
        weights = [1, -1, 1, -1, 1]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        
        engine = GraphEngine(weights, edges)
        cycles = list(engine.generate_cycles())
        
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0][1], 1)

    def test_07_self_loop_filtering(self):
        """Validates behavior of length-1 recursive self loops."""
        weights = [1, -1, 1]
        edges = [
            (0, 0),  # Self loop (Weight = 1, valid)
            (0, 1), (1, 2), (2, 0) # Master loop (Weight = 1, valid)
        ]
        engine = GraphEngine(weights, edges)
        cycles = list(engine.generate_cycles())
        
        self.assertEqual(len(cycles), 2)

    def test_08_completely_disconnected_subgraphs(self):
        """Tests tracking boundary isolation over disjoint sets."""
        weights = [1, 1, 1, 1, -1, 1]
        edges = [
            (0, 1), (1, 0),         # Component 1 (Weight = 2)
            (2, 3), (3, 4), (4, 2), # Component 2 (Weight = 1)
            (5, 5)                  # Component 3 (Weight = 1)
        ]
        engine = GraphEngine(weights, edges)
        cycles = list(engine.generate_cycles())
        
        self.assertEqual(len(cycles), 3)


if __name__ == "__main__":
    unittest.main()
