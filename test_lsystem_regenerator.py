import unittest
import numpy as np
from lsystem_regenerator import LSystemRegenerator

class MockAgent:
    def __init__(self, resource):
        self.resource = resource

class TestLSystemRegenerator(unittest.TestCase):
    def test_initial_axiom(self):
        # Axiom initial: A
        r = LSystemRegenerator("Cell-001", ["Cell-002", "Cell-003", "Cell-004"])
        self.assertEqual(r.string, "A")
        self.assertEqual(r.agent_id, "Cell-001")
        self.assertEqual(r.original_neighbors, ["Cell-002", "Cell-003", "Cell-004"])

    def test_rule_1_rewriting(self):
        # Rule 1: A rewrites to W(5) M
        r = LSystemRegenerator("Cell-001", ["Cell-002", "Cell-003", "Cell-004"])
        res = r.step({}, [])
        self.assertEqual(r.string, "W(5) M")
        self.assertIsNone(res)

    def test_rule_2_and_3_wait_and_activation(self):
        # Rule 2 & 3: W(n) -> W(n-1) M. When n reaches 0, rewrites to process M, which rewrites to T C(3).
        r = LSystemRegenerator("Cell-001", ["Cell-002", "Cell-003", "Cell-004"])
        
        # Step 1: A -> W(5) M
        r.step({}, [])
        self.assertEqual(r.string, "W(5) M")
        
        # Step 2 to 5: W(5) M -> ... -> W(2) M
        for i in range(4):
            r.step({}, [])
            
        self.assertEqual(r.string, "W(1) M")
        
        # Step 6: W(1) M -> M (which activates and rewrites to T C(3))
        r.step({}, [])
        self.assertEqual(r.string, "T C(3)")

    def test_rule_4_and_5_local_regeneration(self):
        # Rule 5: local mean > global mean -> connects to original neighbors
        r = LSystemRegenerator("Cell-001", ["Cell-002", "Cell-003", "Cell-004"])
        r.string = "T C(3)"
        
        agents = {
            "Cell-001": MockAgent(0),
            "Cell-002": MockAgent(90),
            "Cell-003": MockAgent(85),
            "Cell-004": MockAgent(80),
            "Cell-005": MockAgent(60),
        }
        active_ids = ["Cell-002", "Cell-003", "Cell-004", "Cell-005"]
        
        # Local Mean = (90 + 85 + 80) / 3 = 85
        # Global Mean = (90 + 85 + 80 + 60) / 4 = 78.75
        # Local Mean > Global Mean, should return original neighbors
        res = r.step(agents, active_ids)
        self.assertEqual(res, ["Cell-002", "Cell-003", "Cell-004"])
        self.assertEqual(r.string, "")

    def test_rule_4_and_6_plasticity(self):
        # Rule 6: local mean <= global mean -> connects to top 3 resource agents
        r = LSystemRegenerator("Cell-001", ["Cell-002", "Cell-003", "Cell-004"])
        r.string = "T C(3)"
        
        agents = {
            "Cell-001": MockAgent(0),
            "Cell-002": MockAgent(10),
            "Cell-003": MockAgent(20),
            "Cell-004": MockAgent(30),
            "Cell-005": MockAgent(90),
            "Cell-006": MockAgent(80),
            "Cell-007": MockAgent(70),
        }
        active_ids = ["Cell-002", "Cell-003", "Cell-004", "Cell-005", "Cell-006", "Cell-007"]
        
        # Local Mean = (10 + 20 + 30) / 3 = 20
        # Global Mean = (10 + 20 + 30 + 90 + 80 + 70) / 6 = 50
        # Local Mean <= Global Mean, should return agents with highest resources: Cell-005, Cell-006, Cell-007
        res = r.step(agents, active_ids)
        self.assertEqual(res, ["Cell-005", "Cell-006", "Cell-007"])
        self.assertEqual(r.string, "")

if __name__ == "__main__":
    unittest.main()
