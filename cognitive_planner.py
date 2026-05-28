class TaskNode:
    def __init__(self, node_id: str, objective: str, dependencies: list = None, verifier_fn = None):
        self.node_id = node_id
        self.objective = objective
        self.dependencies = dependencies or []
        self.verifier_fn = verifier_fn
        self.status = "PENDING"  # PENDING, READY, EXECUTING, COMPLETED, FAILED
        self.result = None

class DAGPlanner:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node: TaskNode):
        self.nodes[node.node_id] = node

    def get_executable_nodes(self) -> list[TaskNode]:
        """Returns pending nodes whose dependencies are all COMPLETED."""
        ready_nodes = []
        for node in self.nodes.values():
            if node.status == "PENDING":
                deps_resolved = all(
                    dep in self.nodes and self.nodes[dep].status == "COMPLETED" 
                    for dep in node.dependencies
                )
                if deps_resolved:
                    node.status = "READY"
                    ready_nodes.append(node)
        return ready_nodes

    def mark_completed(self, node_id: str, result: str):
        if node_id in self.nodes:
            self.nodes[node_id].status = "COMPLETED"
            self.nodes[node_id].result = result

    def mark_failed(self, node_id: str):
        if node_id in self.nodes:
            self.nodes[node_id].status = "FAILED"

    def is_finished(self) -> bool:
        """Returns True if all nodes are COMPLETED or FAILED."""
        return all(node.status in ["COMPLETED", "FAILED"] for node in self.nodes.values())

    def has_failures(self) -> bool:
        return any(node.status == "FAILED" for node in self.nodes.values())
