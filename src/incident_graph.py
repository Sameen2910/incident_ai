import networkx as nx

class IncidentGraph:

    def __init__(self):

        self.graph = nx.Graph()

    def add_incident(self, incident_id, root_cause):

        self.graph.add_node(incident_id, root=root_cause)

    def find_by_root(self, root):

        related = []

        for node, data in self.graph.nodes(data=True):

            if data.get("root") == root:
                related.append(node)

        return related