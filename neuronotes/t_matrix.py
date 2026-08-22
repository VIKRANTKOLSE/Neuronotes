"""Cold-start routing over the prerequisite DAG (T-matrix)."""

from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


class ColdStartRouter:
    """Choose informative foundation concepts before response evidence exists."""

    def __init__(self, concept_graph_path: Optional[Path] = None):
        path = concept_graph_path or DATA_DIR / "concept_graph.csv"
        self.graph = nx.DiGraph()
        if path.exists():
            df = pd.read_csv(path)
            self.graph.add_edges_from(zip(df["source_concept"], df["target_concept"]))

    def recommend_concept(self, attempted_concepts: set[str]) -> Optional[str]:
        candidates = [node for node in self.graph.nodes if node != "ROOT" and node not in attempted_concepts]
        if not candidates:
            return None
        foundations = [node for node in candidates if all(parent == "ROOT" or parent in attempted_concepts
                       for parent in self.graph.predecessors(node))]
        pool = foundations or candidates
        return max(pool, key=lambda node: (len(nx.descendants(self.graph, node)), self.graph.out_degree(node), node))
