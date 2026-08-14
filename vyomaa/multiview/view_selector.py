"""View selection and viewpoint novelty scoring."""

from __future__ import annotations

from typing import List, Set, Tuple
import numpy as np

from ..core.contracts import Observation
from .view_graph import ViewGraph, ViewPair


class ViewSelector:
    """Selects optimal keyframe subsets with high baseline disparity and rejects redundant frames."""

    @staticmethod
    def select_informative_views(
        view_graph: ViewGraph,
        min_overlap: float = 0.25,
        max_redundancy_overlap: float = 0.95,
    ) -> List[str]:
        """Identify key views providing wide angular baseline without extreme redundancy."""
        selected: Set[str] = set()
        all_views = list(view_graph.views.keys())
        if not all_views:
            return []

        # Start with the highest quality view
        best_root = max(
            all_views,
            key=lambda vid: view_graph.quality_scores[vid].overall_quality if vid in view_graph.quality_scores else 0.5,
        )
        selected.add(best_root)

        for candidate in all_views:
            if candidate in selected:
                continue

            # Check overlap with already selected views
            max_overlap_with_selected = 0.0
            has_sufficient_overlap = False

            for s in selected:
                edge = view_graph.edges.get((s, candidate)) or view_graph.edges.get((candidate, s))
                if edge and edge.is_valid_edge:
                    max_overlap_with_selected = max(max_overlap_with_selected, edge.overlap_score)
                    if edge.overlap_score >= min_overlap:
                        has_sufficient_overlap = True

            if has_sufficient_overlap and max_overlap_with_selected <= max_redundancy_overlap:
                selected.add(candidate)

        return sorted(list(selected))
