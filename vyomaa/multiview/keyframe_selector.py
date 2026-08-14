import logging
from typing import List, Dict, Any, Optional
import numpy as np
from vyomaa.multiview.contracts import ViewSet

logger = logging.getLogger("vyomaa.multiview.keyframe_selector")

class KeyframeSelector:
    def __init__(
        self,
        max_frames: int = 100,
        temporal_window: int = 5,
        sharpness_threshold: float = 100.0,
        similarity_threshold: float = 0.85
    ):
        self.max_frames = max_frames
        self.temporal_window = temporal_window
        self.sharpness_threshold = sharpness_threshold
        self.similarity_threshold = similarity_threshold

    def select_keyframes(
        self,
        observation_ids: List[str],
        timestamps: List[float],
        image_paths: List[str],
        quality_scores: Optional[List[float]] = None
    ) -> ViewSet:
        n = len(observation_ids)
        if n == 0:
            return ViewSet(observation_ids=[], timestamps=[], keyframe_flags=[], image_paths=[])

        if quality_scores is None:
            quality_scores = [1.0] * n

        keyframe_flags = [False] * n
        keyframe_flags[0] = True
        last_selected_idx = 0

        selected_count = 1
        for i in range(1, n):
            time_diff = timestamps[i] - timestamps[last_selected_idx] if timestamps else float(i)
            quality_ok = quality_scores[i] >= (self.sharpness_threshold * 0.01)

            if (i - last_selected_idx >= self.temporal_window) or (quality_ok and time_diff > 1.0):
                keyframe_flags[i] = True
                last_selected_idx = i
                selected_count += 1
                if selected_count >= self.max_frames:
                    break

        if n > 1 and not keyframe_flags[-1]:
            keyframe_flags[-1] = True

        selected_ids = [obs for obs, flag in zip(observation_ids, keyframe_flags) if flag]
        selected_ts = [ts for ts, flag in zip(timestamps, keyframe_flags) if flag] if timestamps else []
        selected_paths = [p for p, flag in zip(image_paths, keyframe_flags) if flag]
        selected_scores = [q for q, flag in zip(quality_scores, keyframe_flags) if flag]

        provenance = {
            "selector": "KeyframeSelector",
            "total_input_frames": n,
            "selected_keyframes": len(selected_ids),
            "temporal_window": self.temporal_window
        }

        return ViewSet(
            observation_ids=selected_ids,
            timestamps=selected_ts,
            keyframe_flags=[True] * len(selected_ids),
            image_paths=selected_paths,
            image_quality_scores=selected_scores,
            selected_view_confidence=selected_scores,
            source_modality="rgb",
            provenance=provenance
        )
