# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from raven.layers.tracking_raven import TrackingMemoryState, TrackingRaven
from raven.models.tracking.tracking_config import TrackingRavenConfig


class TrackingRavenModel(nn.Module):
    def __init__(
        self,
        config: TrackingRavenConfig,
        detection_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_projection = (
            nn.Identity()
            if config.input_dim == config.hidden_size
            else nn.Linear(config.input_dim, config.hidden_size)
        )
        self.tracking = TrackingRaven(
            hidden_size=config.hidden_size,
            appearance_slots=config.appearance_slots,
            motion_slots=config.motion_slots,
            position_slots=config.position_slots,
            occlusion_slots=config.occlusion_slots,
            topk=config.topk,
            short_forgetting=config.short_forgetting,
            long_forgetting=config.long_forgetting,
            trajectory_smoothing=config.trajectory_smoothing,
            reid_dim=config.reid_dim,
            use_raven_aggregation=config.use_raven_aggregation,
            raven_kwargs=config.raven_kwargs,
        )
        self.detection_head = detection_head

    def forward(
        self,
        frame_features: torch.Tensor,
        confidence: torch.Tensor,
        bbox: torch.Tensor,
        state: Optional[TrackingMemoryState] = None,
        occluded: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, torch.Tensor], TrackingMemoryState]:
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape [batch, objects, channels]")
        if not self.config.multi_object and frame_features.shape[1] != 1:
            raise ValueError("multi-object is disabled; expected a single object per sample")

        projected = self.input_projection(frame_features)
        outputs, next_state = self.tracking(
            frame_features=projected,
            confidence=confidence,
            bbox=bbox,
            state=state,
            occluded=occluded,
        )
        if self.detection_head is not None:
            outputs["detection"] = self.detection_head(outputs["aggregated_state"])
        return outputs, next_state
