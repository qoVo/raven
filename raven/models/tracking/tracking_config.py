# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig


class TrackingRavenConfig(PretrainedConfig):
    model_type = "tracking-raven"

    def __init__(
        self,
        input_dim: int = 256,
        hidden_size: int = 256,
        appearance_slots: int = 8,
        motion_slots: int = 4,
        position_slots: int = 4,
        occlusion_slots: int = 4,
        topk: int = 2,
        short_forgetting: float = 0.15,
        long_forgetting: float = 0.03,
        trajectory_smoothing: float = 0.85,
        reid_dim: int = 128,
        multi_object: bool = True,
        use_raven_aggregation: bool = True,
        raven_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.appearance_slots = appearance_slots
        self.motion_slots = motion_slots
        self.position_slots = position_slots
        self.occlusion_slots = occlusion_slots
        self.topk = topk
        self.short_forgetting = short_forgetting
        self.long_forgetting = long_forgetting
        self.trajectory_smoothing = trajectory_smoothing
        self.reid_dim = reid_dim
        self.multi_object = multi_object
        self.use_raven_aggregation = use_raven_aggregation
        self.raven_kwargs = raven_kwargs or {}
        super().__init__(**kwargs)
