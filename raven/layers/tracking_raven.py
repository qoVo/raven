from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from raven.layers.raven import RavenAttention


@dataclass
class TrackingMemoryState:
    appearance_short: torch.Tensor
    appearance_long: torch.Tensor
    motion_short: torch.Tensor
    motion_long: torch.Tensor
    position_short: torch.Tensor
    position_long: torch.Tensor
    occlusion_short: torch.Tensor
    occlusion_long: torch.Tensor
    prev_bbox: torch.Tensor
    prev_position: torch.Tensor
    prev_occlusion: torch.Tensor


class TrackingRaven(nn.Module):
    def __init__(
        self,
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
        use_raven_aggregation: bool = True,
        raven_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.appearance_slots = appearance_slots
        self.motion_slots = motion_slots
        self.position_slots = position_slots
        self.occlusion_slots = occlusion_slots
        self.topk = topk
        self.short_forgetting = short_forgetting
        self.long_forgetting = long_forgetting
        self.trajectory_smoothing = trajectory_smoothing

        routing_input_dim = hidden_size * 2 + 6  # current + historical + confidence/bbox/occlusion
        self.appearance_router = nn.Linear(routing_input_dim, appearance_slots)
        self.motion_router = nn.Linear(routing_input_dim, motion_slots)
        self.position_router = nn.Linear(routing_input_dim, position_slots)
        self.occlusion_router = nn.Linear(routing_input_dim, occlusion_slots)

        self.appearance_update = nn.Linear(hidden_size, hidden_size)
        self.motion_update = nn.Linear(hidden_size + 4, hidden_size)
        self.position_update = nn.Linear(4, hidden_size)
        self.occlusion_update = nn.Linear(hidden_size + 1, hidden_size)

        self.aggregator = None
        if use_raven_aggregation:
            kwargs = raven_kwargs or {}
            self.aggregator = RavenAttention(hidden_size=hidden_size, topk=max(topk, 1), **kwargs)

        self.fusion = nn.Linear(hidden_size * 6 + 9, hidden_size)
        self.bbox_head = nn.Linear(hidden_size, 4)
        self.existence_head = nn.Linear(hidden_size, 1)
        self.reid_head = nn.Linear(hidden_size, reid_dim)
        self.occlusion_head = nn.Linear(hidden_size, 1)

    @staticmethod
    def _empty_memory(
        batch_size: int,
        num_objects: int,
        slots: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, num_objects, slots, hidden_size, dtype=dtype, device=device)

    def init_state(
        self,
        batch_size: int,
        num_objects: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> TrackingMemoryState:
        zeros_bbox = torch.zeros(batch_size, num_objects, 4, dtype=dtype, device=device)
        zeros_occ = torch.zeros(batch_size, num_objects, 1, dtype=dtype, device=device)
        return TrackingMemoryState(
            appearance_short=self._empty_memory(
                batch_size, num_objects, self.appearance_slots, self.hidden_size, dtype, device
            ),
            appearance_long=self._empty_memory(
                batch_size, num_objects, self.appearance_slots, self.hidden_size, dtype, device
            ),
            motion_short=self._empty_memory(batch_size, num_objects, self.motion_slots, self.hidden_size, dtype, device),
            motion_long=self._empty_memory(batch_size, num_objects, self.motion_slots, self.hidden_size, dtype, device),
            position_short=self._empty_memory(
                batch_size, num_objects, self.position_slots, self.hidden_size, dtype, device
            ),
            position_long=self._empty_memory(
                batch_size, num_objects, self.position_slots, self.hidden_size, dtype, device
            ),
            occlusion_short=self._empty_memory(
                batch_size, num_objects, self.occlusion_slots, self.hidden_size, dtype, device
            ),
            occlusion_long=self._empty_memory(
                batch_size, num_objects, self.occlusion_slots, self.hidden_size, dtype, device
            ),
            prev_bbox=zeros_bbox,
            prev_position=zeros_bbox,
            prev_occlusion=zeros_occ,
        )

    def _sparse_weights(self, logits: torch.Tensor) -> torch.Tensor:
        topk = min(self.topk, logits.shape[-1])
        topk_values, topk_indices = logits.topk(topk, dim=-1)
        weights = torch.softmax(topk_values, dim=-1)
        sparse = torch.zeros_like(logits).scatter_(-1, topk_indices, weights)
        return sparse

    def _update_memory(
        self,
        short_memory: torch.Tensor,
        long_memory: torch.Tensor,
        update_value: torch.Tensor,
        routing_weights: torch.Tensor,
        update_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        write_value = update_value.unsqueeze(2)
        routing_weights = routing_weights.unsqueeze(-1)
        gated_weights = routing_weights * update_gate.unsqueeze(-1)
        short_decay = 1.0 - self.short_forgetting * update_gate.unsqueeze(-1)
        long_decay = 1.0 - self.long_forgetting * update_gate.unsqueeze(-1)
        short_updated = short_memory * short_decay + gated_weights * write_value
        long_updated = long_memory * long_decay + gated_weights * short_updated.detach()
        return short_updated, long_updated

    @staticmethod
    def _read_channel(short_memory: torch.Tensor, long_memory: torch.Tensor) -> torch.Tensor:
        return 0.6 * short_memory.mean(dim=2) + 0.4 * long_memory.mean(dim=2)

    def forward(
        self,
        frame_features: torch.Tensor,
        confidence: torch.Tensor,
        bbox: torch.Tensor,
        state: Optional[TrackingMemoryState] = None,
        occluded: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, torch.Tensor], TrackingMemoryState]:
        batch_size, num_objects, _ = frame_features.shape
        if state is None:
            state = self.init_state(batch_size, num_objects, frame_features.dtype, frame_features.device)

        if confidence.ndim == 2:
            confidence = confidence.unsqueeze(-1)
        if occluded is None:
            occluded = state.prev_occlusion
        elif occluded.ndim == 2:
            occluded = occluded.unsqueeze(-1)

        position_offset = bbox - state.prev_position
        historical_appearance = self._read_channel(state.appearance_short, state.appearance_long)
        router_inputs = torch.cat(
            [frame_features, historical_appearance, confidence, position_offset, occluded],
            dim=-1,
        )

        appearance_route = self._sparse_weights(self.appearance_router(router_inputs))
        motion_route = self._sparse_weights(self.motion_router(router_inputs))
        position_route = self._sparse_weights(self.position_router(router_inputs))
        occlusion_route = self._sparse_weights(self.occlusion_router(router_inputs))

        confidence_gate = confidence.clamp(0.0, 1.0)
        occlusion_gate = (1.0 - occluded).clamp(0.0, 1.0)

        appearance_update = torch.tanh(self.appearance_update(frame_features))
        motion_input = torch.cat([frame_features, position_offset], dim=-1)
        motion_update = torch.tanh(self.motion_update(motion_input))
        position_update = torch.tanh(self.position_update(bbox))
        occlusion_update = torch.tanh(self.occlusion_update(torch.cat([frame_features, occluded], dim=-1)))

        appearance_gate = confidence_gate * occlusion_gate
        motion_gate = confidence_gate
        position_gate = confidence_gate
        occlusion_memory_gate = torch.maximum(confidence_gate, occluded)

        appearance_short, appearance_long = self._update_memory(
            state.appearance_short, state.appearance_long, appearance_update, appearance_route, appearance_gate
        )
        motion_short, motion_long = self._update_memory(
            state.motion_short, state.motion_long, motion_update, motion_route, motion_gate
        )
        position_short, position_long = self._update_memory(
            state.position_short, state.position_long, position_update, position_route, position_gate
        )
        occlusion_short, occlusion_long = self._update_memory(
            state.occlusion_short, state.occlusion_long, occlusion_update, occlusion_route, occlusion_memory_gate
        )

        appearance_read = self._read_channel(appearance_short, appearance_long)
        motion_read = self._read_channel(motion_short, motion_long)
        position_read = self._read_channel(position_short, position_long)
        occlusion_read = self._read_channel(occlusion_short, occlusion_long)

        fused = torch.cat(
            [
                frame_features,
                historical_appearance,
                appearance_read,
                motion_read,
                position_read,
                occlusion_read,
                state.prev_bbox,
                position_offset,
                occluded,
            ],
            dim=-1,
        )
        fused = torch.tanh(self.fusion(fused))

        if self.aggregator is not None:
            fused, _, _ = self.aggregator(fused)

        raw_bbox = self.bbox_head(fused)
        smoothed_bbox = self.trajectory_smoothing * state.prev_bbox + (1.0 - self.trajectory_smoothing) * raw_bbox
        existence = torch.sigmoid(self.existence_head(fused))
        stable_embedding = F.normalize(self.reid_head(appearance_read + occlusion_read), dim=-1)
        occlusion_prob = torch.sigmoid(self.occlusion_head(occlusion_read))

        next_state = TrackingMemoryState(
            appearance_short=appearance_short,
            appearance_long=appearance_long,
            motion_short=motion_short,
            motion_long=motion_long,
            position_short=position_short,
            position_long=position_long,
            occlusion_short=occlusion_short,
            occlusion_long=occlusion_long,
            prev_bbox=smoothed_bbox.detach(),
            prev_position=bbox.detach(),
            prev_occlusion=occlusion_prob.detach(),
        )

        outputs = {
            "bbox": smoothed_bbox,
            "bbox_raw": raw_bbox,
            "existence_prob": existence,
            "reid_embedding": stable_embedding,
            "trajectory": smoothed_bbox,
            "occlusion_prob": occlusion_prob,
            "aggregated_state": fused,
            "confidence_gate": confidence_gate,
            "occlusion_gate": appearance_gate,
        }
        return outputs, next_state
