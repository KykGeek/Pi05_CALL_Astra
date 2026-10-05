"""Fresh π0.5 intent proposals isolated from the CALL policy RNG stream."""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, Mapping, Optional
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    observation_id: str
    env_step: int
    actions: np.ndarray
    nominal_trajectory: list[dict[str, Any]]

    def public_packet(self) -> Dict[str, Any]:
        # The normalized proposal itself is deliberately omitted. Astra sees
        # the runtime-calibrated, dimensionful nominal trajectory as context,
        # never as an executable command.
        return {
            "proposal_id": self.proposal_id,
            "observation_id": self.observation_id,
            "env_step": int(self.env_step),
            "executed": False,
            "control_steps": len(self.nominal_trajectory),
            "nominal_trajectory": [dict(row) for row in self.nominal_trajectory],
            "interpretation": "unexecuted_pi05_proposal_not_a_recovery_command",
        }


class Pi05ProposalService:
    """Run fresh π0.5 inference without touching normal sampling or CALL state."""

    def __init__(self, policy: Any, inference: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 observation_encoder: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
                 adapter: Any) -> None:
        self.policy = policy
        self.inference = inference
        self.observation_encoder = observation_encoder
        self.adapter = adapter
        try:
            import jax
            self._jax = jax
            self._proposal_rng = jax.random.fold_in(policy._rng, 0xA57A)
        except Exception as error:
            raise RuntimeError("isolated_proposal_rng_unavailable") from error
        self._lock = threading.RLock()
        self._current: Optional[Proposal] = None

    @property
    def current(self) -> Optional[Proposal]:
        with self._lock:
            return self._current

    def install_call_proposal(self, checkpoint: Any, actions: Any) -> Proposal:
        with self._lock:
            validated = self.adapter.validate_pi05_chunk(actions)
            proposal = self._make(checkpoint, validated)
            self._current = proposal
            return proposal

    def propose(self, checkpoint: Any, task_instruction: str) -> Proposal:
        with self._lock:
            if self._current is not None and self._current.observation_id == str(checkpoint.observation_id):
                return self._current
            encoded = self.observation_encoder(checkpoint.raw, str(task_instruction))
            normal_rng = self.policy._rng
            self.policy._rng = self._proposal_rng
            try:
                result = self.inference(encoded)
            finally:
                self._proposal_rng = self.policy._rng
                self.policy._rng = normal_rng
            if not isinstance(result, Mapping) or "actions" not in result:
                raise RuntimeError("pi05_proposal_inference_missing_actions")
            validated = self.adapter.validate_pi05_chunk(result["actions"])
            proposal = self._make(checkpoint, validated)
            self._current = proposal
            return proposal

    def _make(self, checkpoint: Any, actions: np.ndarray) -> Proposal:
        pose = self.adapter.read_pose(checkpoint.raw)
        trajectory = self.adapter.native_action_to_site_trajectory(actions, pose)
        return Proposal(
            proposal_id="pi05-" + str(uuid4()),
            observation_id=str(checkpoint.observation_id),
            env_step=int(checkpoint.env_step),
            actions=actions.copy(),
            nominal_trajectory=trajectory,
        )
