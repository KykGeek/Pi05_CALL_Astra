"""Small frozen-feature MLP used by the CALL_ASTRA V1 training pipeline."""

from __future__ import annotations


SEMANTIC_DIM = 2048
ACTION_DIM = 1024
STATE_STATS_DIM = 44
FEATURE_SET_DIMS = {
    "full": SEMANTIC_DIM + ACTION_DIM + STATE_STATS_DIM,
    "state_only": STATE_STATS_DIM,
    "semantic_only": SEMANTIC_DIM,
    "action_only": ACTION_DIM,
}
_FEATURE_BRANCHES = {
    "full": ("semantic", "action", "state"),
    "state_only": ("state",),
    "semantic_only": ("semantic",),
    "action_only": ("action",),
}
_BRANCH_INPUT_DIMS = {
    "semantic": SEMANTIC_DIM,
    "action": ACTION_DIM,
    "state": STATE_STATS_DIM,
}
_BRANCH_OUTPUT_DIMS = {"semantic": 128, "action": 128, "state": 64}


def build_call_assist_mlp(*, hidden_size: int = 256, feature_set: str = "full"):
    """Create the V1 AssistMLP without importing Torch for data-only tooling."""
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if feature_set not in _FEATURE_BRANCHES:
        raise ValueError(f"unsupported feature set {feature_set!r}")
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build the CALL_ASTRA head") from exc

    class CallAssistMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_set = feature_set
            self.branches = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(_BRANCH_INPUT_DIMS[name], _BRANCH_OUTPUT_DIMS[name]),
                        nn.GELU(),
                    )
                    for name in _FEATURE_BRANCHES[feature_set]
                }
            )
            body_input_dim = sum(
                _BRANCH_OUTPUT_DIMS[name] for name in _FEATURE_BRANCHES[feature_set]
            )
            self.body = nn.Sequential(
                nn.LayerNorm(body_input_dim),
                nn.Linear(body_input_dim, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
            self.help_head = nn.Linear(hidden_size, 1)

        def forward(self, features):
            expected_dim = FEATURE_SET_DIMS[feature_set]
            if features.shape[-1] != expected_dim:
                raise ValueError(
                    f"CALL_ASTRA V1 {feature_set} feature vector must have dimension "
                    f"{expected_dim}, "
                    f"got {features.shape[-1]}"
                )
            if feature_set == "full":
                segments = {
                    "semantic": features[..., :SEMANTIC_DIM],
                    "action": features[..., SEMANTIC_DIM : SEMANTIC_DIM + ACTION_DIM],
                    "state": features[..., SEMANTIC_DIM + ACTION_DIM :],
                }
            else:
                segments = {_FEATURE_BRANCHES[feature_set][0]: features}
            projected = torch.cat(
                [self.branches[name](segments[name]) for name in _FEATURE_BRANCHES[feature_set]],
                dim=-1,
            )
            return self.help_head(self.body(projected)).squeeze(-1)

    return CallAssistMLP()
