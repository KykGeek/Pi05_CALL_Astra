"""Static-competence CALL decision rules for the π0.5 control loop."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol


DecisionRule = Literal["instant", "2_of_3", "3_of_5", "4_of_5", "hysteresis"]


@dataclass(frozen=True)
class CallDecision:
    p_help: float
    decision: str
    reason: str
    threshold: float | None
    hard_threshold: float | None
    recent_scores: list[float]
    decision_rule: str
    cooldown_remaining: int = 0
    cooldown_suppressed: bool = False
    min_steps_suppressed: bool = False


class DecisionEngine(Protocol):
    def evaluate(self, p_help: float) -> CallDecision: ...

    def reset(self) -> None: ...


class CallDecisionEngine:
    """Turn calibrated help probabilities into CONTINUE or CALL_ASTRA.

    The hard threshold remains an immediate safety path. Otherwise, the
    confirmation rule is applied to the configured recent-query window.
    """

    _RULES = {"instant", "2_of_3", "3_of_5", "4_of_5", "hysteresis"}

    def __init__(
        self,
        *,
        threshold: float,
        hard_threshold: float | None,
        rule: DecisionRule = "4_of_5",
        hysteresis_margin: float = 0.1,
        hysteresis_confirm_queries: int = 2,
        cooldown_queries: int = 0,
    ) -> None:
        self._validate_probability("threshold", threshold)
        if hard_threshold is not None:
            self._validate_probability("hard_threshold", hard_threshold)
        if hard_threshold is not None and hard_threshold < threshold:
            raise ValueError("hard_threshold must be greater than or equal to threshold")
        if rule not in self._RULES:
            raise ValueError(f"unsupported decision rule: {rule}")
        if not math.isfinite(hysteresis_margin) or not 0.0 <= hysteresis_margin <= 1.0:
            raise ValueError("hysteresis_margin must be finite and in [0, 1]")
        if hysteresis_confirm_queries < 1:
            raise ValueError("hysteresis_confirm_queries must be positive")
        if int(cooldown_queries) < 0:
            raise ValueError("cooldown_queries must be nonnegative")

        self.threshold = float(threshold)
        self.hard_threshold = (
            None if hard_threshold is None else float(hard_threshold)
        )
        self.rule: DecisionRule = rule
        self.hysteresis_margin = float(hysteresis_margin)
        self.hysteresis_confirm_queries = int(hysteresis_confirm_queries)
        self.cooldown_queries = int(cooldown_queries)
        self._confirmation_window = 5 if rule in {"3_of_5", "4_of_5"} else 3
        self._confirmation_count = (
            4 if rule == "4_of_5" else 3 if rule == "3_of_5" else 2
        )
        self.recent_scores: list[float] = []
        self._hysteresis_streak = 0
        self._cooldown_remaining = 0

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    def reset(self) -> None:
        self.recent_scores.clear()
        self._hysteresis_streak = 0
        self._cooldown_remaining = 0

    def evaluate(self, p_help: float) -> CallDecision:
        score = float(p_help)
        self._validate_probability("p_help", score)
        self.recent_scores = (self.recent_scores + [score])[-self._confirmation_window :]

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return CallDecision(
                p_help=score,
                decision="CONTINUE",
                reason="cooldown_active",
                threshold=self.threshold,
                hard_threshold=self.hard_threshold,
                recent_scores=list(self.recent_scores),
                decision_rule=self.rule,
                cooldown_remaining=self._cooldown_remaining,
                cooldown_suppressed=True,
            )

        call = False
        if self.hard_threshold is not None and score >= self.hard_threshold:
            call = True
            reason = "hard_threshold"
        elif self.rule == "instant":
            call = score >= self.threshold
            reason = "threshold" if call else "below_threshold"
        elif self.rule in {"2_of_3", "3_of_5", "4_of_5"}:
            call = (
                len(self.recent_scores) == self._confirmation_window
                and sum(value >= self.threshold for value in self.recent_scores)
                >= self._confirmation_count
            )
            reason_names = {
                "2_of_3": "two_of_three",
                "3_of_5": "three_of_five",
                "4_of_5": "four_of_five",
            }
            reason_name = reason_names[self.rule]
            reason = reason_name if call else f"{reason_name}_not_met"
        else:
            lower_edge = max(0.0, self.threshold - self.hysteresis_margin)
            if score < lower_edge:
                self._hysteresis_streak = 0
            elif score >= self.threshold:
                self._hysteresis_streak += 1
            call = self._hysteresis_streak >= self.hysteresis_confirm_queries
            reason = "hysteresis_confirmed" if call else "hysteresis_pending"

        if call:
            self._cooldown_remaining = self.cooldown_queries

        return CallDecision(
            p_help=score,
            decision="CALL_ASTRA" if call else "CONTINUE",
            reason=reason,
            threshold=self.threshold,
            hard_threshold=self.hard_threshold,
            recent_scores=list(self.recent_scores),
            decision_rule=self.rule,
            cooldown_remaining=self._cooldown_remaining,
            cooldown_suppressed=False,
        )


class BudgetDecisionEngine:
    """Fixed-budget and never-call baselines with the same controller contract."""

    _STRATEGIES = {"never_call", "fixed_step", "fixed_query"}

    def __init__(self, strategy: str, *, budget: int | None = None) -> None:
        if strategy not in self._STRATEGIES:
            raise ValueError(f"unsupported baseline strategy: {strategy}")
        if strategy == "never_call":
            if budget is not None:
                raise ValueError("never_call does not accept a budget")
        elif budget is None or int(budget) <= 0:
            raise ValueError(f"{strategy} requires a positive budget")
        self.strategy = strategy
        self.budget = None if budget is None else int(budget)
        self.recent_scores: list[float] = []
        self._env_step: int | None = None
        self._policy_query_idx: int | None = None

    def set_context(self, *, env_step: int, policy_query_idx: int) -> None:
        if int(env_step) < 0 or int(policy_query_idx) < 0:
            raise ValueError("baseline context indices must be nonnegative")
        self._env_step = int(env_step)
        self._policy_query_idx = int(policy_query_idx)

    def reset(self) -> None:
        self.recent_scores.clear()
        self._env_step = None
        self._policy_query_idx = None

    def evaluate(self, p_help: float) -> CallDecision:
        score = float(p_help)
        CallDecisionEngine._validate_probability("p_help", score)
        self.recent_scores = (self.recent_scores + [score])[-3:]
        if self.strategy == "never_call":
            call = False
            reason = "never_call_baseline"
        else:
            if self._env_step is None or self._policy_query_idx is None:
                raise RuntimeError("fixed-budget baseline context was not set for this query")
            if self.strategy == "fixed_step":
                call = self._env_step >= int(self.budget)
                reason = "fixed_step_budget_reached" if call else "fixed_step_budget_pending"
            else:
                # Query indices are zero-based; a budget of N calls on query N-1.
                call = self._policy_query_idx + 1 >= int(self.budget)
                reason = "fixed_query_budget_reached" if call else "fixed_query_budget_pending"
        return CallDecision(
            p_help=score,
            decision="CALL_ASTRA" if call else "CONTINUE",
            reason=reason,
            threshold=None,
            hard_threshold=None,
            recent_scores=list(self.recent_scores),
            decision_rule=f"baseline:{self.strategy}",
        )


class OracleCompetenceDecisionEngine:
    """Recovery-Assets-only upper-bound rule driven by measured q_pi."""

    def __init__(self, *, low_competence_threshold: float = 0.2) -> None:
        CallDecisionEngine._validate_probability(
            "low_competence_threshold", low_competence_threshold
        )
        self.low_competence_threshold = float(low_competence_threshold)
        self._q_pi: float | None = None
        self.recent_scores: list[float] = []

    def set_context(self, *, q_pi: float) -> None:
        value = float(q_pi)
        CallDecisionEngine._validate_probability("q_pi", value)
        self._q_pi = value

    def reset(self) -> None:
        self._q_pi = None
        self.recent_scores.clear()

    def evaluate(self, p_help: float) -> CallDecision:
        score = float(p_help)
        CallDecisionEngine._validate_probability("p_help", score)
        if self._q_pi is None:
            raise RuntimeError("oracle competence context is missing for this state")
        self.recent_scores = (self.recent_scores + [score])[-3:]
        call = self._q_pi <= self.low_competence_threshold
        return CallDecision(
            p_help=score,
            decision="CALL_ASTRA" if call else "CONTINUE",
            reason="oracle_low_competence" if call else "oracle_continue",
            threshold=None,
            hard_threshold=None,
            recent_scores=list(self.recent_scores),
            decision_rule="baseline:oracle_competence",
        )
