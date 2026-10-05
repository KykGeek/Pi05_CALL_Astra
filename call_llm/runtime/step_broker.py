"""Only physical step owner; host checkpoint authority and confirmed counts."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import threading
from uuid import uuid4
import numpy as np

from .journal import ExecutionUncertain


@dataclass(frozen=True)
class Checkpoint:
    episode_id: str
    checkpoint_id: str
    observation_id: str
    env_step: int
    raw: dict


@dataclass(frozen=True)
class StepResult:
    checkpoint: Checkpoint
    task_succeeded: bool
    environment_ended: bool


class StepBroker:
    def __init__(self, env, initial_observation, *, episode_id, journal,
                 history=None, pi05_action_validator=None, astra_action_validator=None):
        self._env = env
        self.episode_id = episode_id
        self.journal = journal
        self.history = history
        self.pi05_action_validator = pi05_action_validator
        self.astra_action_validator = astra_action_validator
        self.frame_sink = None
        self.env_steps = self.phase_pi05_steps = 0
        self.pi05_total_steps = 0
        # Host-side count of actions executed by the recovery model.  The
        # legacy astra_total_steps property below remains as a compatibility
        # alias because old reports already use that field name.
        self.recovery_model_total_steps = 0
        self.owner = 'pi05'
        self.uncertain = False
        self.task_succeeded = False
        self.environment_ended = False
        self.stop_reason = None
        self._lock = threading.RLock()
        self._checkpoint = self._capture(initial_observation)
        self._registry = {self._checkpoint.checkpoint_id: self._checkpoint}

    def _capture(self, observation):
        if not isinstance(observation, dict):
            raise ValueError('invalid_environment_observation')
        return Checkpoint(self.episode_id, str(uuid4()), str(uuid4()), self.env_steps,
                          copy.deepcopy(observation))

    def current_checkpoint(self):
        with self._lock:
            if self.uncertain:
                raise ExecutionUncertain('checkpoint_not_known')
            return copy.deepcopy(self._checkpoint)

    def lookup_checkpoint(self, identifier):
        with self._lock:
            return copy.deepcopy(self._registry.get(identifier))

    def transfer(self, expected, new):
        with self._lock:
            if new not in ('pi05', 'astra', 'stopped') or self.owner != expected:
                raise RuntimeError('invalid_owner_transfer')
            if self.uncertain or self.environment_ended or self.task_succeeded:
                raise RuntimeError('episode_not_controllable')
            self.owner = new

    def stop(self, reason):
        with self._lock:
            self.owner, self.stop_reason = 'stopped', str(reason)

    def begin_pi05_phase(self):
        """Resume pi05 ownership without restarting its episode budget."""
        with self._lock:
            if self.owner != 'astra' or self.uncertain:
                raise RuntimeError('phase_reset_without_handoff')
            # Keep the cumulative pi05 step count.  A handback must not grant
            # pi05 another max-step window in the same episode.

    def step(self, action, *, source, decision_id=None):
        with self._lock:
            if self.owner != source or source not in ('pi05', 'astra'):
                raise RuntimeError('wrong_action_owner')
            if self.uncertain or self.environment_ended or self.task_succeeded:
                raise RuntimeError('episode_not_controllable')
            a = np.asarray(action, dtype=np.float64)
            if a.shape != (7,) or not np.isfinite(a).all():
                raise ValueError('invalid_native_action')
            if source == 'astra':
                if not decision_id or self.astra_action_validator is None:
                    raise RuntimeError('unvalidated_astra_action')
                self.astra_action_validator(a)
            elif self.pi05_action_validator is not None:
                self.pi05_action_validator(a)
            before = self._checkpoint
            attempt_id = str(uuid4())
            try:
                self.journal.append(dict(event='step_started', attempt_id=attempt_id,
                    source=source, decision_id=decision_id, before_step=self.env_steps,
                    observation_id=before.observation_id, action=a.tolist()))
            except BaseException:
                self.stop('journal_failed_before_step')
                raise
            try:
                raw, reward, done, info = self._env.step(a.tolist())
            except BaseException as error:
                self.uncertain = True
                self.stop('env_step_uncertain')
                try:
                    self.journal.append(dict(event='step_uncertain', attempt_id=attempt_id,
                                             confirmed_steps=self.env_steps))
                finally:
                    raise ExecutionUncertain('env_step_uncertain') from error
            # Never undo counts if subsequent capture/history/check_success fails.
            self.env_steps += 1
            if source == 'pi05':
                self.pi05_total_steps += 1
                self.phase_pi05_steps += 1
            else:
                self.recovery_model_total_steps += 1
            try:
                self._checkpoint = self._capture(raw)
                self._registry[self._checkpoint.checkpoint_id] = self._checkpoint
                self.task_succeeded = bool(self._env.check_success())
                self.environment_ended = bool(done)
                if self.frame_sink is not None:
                    try:
                        self.frame_sink(self._checkpoint.raw, self.env_steps, source)
                    except Exception as video_error:
                        self.journal.append(dict(
                            event='video_frame_error', env_step=self.env_steps,
                            error=type(video_error).__name__,
                        ))
                self.journal.append(dict(event='step_finished', attempt_id=attempt_id,
                    source=source, decision_id=decision_id, env_step=self.env_steps,
                    observation_id=self._checkpoint.observation_id,
                    task_succeeded=self.task_succeeded, environment_ended=self.environment_ended))
                if self.history is not None:
                    self.history.record_transition(before, a.copy(), self._checkpoint,
                                                   source, decision_id)
            except BaseException:
                self.stop('post_step_audit_failed')
                raise
            return StepResult(copy.deepcopy(self._checkpoint), self.task_succeeded,
                              self.environment_ended)

    @property
    def astra_total_steps(self):
        """Deprecated compatibility alias; Astra is the model name."""
        return self.recovery_model_total_steps
