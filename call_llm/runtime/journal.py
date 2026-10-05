"""Durable host-only execution journal. Unfinished decisions are never replayed."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import threading

from .protocol import ProtocolError


class ExecutionUncertain(RuntimeError):
    pass


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.decisions = {}
        self.consumed = set()
        if self.path.exists():
            raise FileExistsError('Never reopen an episode journal for automatic action replay')
        self._stream = self.path.open('x', encoding='utf-8')
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            self._stream.close()
            raise

    def append(self, event):
        line = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False)
        with self._lock:
            self._stream.write(line + '\n')
            self._stream.flush()
            os.fsync(self._stream.fileno())

    @staticmethod
    def _key(d):
        return (d['episode_id'], d['intervention_id'], d['decision_id'])

    def completed_duplicate(self, d):
        with self._lock:
            old = self.decisions.get(self._key(d))
            if old is None:
                return None
            if old['hash'] != payload_hash(d):
                raise ProtocolError('decision_id_collision')
            if old['status'] != 'finished':
                raise ExecutionUncertain('decision_not_replayable')
            return copy.deepcopy(old['result'])

    def accept_and_consume(self, d):
        with self._lock:
            key = self._key(d)
            request = (d['episode_id'], d['intervention_id'], d['request_id'])
            if key in self.decisions:
                raise ProtocolError('decision_already_submitted')
            if request in self.consumed:
                raise ProtocolError('request_already_consumed')
            digest = payload_hash(d)
            self.append(dict(event='decision_accepted', key=list(key),
                             request=list(request), payload_hash=digest))
            self.decisions[key] = dict(hash=digest, status='executing')
            self.consumed.add(request)

    def finish(self, d, result):
        with self._lock:
            key = self._key(d)
            if self.decisions[key]['status'] != 'executing':
                raise ExecutionUncertain('decision_not_executing')
            self.append(dict(event='decision_finished', key=list(key), result=result))
            self.decisions[key].update(status='finished', result=copy.deepcopy(result))

    def close(self):
        with self._lock:
            self._stream.close()
