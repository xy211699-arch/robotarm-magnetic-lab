"""Non-queuing session protocol at the atomic action request boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class RequestOutcome(str, Enum):
    ACCEPTED = "ACCEPTED"
    IGNORED_WHILE_BUSY = "IGNORED_WHILE_BUSY"
    MASKED_ACTION = "MASKED_ACTION"
    RESET_WHILE_BUSY = "RESET_WHILE_BUSY"
    RESET_ACCEPTED = "RESET_ACCEPTED"
    EPISODE_TERMINATED = "EPISODE_TERMINATED"


@dataclass(frozen=True)
class SessionRecord:
    request_id: int
    timestamp_s: float
    outcome: RequestOutcome
    action_id: int | None = None
    device_result: str | None = None


class SessionController:
    """Accept one atomic request at a time and discard all busy requests."""

    def __init__(self) -> None:
        self._next_request_id = 1
        self._active: SessionRecord | None = None
        self._terminated = False

    @property
    def busy(self) -> bool:
        return self._active is not None

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def active_request(self) -> SessionRecord | None:
        return self._active

    def request_action(
        self,
        action_id: int,
        action_mask: Sequence[bool],
        timestamp_s: float,
    ) -> SessionRecord:
        request_id = self._next_request_id
        self._next_request_id += 1
        action_id = int(action_id)
        if self._terminated:
            return SessionRecord(request_id, float(timestamp_s), RequestOutcome.EPISODE_TERMINATED, action_id)
        if self.busy:
            return SessionRecord(request_id, float(timestamp_s), RequestOutcome.IGNORED_WHILE_BUSY, action_id)
        if action_id < 0 or action_id >= len(action_mask) or not bool(action_mask[action_id]):
            return SessionRecord(request_id, float(timestamp_s), RequestOutcome.MASKED_ACTION, action_id)
        accepted = SessionRecord(request_id, float(timestamp_s), RequestOutcome.ACCEPTED, action_id)
        self._active = accepted
        return accepted

    def acknowledge(self, device_result: str, timestamp_s: float) -> SessionRecord | None:
        """Acknowledge the active request exactly once at its terminal boundary."""
        if self._active is None:
            return None
        if device_result not in ("DONE", "HARD_FAILURE"):
            raise ValueError("device result must be DONE or HARD_FAILURE")
        completed = SessionRecord(
            request_id=self._active.request_id,
            timestamp_s=float(timestamp_s),
            outcome=RequestOutcome.ACCEPTED,
            action_id=self._active.action_id,
            device_result=device_result,
        )
        self._active = None
        if device_result == "HARD_FAILURE":
            self._terminated = True
        return completed

    def request_reset(self, timestamp_s: float) -> SessionRecord:
        request_id = self._next_request_id
        self._next_request_id += 1
        if self.busy:
            return SessionRecord(request_id, float(timestamp_s), RequestOutcome.RESET_WHILE_BUSY)
        self._terminated = False
        return SessionRecord(request_id, float(timestamp_s), RequestOutcome.RESET_ACCEPTED)
