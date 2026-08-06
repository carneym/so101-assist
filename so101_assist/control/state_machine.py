"""System state machine.

This encodes the interaction contract with the operator:
nothing moves without an explicit GO, and STOP/CANCEL are honored
from every state. Every transition is published on TOPIC_STATE so
the UI can announce it visually and audibly.

    IDLE ──(voice description / UI click)──► TARGETING
    TARGETING ──(target grounded + path shown)──► CONFIRM
    CONFIRM ──"go"──► APPROACH          (autonomous, speed-capped)
    APPROACH ──(reached pre-grasp)──► FINE     (operator jog + servoing)
    FINE ──"grip"──► GRASP ──► RETRACT ──► IDLE

    any state ──"stop"──► HALTED (motors stopped, requires "go" or "cancel")
    any state ──"cancel"──► IDLE (arm retracts to safe home first)
"""
from __future__ import annotations

from enum import Enum, auto

from ..bus import TOPIC_STATE, Bus
from ..messages import VoiceCommand


class State(Enum):
    IDLE = auto()
    TARGETING = auto()
    CONFIRM = auto()
    APPROACH = auto()
    FINE = auto()
    GRASP = auto()
    RETRACT = auto()
    HALTED = auto()


class StateMachine:
    def __init__(self, bus: Bus) -> None:
        self.bus = bus
        self.state = State.IDLE
        self._prev = State.IDLE

    def _set(self, new: State) -> None:
        self._prev, self.state = self.state, new
        self.bus.publish(TOPIC_STATE, new)

    def on_voice(self, cmd: VoiceCommand) -> None:
        # Safety commands first — valid from ANY state.
        if cmd is VoiceCommand.STOP:
            self._set(State.HALTED)
            return
        if cmd is VoiceCommand.CANCEL:
            self._set(State.RETRACT)
            return

        # TODO: table-driven transitions for the nominal flow.
        raise NotImplementedError

    def on_target_grounded(self) -> None:
        if self.state is State.TARGETING:
            self._set(State.CONFIRM)

    def on_pregrasp_reached(self) -> None:
        if self.state is State.APPROACH:
            self._set(State.FINE)

    def on_retract_done(self) -> None:
        if self.state is State.RETRACT:
            self._set(State.IDLE)
