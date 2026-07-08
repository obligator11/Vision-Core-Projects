"""Finite state machine driving the game's flow."""

from enum import Enum, auto


class GameState(Enum):
    MENU = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto()


class StateMachine:
    """Tiny explicit FSM. Keeps transition logic auditable in one place."""

    _ALLOWED_TRANSITIONS = {
        GameState.MENU: {GameState.COUNTDOWN},
        GameState.COUNTDOWN: {GameState.PLAYING, GameState.MENU},
        GameState.PLAYING: {GameState.PAUSED, GameState.GAME_OVER},
        GameState.PAUSED: {GameState.PLAYING, GameState.MENU},
        GameState.GAME_OVER: {GameState.MENU, GameState.COUNTDOWN},
    }

    def __init__(self, initial=GameState.MENU):
        self._state = initial
        self._entered_at = None
        self._listeners = []

    @property
    def state(self):
        return self._state

    def time_in_state(self, now):
        if self._entered_at is None:
            return 0.0
        return now - self._entered_at

    def on_change(self, callback):
        """Register a callback(old_state, new_state) invoked on transition."""
        self._listeners.append(callback)

    def transition(self, new_state, now):
        if new_state == self._state:
            return True
        allowed = self._ALLOWED_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            return False
        old_state = self._state
        self._state = new_state
        self._entered_at = now
        for cb in self._listeners:
            cb(old_state, new_state)
        return True

    def force(self, state, now):
        """Escape hatch for setup code — bypasses the transition table."""
        self._state = state
        self._entered_at = now
