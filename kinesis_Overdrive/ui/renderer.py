"""
Composites every visual layer for the current frame: dimmed camera feed,
trails, targets, particles, and whichever screen matches the current
GameState. This is the single place that decides draw order.
"""

import cv2
import numpy as np

from config import DIM_BACKGROUND_ALPHA
from engine.state_machine import GameState
from ui.hud import HUD
from ui.screens import MenuScreen, CountdownScreen, PauseScreen, GameOverScreen


class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.hud = HUD()
        self.menu_screen = MenuScreen()
        self.countdown_screen = CountdownScreen()
        self.pause_screen = PauseScreen()
        self.game_over_screen = GameOverScreen()

    def dim(self, raw_frame):
        if raw_frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        black = np.zeros_like(raw_frame)
        return cv2.addWeighted(raw_frame, DIM_BACKGROUND_ALPHA, black,
                                1 - DIM_BACKGROUND_ALPHA, 0)

    def render(self, raw_frame, state, ctx):
        """
        ctx is a dict the app fills in with whatever the current state needs:
        trails, spawner, particles, score_manager, difficulty, timers, etc.
        """
        frame = self.dim(raw_frame)

        # Gameplay layers render underneath any state (visible during
        # countdown/pause too, so the player sees the frozen board).
        for trail in ctx.get("trails", []):
            trail.draw(frame)

        spawner = ctx.get("spawner")
        if spawner:
            spawner.draw(frame)

        particles = ctx.get("particles")
        if particles:
            particles.draw(frame)

        if state == GameState.PLAYING:
            self.hud.draw(frame, ctx["score_manager"], ctx["difficulty"],
                          ctx["time_remaining"], ctx["show_impact"])
        elif state == GameState.MENU:
            self.menu_screen.draw(frame, ctx["palm_progress"], ctx["difficulty"].name)
        elif state == GameState.COUNTDOWN:
            self.hud.draw(frame, ctx["score_manager"], ctx["difficulty"],
                          ctx["time_remaining"], False)
            self.countdown_screen.draw(frame, ctx["countdown_remaining"])
        elif state == GameState.PAUSED:
            self.hud.draw(frame, ctx["score_manager"], ctx["difficulty"],
                          ctx["time_remaining"], False)
            self.pause_screen.draw(frame, ctx["fist_progress"])
        elif state == GameState.GAME_OVER:
            self.game_over_screen.draw(frame, ctx["score_manager"], ctx["difficulty"].name,
                                        ctx["leaderboard"], ctx["palm_progress"])

        return frame
