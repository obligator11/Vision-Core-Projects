"""Scoring, combo/streak multiplier, and persistent JSON leaderboard."""

import json
import os
import time

from config import (
    LEADERBOARD_PATH, LEADERBOARD_SIZE, BASE_SCORE_PER_MPH,
    COMBO_MAX_MULTIPLIER, COMBO_STEP, COMBO_IDLE_RESET_SECONDS,
)


class ScoreManager:
    def __init__(self):
        self.score = 0
        self.combo = 0
        self.combo_multiplier = 1.0
        self.last_hit_time = None
        self.last_hit_score = 0
        self.last_hit_speed = 0.0
        self.last_hit_at = 0.0

    def reset(self):
        self.score = 0
        self.combo = 0
        self.combo_multiplier = 1.0
        self.last_hit_time = None

    def register_hit(self, speed_mph, difficulty_multiplier, now):
        # Combo grows every hit; resets if the player went idle too long.
        if self.last_hit_time is not None and (now - self.last_hit_time) > COMBO_IDLE_RESET_SECONDS:
            self.combo = 0
            self.combo_multiplier = 1.0

        self.combo += 1
        self.combo_multiplier = min(
            COMBO_MAX_MULTIPLIER, 1.0 + (self.combo - 1) * COMBO_STEP
        )

        points = speed_mph * BASE_SCORE_PER_MPH * difficulty_multiplier * self.combo_multiplier
        points = int(points)
        self.score += points

        self.last_hit_time = now
        self.last_hit_score = points
        self.last_hit_speed = speed_mph
        self.last_hit_at = now
        return points

    def register_miss(self, now):
        # A missed/expired target breaks the combo immediately.
        self.combo = 0
        self.combo_multiplier = 1.0

    def maybe_decay_combo(self, now):
        if self.last_hit_time is not None and (now - self.last_hit_time) > COMBO_IDLE_RESET_SECONDS:
            self.combo = 0
            self.combo_multiplier = 1.0

    # -- Leaderboard -------------------------------------------------------
    def load_leaderboard(self):
        if not os.path.exists(LEADERBOARD_PATH):
            return []
        try:
            with open(LEADERBOARD_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def save_score(self, difficulty_name):
        board = self.load_leaderboard()
        board.append({
            "score": self.score,
            "difficulty": difficulty_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        })
        board.sort(key=lambda e: e["score"], reverse=True)
        board = board[:LEADERBOARD_SIZE]
        with open(LEADERBOARD_PATH, "w") as f:
            json.dump(board, f, indent=2)
        return board

    def high_score(self):
        board = self.load_leaderboard()
        return board[0]["score"] if board else 0
