"""Non-gameplay screens: main menu, countdown, pause overlay, game over."""

import math

import cv2

from config import COLOR_GOLD, COLOR_WHITE, COLOR_RED, COLOR_GREEN, COLOR_CYAN
from ui.text_utils import draw_text_shadow, draw_centered_text, draw_panel


class MenuScreen:
    def draw(self, frame, palm_progress, difficulty_name):
        h, w = frame.shape[:2]
        draw_panel(frame, 0, 0, w, h, alpha=0.45)

        draw_centered_text(frame, "PROJECT KINESIS: NEXT", h // 2 - 160, 1.6, COLOR_GOLD, 4)
        draw_centered_text(frame, "Show an OPEN PALM to the camera to begin",
                            h // 2 - 90, 0.85, COLOR_WHITE, 2)
        draw_centered_text(frame, f"Difficulty: {difficulty_name}  (press 1/2/3 to change)",
                            h // 2 - 50, 0.7, COLOR_CYAN, 2)

        # Palm-hold progress ring, centered
        cx, cy, r = w // 2, h // 2 + 40, 55
        cv2.circle(frame, (cx, cy), r, (0, 0, 0), 8)
        cv2.circle(frame, (cx, cy), r, (90, 90, 90), 4)
        if palm_progress > 0:
            angle = int(360 * palm_progress)
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, angle, COLOR_GREEN, 8)
        draw_centered_text(frame, "HOLD PALM", cy + 8, 0.5, COLOR_WHITE, 1)

        draw_centered_text(frame, "Q to quit  |  P to pause mid-round",
                            h - 40, 0.55, (200, 200, 200), 1)


class CountdownScreen:
    def draw(self, frame, seconds_left_float):
        h, w = frame.shape[:2]
        draw_panel(frame, 0, 0, w, h, alpha=0.3)
        n = max(1, math.ceil(seconds_left_float))
        scale = 4.0 - min(0.8, (math.ceil(seconds_left_float) - seconds_left_float))
        draw_centered_text(frame, str(n), h // 2 + 40, scale, COLOR_GOLD, 6)


class PauseScreen:
    def draw(self, frame, fist_progress):
        h, w = frame.shape[:2]
        draw_panel(frame, 0, 0, w, h, alpha=0.6)
        draw_centered_text(frame, "PAUSED", h // 2 - 60, 1.6, COLOR_WHITE, 4)
        draw_centered_text(frame, "Hold FIST or press P to resume",
                            h // 2, 0.75, COLOR_CYAN, 2)
        cx, cy, r = w // 2, h // 2 + 70, 40
        cv2.circle(frame, (cx, cy), r, (0, 0, 0), 6)
        cv2.circle(frame, (cx, cy), r, (90, 90, 90), 3)
        if fist_progress > 0:
            angle = int(360 * fist_progress)
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, angle, COLOR_GREEN, 6)


class GameOverScreen:
    def draw(self, frame, score_manager, difficulty_name, leaderboard, palm_progress):
        h, w = frame.shape[:2]
        draw_panel(frame, 0, 0, w, h, alpha=0.55)

        draw_centered_text(frame, "ROUND OVER", 90, 1.6, COLOR_RED, 4)
        draw_centered_text(frame, f"Final Score: {score_manager.score}", 145, 1.1, COLOR_WHITE, 3)
        draw_centered_text(frame, f"Difficulty: {difficulty_name}", 185, 0.7, COLOR_CYAN, 2)

        top_y = 230
        draw_text_shadow(frame, "LEADERBOARD", (w // 2 - 300, top_y), 0.8, COLOR_GOLD, 2)
        for i, entry in enumerate(leaderboard[:8]):
            line = f"{i + 1:>2}.  {entry['score']:>6}   {entry['difficulty']:<7} {entry['timestamp']}"
            draw_text_shadow(frame, line, (w // 2 - 300, top_y + 35 + i * 32), 0.6, COLOR_WHITE, 1)

        draw_centered_text(frame, "Show an OPEN PALM to play again", h - 110, 0.8, COLOR_WHITE, 2)
        cx, cy, r = w // 2, h - 55, 30
        cv2.circle(frame, (cx, cy), r, (0, 0, 0), 6)
        cv2.circle(frame, (cx, cy), r, (90, 90, 90), 3)
        if palm_progress > 0:
            angle = int(360 * palm_progress)
            cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, angle, COLOR_GREEN, 6)
