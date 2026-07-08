"""Heads-up display shown while PLAYING."""

import cv2

from config import COLOR_RED, COLOR_CYAN, COLOR_GREEN, COLOR_WHITE
from ui.text_utils import draw_text_shadow, draw_panel


class HUD:
    def draw(self, frame, score_manager, difficulty, time_remaining, show_impact):
        w = frame.shape[1]

        draw_panel(frame, 0, 0, w, 100, alpha=0.45)

        draw_text_shadow(frame, f"SCORE: {score_manager.score}", (24, 45), 1.1, COLOR_WHITE, 2)
        combo_txt = f"COMBO x{score_manager.combo_multiplier:.2f} ({score_manager.combo})"
        combo_color = COLOR_GREEN if score_manager.combo >= 3 else COLOR_WHITE
        draw_text_shadow(frame, combo_txt, (24, 85), 0.75, combo_color, 2)

        timer_txt = f"{int(time_remaining):02d}s"
        (tw, _), _ = cv2.getTextSize(timer_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        timer_color = COLOR_RED if time_remaining <= 10 else COLOR_WHITE
        draw_text_shadow(frame, timer_txt, (w - tw - 30, 55), 1.3, timer_color, 3)

        diff_txt = f"[{difficulty.name}]"
        (dw, _), _ = cv2.getTextSize(diff_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        draw_text_shadow(frame, diff_txt, (w - dw - 30, 85), 0.6, COLOR_CYAN, 2)

        if show_impact:
            draw_text_shadow(frame, f"IMPACT: +{score_manager.last_hit_score}",
                              (30, 150), 1.5, COLOR_RED, 4)
            draw_text_shadow(frame, f"VELOCITY: {int(score_manager.last_hit_speed)} MPH",
                              (30, 195), 1.0, COLOR_CYAN, 2)
