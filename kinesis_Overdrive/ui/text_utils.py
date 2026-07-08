"""Shared drawing helpers so every screen renders text identically."""

import cv2


def draw_text_shadow(frame, text, pos, scale, color, thickness):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_centered_text(frame, text, y, scale, color, thickness):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (frame.shape[1] - tw) // 2
    draw_text_shadow(frame, text, (x, y), scale, color, thickness)
    return x


def draw_panel(frame, x1, y1, x2, y2, alpha=0.55, color=(0, 0, 0)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
