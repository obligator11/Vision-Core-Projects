"""Tracks a single hand's recent positions to compute punch speed and
render its motion trail."""

from collections import deque

import cv2
import numpy as np


class HandTrail:
    def __init__(self, color, frame_height, maxlen=15, real_arm_span_m=1.5):
        self.color = color
        self.frame_height = frame_height
        self.real_arm_span_m = real_arm_span_m
        self._history = deque(maxlen=maxlen)

    def update(self, pos, t):
        self._history.append((pos, t))

    @property
    def latest_pos(self):
        return self._history[-1][0] if self._history else None

    def speed_mph(self):
        if len(self._history) < 3:
            return 0.0
        p1, t1 = self._history[-3]
        p2, t2 = self._history[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0
        dist_pixels = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        dist_meters = dist_pixels * (self.real_arm_span_m / self.frame_height)
        return (dist_meters / dt) * 2.23694

    def draw(self, frame):
        if len(self._history) < 2:
            return
        pts = [h[0] for h in self._history]
        for i in range(1, len(pts)):
            thickness = int(np.interp(i, [1, len(pts)], [2, 18]))
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 0), thickness + 6)
            cv2.line(frame, pts[i - 1], pts[i], self.color, thickness)

        cv2.circle(frame, pts[-1], 20, (0, 0, 0), -1)
        cv2.circle(frame, pts[-1], 15, (255, 255, 255), -1)
        cv2.circle(frame, pts[-1], 18, self.color, 3)
