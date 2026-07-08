"""The hittable target the player punches. Difficulty-aware: size, lifetime,
and drift all come from the active Difficulty preset."""

import math
import random
import time

import cv2
import numpy as np

from config import COLOR_GOLD


class Target:
    def __init__(self, w, h, difficulty):
        if random.choice([True, False]):
            self.x = random.randint(int(w * 0.05), int(w * 0.35))
        else:
            self.x = random.randint(int(w * 0.65), int(w * 0.95))
        self.y = random.randint(int(h * 0.1), int(h * 0.5))

        self.w, self.h = w, h
        self.radius = difficulty.target_radius
        self.lifetime = difficulty.target_lifetime
        self.drift = difficulty.drift
        self.drift_speed = difficulty.drift_speed

        self.color = COLOR_GOLD
        self.active = True
        self.spawn_time = time.time()

        self._drift_angle = random.uniform(0, 2 * math.pi)

    @property
    def age(self):
        return time.time() - self.spawn_time

    @property
    def expired(self):
        return self.age >= self.lifetime

    @property
    def time_remaining_frac(self):
        return max(0.0, 1.0 - (self.age / self.lifetime))

    def update(self):
        if not self.drift:
            return
        self.x += math.cos(self._drift_angle) * self.drift_speed
        self.y += math.sin(self._drift_angle) * self.drift_speed
        # gently bounce off the frame edges instead of drifting off-screen
        margin = self.radius + 10
        if self.x < margin or self.x > self.w - margin:
            self._drift_angle = math.pi - self._drift_angle
        if self.y < margin or self.y > self.h - margin:
            self._drift_angle = -self._drift_angle
        self.x = int(max(margin, min(self.w - margin, self.x)))
        self.y = int(max(margin, min(self.h - margin, self.y)))

    def draw(self, frame):
        if not self.active:
            return

        elapsed = self.age
        pulse = int(8 * math.sin(elapsed * 8))
        angle = int(elapsed * 120) % 360
        axes = (self.radius + pulse, self.radius + pulse)

        cv2.ellipse(frame, (int(self.x), int(self.y)), axes, angle, 0, 100, (0, 0, 0), 7)
        cv2.ellipse(frame, (int(self.x), int(self.y)), axes, angle, 180, 280, (0, 0, 0), 7)
        cv2.ellipse(frame, (int(self.x), int(self.y)), axes, angle, 0, 100, self.color, 3)
        cv2.ellipse(frame, (int(self.x), int(self.y)), axes, angle, 180, 280, self.color, 3)

        pts = []
        for i in range(6):
            theta = math.radians(60 * i + (angle * -0.6))
            px = int(self.x + (self.radius - 12) * math.cos(theta))
            py = int(self.y + (self.radius - 12) * math.sin(theta))
            pts.append([px, py])
        pts_array = np.array(pts, np.int32)
        cv2.polylines(frame, [pts_array], isClosed=True, color=(0, 0, 0), thickness=6)
        cv2.polylines(frame, [pts_array], isClosed=True, color=(255, 255, 255), thickness=2)

        cv2.circle(frame, (int(self.x), int(self.y)), 6, (0, 0, 0), -1)
        cv2.circle(frame, (int(self.x), int(self.y)), 4, (0, 0, 255), -1)

        # Depleting lifetime ring — tells the player a miss is coming.
        life_angle = int(360 * self.time_remaining_frac)
        cv2.ellipse(frame, (int(self.x), int(self.y)), (self.radius + 16, self.radius + 16),
                    -90, 0, life_angle, (255, 255, 255), 2)

    def contains(self, point, hit_margin=30):
        dist = math.hypot(point[0] - self.x, point[1] - self.y)
        return dist < self.radius + hit_margin


class TargetSpawner:
    """Owns the pool of live targets and applies difficulty spawn rules."""

    def __init__(self, w, h, difficulty):
        self.w, self.h = w, h
        self.difficulty = difficulty
        self.targets = []
        self._last_spawn = 0.0

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty

    def update(self, now):
        for t in self.targets:
            t.update()

        expired_misses = [t for t in self.targets if t.active and t.expired]
        for t in expired_misses:
            t.active = False
        self.targets = [t for t in self.targets if t.active]

        if (len(self.targets) < self.difficulty.max_active_targets
                and now - self._last_spawn >= self.difficulty.spawn_interval):
            self.targets.append(Target(self.w, self.h, self.difficulty))
            self._last_spawn = now

        return len(expired_misses)  # so the caller can register combo breaks

    def draw(self, frame):
        for t in self.targets:
            t.draw(frame)

    def check_hit(self, point):
        """Deactivates and returns the first target hit, or None."""
        for t in self.targets:
            if t.active and t.contains(point):
                t.active = False
                return t
        return None

    def clear(self):
        self.targets = []
        self._last_spawn = 0.0
