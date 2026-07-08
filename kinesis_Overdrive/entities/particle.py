"""Small physics-driven burst particle used for hit effects."""

import random
import cv2


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "decay", "color")

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-25, 25)
        self.vy = random.uniform(-25, 25)
        self.life = 1.0
        self.decay = random.uniform(0.04, 0.07)
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 1.2  # gravity
        self.life -= self.decay

    @property
    def alive(self):
        return self.life > 0

    def draw(self, frame):
        if not self.alive:
            return
        alpha = max(0.0, self.life)
        c = (int(self.color[0] * alpha), int(self.color[1] * alpha), int(self.color[2] * alpha))
        radius = max(1, int(6 * alpha))
        cv2.circle(frame, (int(self.x), int(self.y)), radius + 2, (0, 0, 0), -1)
        cv2.circle(frame, (int(self.x), int(self.y)), radius, c, -1)


class ParticleSystem:
    """Owns and updates the full particle pool so callers don't manage lists."""

    def __init__(self):
        self._particles = []

    def burst(self, x, y, color, count=45):
        self._particles.extend(Particle(x, y, color) for _ in range(count))

    def update_and_prune(self):
        for p in self._particles:
            p.update()
        self._particles = [p for p in self._particles if p.alive]

    def draw(self, frame):
        for p in self._particles:
            p.draw(frame)

    def __len__(self):
        return len(self._particles)
