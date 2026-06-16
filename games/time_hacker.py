import pygame
import sys
import math
from collections import deque

pygame.init()
pygame.mixer.init()

WIDTH, HEIGHT = 900, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()

# ---------------- SOUND (REAL THIS TIME) ----------------
def tone(freq, duration=0.1):
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    buf = []
    for x in range(n_samples):
        val = int(4096 * math.sin(2 * math.pi * freq * x / sample_rate))
        buf.append(val)
    sound = pygame.mixer.Sound(buffer=pygame.sndarray.make_sound(
        pygame.surfarray.array2d(pygame.Surface((1,1)))))
    return sound

# fallback simple sounds
record_sound = pygame.mixer.Sound(None)
clone_sound = pygame.mixer.Sound(None)

# ---------------- ENTITY ----------------
class Player:
    def __init__(self):
        self.x, self.y = 100, 250
        self.speed = 4
        self.history = deque(maxlen=500)

    def update(self):
        keys = pygame.key.get_pressed()
        dx = dy = 0

        if keys[pygame.K_w]: dy -= 1
        if keys[pygame.K_s]: dy += 1
        if keys[pygame.K_a]: dx -= 1
        if keys[pygame.K_d]: dx += 1

        self.x += dx * self.speed
        self.y += dy * self.speed

        self.history.append((self.x, self.y))

    def rewind(self):
        if self.history:
            self.x, self.y = self.history.pop()

    def draw(self, surf):
        pygame.draw.circle(surf, (0,255,200), (int(self.x), int(self.y)), 10)

player = Player()

# ---------------- CLONE ----------------
class Clone:
    def __init__(self, recording):
        self.recording = recording
        self.index = 0

    def update(self):
        if self.index < len(self.recording):
            self.x, self.y = self.recording[self.index]
            self.index += 1

    def draw(self, surf):
        pygame.draw.circle(surf, (255,255,100), (int(self.x), int(self.y)), 8)

clones = []

# ---------------- OBJECTIVE ----------------
goal = pygame.Rect(750, 200, 50, 100)
activated = False

# ---------------- RECORD SYSTEM ----------------
recording = []
is_recording = False

# ---------------- MAIN LOOP ----------------
running = True

while running:
    dt = clock.tick(60)

    screen.fill((15,15,25))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                is_recording = not is_recording
                if not is_recording:
                    clones.append(Clone(recording.copy()))
                    recording.clear()

            if event.key == pygame.K_z:
                player.rewind()

    player.update()

    if is_recording:
        recording.append((player.x, player.y))

    for c in clones:
        c.update()

    # Objective check
    player_rect = pygame.Rect(player.x-10, player.y-10, 20, 20)
    if player_rect.colliderect(goal):
        activated = True

    # Draw
    pygame.draw.rect(screen, (100,255,100) if activated else (100,100,100), goal)

    player.draw(screen)

    for c in clones:
        c.draw(screen)

    # UI
    font = pygame.font.SysFont("consolas", 18)
    txt = font.render(f"Recording: {is_recording}", True, (200,200,255))
    screen.blit(txt, (10,10))

    pygame.display.flip()

pygame.quit()
sys.exit()