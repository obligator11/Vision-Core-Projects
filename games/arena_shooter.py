import pygame
import sys
import math
import random
import array
import json
import os
from collections import deque
from enum import Enum

# ============================================================================
# ADJUSTABLE SCREEN SETTINGS
# ============================================================================
SCREEN_PRESETS = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "1440p": (2560, 1440),
    "custom": (1400, 900),  # Default
}

# Change this to adjust screen size
CURRENT_PRESET = "custom"  # Options: "1080p", "720p", "1440p", "custom"
CUSTOM_WIDTH = 1400  # Adjust if using "custom"
CUSTOM_HEIGHT = 900

# Auto-set based on preset
if CURRENT_PRESET in SCREEN_PRESETS:
    WIDTH, HEIGHT = SCREEN_PRESETS[CURRENT_PRESET]
else:
    WIDTH, HEIGHT = CUSTOM_WIDTH, CUSTOM_HEIGHT

FPS = 60

# ============================================================================
# COLORS
# ============================================================================
COLOR_BG = (5, 5, 15)
COLOR_BG_ACCENT = (15, 20, 35)
COLOR_PRIMARY = (0, 255, 220)
COLOR_SECONDARY = (255, 0, 150)
COLOR_ACCENT = (0, 200, 255)
COLOR_DANGER = (255, 0, 100)
COLOR_SUCCESS = (100, 255, 100)
COLOR_WARNING = (255, 150, 0)
COLOR_TEXT = (220, 230, 255)
COLOR_DARK = (10, 10, 20)

# ============================================================================
# WEAPONS
# ============================================================================
class WeaponType(Enum):
    PISTOL = {"damage": 8, "cooldown": 0.12, "speed": 600, "spread": 0, "name": "PISTOL"}
    RIFLE = {"damage": 12, "cooldown": 0.2, "speed": 700, "spread": 0.1, "name": "RIFLE"}
    SHOTGUN = {"damage": 25, "cooldown": 0.5, "speed": 450, "spread": 0.4, "name": "SHOTGUN"}
    PLASMA = {"damage": 15, "cooldown": 0.3, "speed": 550, "spread": 0.05, "name": "PLASMA"}
    LASER = {"damage": 20, "cooldown": 0.15, "speed": 900, "spread": 0, "name": "LASER"}

# ============================================================================
# ENEMIES WITH UNIQUE VISUALS
# ============================================================================
class EnemyType(Enum):
    # Tier 1
    DRONE = {"health": 60, "speed": 220, "damage": 7, "cooldown": 0.8, "color": (100, 200, 255), "score": 100, "size": 18, "shape": "circle"}
    ROGUE = {"health": 50, "speed": 280, "damage": 5, "cooldown": 1.0, "color": (150, 255, 100), "score": 120, "size": 16, "shape": "diamond"}
    SWARM = {"health": 20, "speed": 320, "damage": 3, "cooldown": 0.5, "color": (255, 200, 0), "score": 50, "size": 10, "shape": "triangle"}
    
    # Tier 2
    TANK = {"health": 200, "speed": 140, "damage": 12, "cooldown": 1.2, "color": (255, 100, 50), "score": 250, "size": 28, "shape": "square"}
    SNIPER = {"health": 80, "speed": 180, "damage": 20, "cooldown": 1.5, "color": (255, 150, 0), "score": 300, "size": 19, "shape": "arrow"}
    CHARGER = {"health": 120, "speed": 250, "damage": 15, "cooldown": 0.6, "color": (200, 0, 255), "score": 200, "size": 22, "shape": "hexagon"}
    SHIELDER = {"health": 150, "speed": 160, "damage": 8, "cooldown": 0.9, "color": (0, 255, 150), "score": 280, "size": 24, "shape": "shield"}
    HEALER = {"health": 90, "speed": 200, "damage": 4, "cooldown": 2.0, "color": (100, 255, 200), "score": 350, "size": 21, "shape": "star"}
    
    # Tier 3
    ASSASSIN = {"health": 70, "speed": 350, "damage": 18, "cooldown": 1.2, "color": (150, 50, 255), "score": 400, "size": 17, "shape": "spike"}
    EXPLOSIVE = {"health": 80, "speed": 200, "damage": 30, "cooldown": 2.0, "color": (255, 100, 0), "score": 500, "size": 24, "shape": "bomb"}
    FREEZER = {"health": 100, "speed": 190, "damage": 10, "cooldown": 1.5, "color": (100, 200, 255), "score": 350, "size": 21, "shape": "snowflake"}
    
    # Boss Variants
    BOSS_TANK = {"health": 600, "speed": 120, "damage": 28, "cooldown": 0.4, "color": (255, 50, 50), "score": 3000, "size": 36, "shape": "fortress"}
    BOSS_NOVA = {"health": 500, "speed": 180, "damage": 20, "cooldown": 0.3, "color": (255, 150, 0), "score": 3500, "size": 34, "shape": "nova"}

# ============================================================================
# POWER-UPS
# ============================================================================
class PowerUpType(Enum):
    HEALTH = {"color": (100, 255, 100), "symbol": "H", "effect": "Restore 30 HP"}
    SHIELD = {"color": (0, 200, 255), "symbol": "S", "effect": "Full Shield"}
    DAMAGE = {"color": (255, 0, 100), "symbol": "D", "effect": "+50% Damage for 8s"}
    SPEED = {"color": (255, 200, 0), "symbol": "V", "effect": "+30% Speed for 8s"}
    AMMO = {"color": (0, 255, 150), "symbol": "A", "effect": "Reset Cooldowns"}
    SLOW = {"color": (100, 150, 255), "symbol": "T", "effect": "Slow Enemies 5s"}

class PowerUp:
    def __init__(self, x, y, power_type):
        self.x = x
        self.y = y
        self.power_type = power_type
        self.radius = 12
        self.rotation = 0
        self.should_remove = False
        self.lifetime = 10.0

    def update(self, dt):
        self.rotation += dt * 5
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.should_remove = True

    def draw(self, surface, camera_offset):
        pos = (int(self.x + camera_offset[0]), int(self.y + camera_offset[1]))
        color = self.power_type.value["color"]
        
        glow = pygame.Surface((self.radius * 3, self.radius * 3), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*color, 80), (self.radius * 1.5, self.radius * 1.5), self.radius * 1.5)
        surface.blit(glow, (pos[0] - self.radius * 1.5, pos[1] - self.radius * 1.5))
        
        pygame.draw.circle(surface, color, pos, self.radius)
        pygame.draw.circle(surface, (255, 255, 255), pos, self.radius, 2)

# ============================================================================
# SOUND MANAGER
# ============================================================================
class SoundManager:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
        self.sounds = {}
        self._generate_procedural_sounds()
        self.set_volume(0.15)

    def _generate_procedural_sounds(self):
        def create_sound(freq_func, duration_secs):
            sample_rate = 22050
            num_samples = int(sample_rate * duration_secs)
            buf = array.array('h', [0] * num_samples)
            for i in range(num_samples):
                t = i / sample_rate
                freq = freq_func(t)
                volume_envelope = math.exp(-5 * t)
                value = math.sin(2 * math.pi * freq * t) * volume_envelope
                buf[i] = int(value * 16383)
            return pygame.mixer.Sound(buffer=buf)

        self.sounds['shoot'] = create_sound(lambda t: 1000 - 800 * t, 0.1)
        self.sounds['shoot_laser'] = create_sound(lambda t: 1500 - 1000 * t, 0.08)
        self.sounds['hit'] = create_sound(lambda t: 150 - 100 * t, 0.08)
        self.sounds['powerup'] = create_sound(lambda t: 800 + 400 * math.sin(20 * t), 0.25)
        self.sounds['levelup'] = create_sound(lambda t: 500 + 300 * math.sin(15 * t), 0.35)
        self.sounds['boss'] = create_sound(lambda t: 200 - 150 * t, 0.5)

    def play(self, name, loops=0):
        if name in self.sounds:
            self.sounds[name].play(loops=loops)

    def set_volume(self, volume):
        for sound in self.sounds.values():
            sound.set_volume(volume)

# ============================================================================
# PARTICLES & EFFECTS
# ============================================================================
class Particle:
    def __init__(self, x, y, vx, vy, color, lifetime, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.rotation = 0
        self.rot_speed = random.uniform(-5, 5)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt
        self.rotation += self.rot_speed

    def draw(self, surface, camera_offset):
        if self.lifetime <= 0:
            return
        alpha = int((self.lifetime / self.max_lifetime) * 255)
        size = max(1, int(self.size * (self.lifetime / self.max_lifetime)))
        p_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
        pygame.draw.circle(p_surf, (*self.color, alpha), (size, size), size)
        surface.blit(p_surf, (self.x - size + camera_offset[0], self.y - size + camera_offset[1]))

class DamageNumber:
    def __init__(self, x, y, damage, color):
        self.x = x
        self.y = y
        self.damage = damage
        self.color = color
        self.lifetime = 1.0
        self.max_lifetime = 1.0

    def update(self, dt):
        self.lifetime -= dt
        self.y -= 50 * dt

    def draw(self, surface, camera_offset, font):
        if self.lifetime <= 0:
            return
        alpha = int((self.lifetime / self.max_lifetime) * 255)
        text = font.render(str(int(self.damage)), True, self.color)
        text.set_alpha(alpha)
        pos = (int(self.x + camera_offset[0]), int(self.y + camera_offset[1]))
        surface.blit(text, pos)

class ParticleSystem:
    def __init__(self):
        self.particles = []
        self.damage_numbers = []

    def emit(self, x, y, color, count=10, speed_range=(100, 300), lifetime_range=(0.2, 0.5)):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(*speed_range)
            lifetime = random.uniform(*lifetime_range)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, random.uniform(2, 5)))

    def emit_directed(self, x, y, color, angle, count=8, speed_range=(150, 250)):
        for _ in range(count):
            angle_offset = angle + random.uniform(-0.3, 0.3)
            speed = random.uniform(*speed_range)
            lifetime = random.uniform(0.15, 0.35)
            vx = math.cos(angle_offset) * speed
            vy = math.sin(angle_offset) * speed
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, random.uniform(2, 4)))

    def add_damage_number(self, x, y, damage, color):
        self.damage_numbers.append(DamageNumber(x, y, damage, color))

    def update(self, dt):
        for p in self.particles[:]:
            p.update(dt)
            if p.lifetime <= 0:
                self.particles.remove(p)
        
        for dn in self.damage_numbers[:]:
            dn.update(dt)
            if dn.lifetime <= 0:
                self.damage_numbers.remove(dn)

    def draw(self, surface, camera_offset, font):
        for p in self.particles:
            p.draw(surface, camera_offset)
        for dn in self.damage_numbers:
            dn.draw(surface, camera_offset, font)

# ============================================================================
# BULLETS
# ============================================================================
class Bullet:
    def __init__(self, x, y, angle, speed, damage, color, is_enemy=False, bullet_type="normal"):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.radius = 5 if bullet_type != "laser" else 3
        self.damage = damage
        self.color = color
        self.is_enemy = is_enemy
        self.should_remove = False
        self.lifetime = 5.0
        self.bullet_type = bullet_type

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.lifetime -= dt
        if self.lifetime <= 0 or self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT:
            self.should_remove = True

    def draw(self, surface, camera_offset):
        pos = (int(self.x + camera_offset[0]), int(self.y + camera_offset[1]))
        
        if self.bullet_type == "laser":
            pygame.draw.line(surface, self.color, (pos[0] - 10, pos[1]), (pos[0] + 10, pos[1]), 3)
        else:
            glow_size = self.radius + 3
            glow = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*self.color, 60), (glow_size, glow_size), glow_size)
            surface.blit(glow, (pos[0] - glow_size, pos[1] - glow_size))
            pygame.draw.circle(surface, self.color, pos, self.radius)

# ============================================================================
# OBSTACLES
# ============================================================================
class Obstacle:
    def __init__(self, x, y, width, height, obstacle_type="static"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.obstacle_type = obstacle_type
        self.color = (100, 100, 150)
        
        if obstacle_type == "moving":
            self.vx = random.choice([-50, 50])
            self.vy = 0
        else:
            self.vx = 0
            self.vy = 0

    def update(self, dt):
        if self.obstacle_type == "moving":
            self.x += self.vx * dt
            if self.x <= 50 or self.x + self.width >= WIDTH - 50:
                self.vx *= -1

    def draw(self, surface, camera_offset):
        rect = pygame.Rect(
            int(self.x + camera_offset[0]),
            int(self.y + camera_offset[1]),
            int(self.width),
            int(self.height)
        )
        pygame.draw.rect(surface, self.color, rect)
        pygame.draw.rect(surface, (150, 150, 200), rect, 2)

    def collides_with_point(self, x, y, radius):
        closest_x = max(self.x, min(x, self.x + self.width))
        closest_y = max(self.y, min(y, self.y + self.height))
        distance = math.hypot(x - closest_x, y - closest_y)
        return distance < radius

# ============================================================================
# UNIQUE ENEMY DRAWING FUNCTIONS
# ============================================================================
def draw_enemy(surface, enemy, pos):
    """Draw enemy with unique visual design based on type"""
    
    if enemy.enemy_type == EnemyType.DRONE:
        # Circle with center dot
        pygame.draw.circle(surface, enemy.color, pos, enemy.radius)
        pygame.draw.circle(surface, (255, 255, 255), pos, enemy.radius // 2)
        
    elif enemy.enemy_type == EnemyType.ROGUE:
        # Diamond shape
        points = [
            (pos[0], pos[1] - enemy.radius),
            (pos[0] + enemy.radius, pos[1]),
            (pos[0], pos[1] + enemy.radius),
            (pos[0] - enemy.radius, pos[1])
        ]
        pygame.draw.polygon(surface, enemy.color, points)
        pygame.draw.polygon(surface, (255, 255, 255), points, 2)
        
    elif enemy.enemy_type == EnemyType.SWARM:
        # Small triangle
        points = [
            (pos[0], pos[1] - enemy.radius),
            (pos[0] + enemy.radius, pos[1] + enemy.radius),
            (pos[0] - enemy.radius, pos[1] + enemy.radius)
        ]
        pygame.draw.polygon(surface, enemy.color, points)
        
    elif enemy.enemy_type == EnemyType.TANK:
        # Square with pattern
        pygame.draw.rect(surface, enemy.color, (pos[0] - enemy.radius, pos[1] - enemy.radius, enemy.radius * 2, enemy.radius * 2))
        pygame.draw.rect(surface, (255, 255, 255), (pos[0] - enemy.radius, pos[1] - enemy.radius, enemy.radius * 2, enemy.radius * 2), 3)
        # Cross pattern
        pygame.draw.line(surface, (255, 255, 255), (pos[0] - enemy.radius, pos[1]), (pos[0] + enemy.radius, pos[1]), 1)
        pygame.draw.line(surface, (255, 255, 255), (pos[0], pos[1] - enemy.radius), (pos[0], pos[1] + enemy.radius), 1)
        
    elif enemy.enemy_type == EnemyType.SNIPER:
        # Arrow pointing in direction
        angle = enemy.direction
        points = [
            (pos[0] + math.cos(angle) * enemy.radius * 1.2, pos[1] + math.sin(angle) * enemy.radius * 1.2),
            (pos[0] + math.cos(angle + 2.5) * enemy.radius, pos[1] + math.sin(angle + 2.5) * enemy.radius),
            (pos[0] + math.cos(angle - 2.5) * enemy.radius, pos[1] + math.sin(angle - 2.5) * enemy.radius)
        ]
        pygame.draw.polygon(surface, enemy.color, points)
        pygame.draw.polygon(surface, (255, 255, 255), points, 2)
        
    elif enemy.enemy_type == EnemyType.CHARGER:
        # Hexagon
        angle_step = math.pi / 3
        points = [(pos[0] + math.cos(i * angle_step) * enemy.radius, pos[1] + math.sin(i * angle_step) * enemy.radius) for i in range(6)]
        pygame.draw.polygon(surface, enemy.color, points)
        pygame.draw.polygon(surface, (255, 255, 255), points, 2)
        
    elif enemy.enemy_type == EnemyType.SHIELDER:
        # Shield shape with protective ring
        pygame.draw.circle(surface, enemy.color, pos, enemy.radius)
        pygame.draw.circle(surface, (0, 255, 200), pos, enemy.radius + 5, 3)
        pygame.draw.circle(surface, (0, 255, 200), pos, enemy.radius - 5, 2)
        
    elif enemy.enemy_type == EnemyType.HEALER:
        # Star shape with healing aura
        star_points = []
        for i in range(10):
            angle = i * math.pi / 5
            radius = enemy.radius if i % 2 == 0 else enemy.radius / 2
            star_points.append((pos[0] + math.cos(angle) * radius, pos[1] + math.sin(angle) * radius))
        pygame.draw.polygon(surface, enemy.color, star_points)
        pygame.draw.circle(surface, (100, 255, 200), pos, enemy.radius + 8, 1)
        
    elif enemy.enemy_type == EnemyType.ASSASSIN:
        # Spike shape with pointed edges
        points = [
            (pos[0], pos[1] - enemy.radius * 1.2),
            (pos[0] + enemy.radius * 0.8, pos[1] + enemy.radius * 0.6),
            (pos[0] - enemy.radius * 0.8, pos[1] + enemy.radius * 0.6)
        ]
        pygame.draw.polygon(surface, enemy.color, points)
        pygame.draw.polygon(surface, (255, 100, 200), points, 2)
        # Extra spikes
        for offset in [-1, 1]:
            pygame.draw.line(surface, (150, 50, 255), (pos[0] + offset * enemy.radius, pos[1] - enemy.radius // 2), 
                           (pos[0] + offset * enemy.radius * 1.3, pos[1] - enemy.radius * 1.2), 2)
        
    elif enemy.enemy_type == EnemyType.EXPLOSIVE:
        # Bomb shape
        pygame.draw.circle(surface, enemy.color, pos, enemy.radius)
        pygame.draw.circle(surface, (255, 150, 0), pos, enemy.radius, 3)
        # Fuse
        pygame.draw.line(surface, (100, 100, 100), pos, (pos[0], pos[1] - enemy.radius * 1.3), 2)
        pygame.draw.circle(surface, (255, 200, 0), (pos[0], pos[1] - enemy.radius * 1.3), 2)
        
    elif enemy.enemy_type == EnemyType.FREEZER:
        # Snowflake
        pygame.draw.circle(surface, enemy.color, pos, enemy.radius)
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            end_x = pos[0] + math.cos(rad) * enemy.radius * 1.3
            end_y = pos[1] + math.sin(rad) * enemy.radius * 1.3
            pygame.draw.line(surface, (150, 200, 255), pos, (end_x, end_y), 2)
        
    elif enemy.enemy_type in [EnemyType.BOSS_TANK, EnemyType.BOSS_NOVA]:
        # Boss fortress shape
        pygame.draw.rect(surface, enemy.color, (pos[0] - enemy.radius, pos[1] - enemy.radius, enemy.radius * 2, enemy.radius * 2))
        pygame.draw.rect(surface, (255, 0, 0), (pos[0] - enemy.radius, pos[1] - enemy.radius, enemy.radius * 2, enemy.radius * 2), 4)
        # Castle turrets
        for offset in [-enemy.radius // 2, enemy.radius // 2]:
            pygame.draw.rect(surface, enemy.color, (pos[0] + offset - 3, pos[1] - enemy.radius - 5, 6, 5))
        # Boss eye
        pygame.draw.circle(surface, (255, 255, 255), pos, enemy.radius // 3)
        pygame.draw.circle(surface, (255, 0, 0), pos, enemy.radius // 5)
    
    else:
        # Fallback to circle
        pygame.draw.circle(surface, enemy.color, pos, enemy.radius)

# ============================================================================
# PLAYER
# ============================================================================
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 18
        self.speed = 350
        self.health = 100
        self.max_health = 100
        self.shield = 0
        self.max_shield = 50
        
        self.current_weapon = WeaponType.PISTOL
        self.shoot_timer = 0
        
        self.dash_cooldown = 0
        self.dash_active = False
        self.dash_duration = 0.3
        self.dash_direction = (0, 0)
        
        self.shield_cooldown = 0
        self.shield_active = False
        self.shield_timer = 0
        
        self.score = 0
        self.combo = 0
        self.combo_timer = 0
        
        self.damage_multiplier = 1.0
        self.damage_boost_timer = 0
        self.speed_multiplier = 1.0
        self.speed_boost_timer = 0
        self.slow_enemies = False
        self.slow_timer = 0
        
        self.kills = 0

    def get_movement_vector(self):
        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_w] or keys[pygame.K_UP]:    dy -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  dy += 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  dx -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: dx += 1
        
        if dx != 0 and dy != 0:
            factor = 1 / math.sqrt(2)
            dx *= factor
            dy *= factor
        return dx, dy

    def change_weapon(self, weapon_type):
        self.current_weapon = weapon_type
        self.shoot_timer = 0

    def activate_dash(self, direction):
        if self.dash_cooldown <= 0:
            self.dash_active = True
            self.dash_direction = direction
            self.dash_cooldown = 1.5

    def activate_shield(self):
        if self.shield_cooldown <= 0 and not self.shield_active:
            self.shield_active = True
            self.shield_timer = 0.8
            self.shield = self.max_shield
            self.shield_cooldown = 3.0

    def apply_powerup(self, powerup_type):
        if powerup_type == PowerUpType.HEALTH:
            self.health = min(self.max_health, self.health + 30)
        elif powerup_type == PowerUpType.SHIELD:
            self.shield = self.max_shield
            self.shield_active = True
            self.shield_timer = 0.8
        elif powerup_type == PowerUpType.DAMAGE:
            self.damage_multiplier = 1.5
            self.damage_boost_timer = 8.0
        elif powerup_type == PowerUpType.SPEED:
            self.speed_multiplier = 1.3
            self.speed_boost_timer = 8.0
        elif powerup_type == PowerUpType.AMMO:
            self.shoot_timer = 0
            self.dash_cooldown = 0
        elif powerup_type == PowerUpType.SLOW:
            self.slow_enemies = True
            self.slow_timer = 5.0

    def update(self, dt, bullets, sound_manager, particle_system, obstacles):
        if self.shoot_timer > 0:
            self.shoot_timer -= dt
        if self.dash_cooldown > 0:
            self.dash_cooldown -= dt
        if self.shield_cooldown > 0:
            self.shield_cooldown -= dt
        if self.combo_timer > 0:
            self.combo_timer -= dt
        else:
            self.combo = 0
        
        if self.damage_boost_timer > 0:
            self.damage_boost_timer -= dt
        else:
            self.damage_multiplier = 1.0
        
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= dt
        else:
            self.speed_multiplier = 1.0
        
        if self.slow_timer > 0:
            self.slow_timer -= dt
        else:
            self.slow_enemies = False

        dx, dy = self.get_movement_vector()
        
        if self.dash_active:
            dash_speed = 800
            self.x += self.dash_direction[0] * dash_speed * dt
            self.y += self.dash_direction[1] * dash_speed * dt
            self.dash_duration -= dt
            if self.dash_duration <= 0:
                self.dash_active = False
                self.dash_duration = 0.3
        else:
            self.x += dx * self.speed * self.speed_multiplier * dt
            self.y += dy * self.speed * self.speed_multiplier * dt

        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

        for obs in obstacles:
            if obs.collides_with_point(self.x, self.y, self.radius):
                self.x = max(obs.x + obs.width + self.radius, min(obs.x - self.radius, self.x))
                self.y = max(obs.y + obs.height + self.radius, min(obs.y - self.radius, self.y))

        if self.shield_active:
            self.shield_timer -= dt
            if self.shield_timer <= 0:
                self.shield_active = False
                self.shield = 0

        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()
        if mouse_pressed[0] and self.shoot_timer <= 0:
            angle = math.atan2(mouse_pos[1] - self.y, mouse_pos[0] - self.x)
            weapon = self.current_weapon.value
            damage = int(weapon['damage'] * self.damage_multiplier)
            
            if self.current_weapon == WeaponType.SHOTGUN:
                for i in range(5):
                    spread_angle = angle + (i - 2) * weapon['spread']
                    bullets.append(Bullet(self.x, self.y, spread_angle, weapon['speed'], damage, COLOR_PRIMARY, is_enemy=False))
                particle_system.emit_directed(self.x + math.cos(angle)*20, self.y + math.sin(angle)*20, COLOR_PRIMARY, angle, count=15)
            elif self.current_weapon == WeaponType.LASER:
                bullets.append(Bullet(self.x, self.y, angle, weapon['speed'], damage, COLOR_SECONDARY, is_enemy=False, bullet_type="laser"))
                particle_system.emit_directed(self.x + math.cos(angle)*20, self.y + math.sin(angle)*20, COLOR_SECONDARY, angle, count=8)
            else:
                bullets.append(Bullet(self.x, self.y, angle, weapon['speed'], damage, COLOR_PRIMARY, is_enemy=False))
                particle_system.emit_directed(self.x + math.cos(angle)*20, self.y + math.sin(angle)*20, COLOR_PRIMARY, angle, count=5)
            
            sound_manager.play('shoot')
            self.shoot_timer = weapon['cooldown']

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            if dx != 0 or dy != 0:
                norm = math.sqrt(dx*dx + dy*dy)
                self.activate_dash((dx/norm, dy/norm))
        if keys[pygame.K_LSHIFT]:
            self.activate_shield()

    def draw(self, surface, camera_offset):
        pos = (int(self.x + camera_offset[0]), int(self.y + camera_offset[1]))
        
        glow = pygame.Surface((self.radius * 3, self.radius * 3), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*COLOR_PRIMARY, 40), (self.radius * 1.5, self.radius * 1.5), self.radius * 1.5)
        surface.blit(glow, (pos[0] - self.radius * 1.5, pos[1] - self.radius * 1.5))
        
        pygame.draw.circle(surface, COLOR_PRIMARY, pos, self.radius)
        
        if self.shield_active:
            shield_radius = self.radius + 10
            pygame.draw.circle(surface, COLOR_ACCENT, pos, shield_radius, 2)
            pygame.draw.circle(surface, (*COLOR_ACCENT, 30), pos, shield_radius - 3)
        
        if self.damage_boost_timer > 0:
            pygame.draw.circle(surface, COLOR_DANGER, pos, self.radius + 3, 2)
        if self.speed_boost_timer > 0:
            pygame.draw.circle(surface, COLOR_WARNING, pos, self.radius + 5, 2)
        
        mouse_pos = pygame.mouse.get_pos()
        angle = math.atan2(mouse_pos[1] - self.y, mouse_pos[0] - self.x)
        gun_end = (pos[0] + math.cos(angle) * 30, pos[1] + math.sin(angle) * 30)
        pygame.draw.line(surface, COLOR_SECONDARY, pos, gun_end, 4)

    def take_damage(self, damage):
        if self.shield_active:
            actual_damage = max(0, damage - self.shield)
            self.shield = max(0, self.shield - damage)
        else:
            actual_damage = damage
        self.health -= actual_damage
        return actual_damage

    def add_score(self, points):
        multiplier = 1.0 + (self.combo * 0.1)
        points = int(points * multiplier)
        self.score += points
        self.combo += 1
        self.combo_timer = 3.0
        self.kills += 1
        return points

# ============================================================================
# ENEMY
# ============================================================================
class Enemy:
    def __init__(self, x, y, enemy_type):
        self.x = x
        self.y = y
        self.enemy_type = enemy_type
        
        stats = enemy_type.value
        self.health = stats['health']
        self.max_health = stats['health']
        self.speed = stats['speed']
        self.damage = stats['damage']
        self.shoot_cooldown = stats['cooldown']
        self.color = stats['color']
        self.score_value = stats['score']
        self.radius = stats['size'] / 2
        self.shape = stats['shape']
        
        self.shoot_timer = 0
        self.direction = 0
        self.phase = 0
        self.phase_timer = random.uniform(1, 3)
        self.slow_multiplier = 1.0

    def update(self, dt, player, bullets, sound_manager, particle_system, obstacles, slow_active):
        if slow_active:
            self.slow_multiplier = 0.5
        else:
            self.slow_multiplier = 1.0

        if self.shoot_timer > 0:
            self.shoot_timer -= dt
        
        self.phase_timer -= dt
        if self.phase_timer <= 0:
            self.phase = (self.phase + 1) % 2
            self.phase_timer = random.uniform(1.5, 3)

        dx = player.x - self.x
        dy = player.y - self.y
        dist = math.hypot(dx, dy)

        if self.enemy_type == EnemyType.TANK:
            if dist > 150:
                self.direction = math.atan2(dy, dx)
            else:
                self.direction = math.atan2(-dy, -dx)
        elif self.enemy_type == EnemyType.SNIPER:
            if dist < 400:
                self.direction = math.atan2(-dy, -dx)
            else:
                self.direction = math.atan2(dy, dx)
        elif self.enemy_type == EnemyType.CHARGER:
            self.direction = math.atan2(dy, dx) + math.sin(self.phase_timer) * 0.5
        elif self.enemy_type == EnemyType.ASSASSIN:
            self.direction = math.atan2(dy, dx)
        elif self.enemy_type == EnemyType.SWARM:
            self.direction = math.atan2(dy, dx) + math.sin(self.phase_timer * 3) * 0.8
        else:
            self.direction = math.atan2(dy, dx)

        if dist > 50:
            move_speed = self.speed * self.slow_multiplier
            self.x += math.cos(self.direction) * move_speed * dt
            self.y += math.sin(self.direction) * move_speed * dt

        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

        for obs in obstacles:
            if obs.collides_with_point(self.x, self.y, self.radius):
                self.x = max(obs.x + obs.width + self.radius, min(obs.x - self.radius, self.x))
                self.y = max(obs.y + obs.height + self.radius, min(obs.y - self.radius, self.y))

        if self.shoot_timer <= 0 and dist > 40:
            angle = math.atan2(dy, dx)
            
            if self.enemy_type in [EnemyType.BOSS_TANK, EnemyType.BOSS_NOVA]:
                for i in range(4):
                    spread_angle = angle + (i - 1.5) * 0.4
                    bullets.append(Bullet(self.x, self.y, spread_angle, 400, self.damage, COLOR_DANGER, is_enemy=True))
                    particle_system.emit_directed(self.x + math.cos(spread_angle)*22, self.y + math.sin(spread_angle)*22, COLOR_DANGER, spread_angle, count=4)
                sound_manager.play('boss')
            elif self.enemy_type == EnemyType.TANK:
                bullets.append(Bullet(self.x, self.y, angle, 380, self.damage, COLOR_DANGER, is_enemy=True))
                particle_system.emit_directed(self.x + math.cos(angle)*22, self.y + math.sin(angle)*22, COLOR_DANGER, angle, count=5)
            elif self.enemy_type == EnemyType.SWARM:
                for i in range(2):
                    spread_angle = angle + (i - 0.5) * 0.3
                    bullets.append(Bullet(self.x, self.y, spread_angle, 450, self.damage, COLOR_DANGER, is_enemy=True))
            else:
                bullets.append(Bullet(self.x, self.y, angle, 450, self.damage, COLOR_DANGER, is_enemy=True))
                particle_system.emit_directed(self.x + math.cos(angle)*22, self.y + math.sin(angle)*22, COLOR_DANGER, angle, count=3)
            
            sound_manager.play('hit')
            self.shoot_timer = self.shoot_cooldown * self.slow_multiplier

    def draw(self, surface, camera_offset):
        pos = (int(self.x + camera_offset[0]), int(self.y + camera_offset[1]))
        
        glow = pygame.Surface((self.radius * 2.5, self.radius * 2.5), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*self.color, 50), (self.radius * 1.25, self.radius * 1.25), self.radius * 1.25)
        surface.blit(glow, (pos[0] - self.radius * 1.25, pos[1] - self.radius * 1.25))
        
        draw_enemy(surface, self, pos)
        
        health_pct = self.health / self.max_health
        pygame.draw.circle(surface, (255, 100, 0) if health_pct > 0.5 else (255, 0, 0), pos, self.radius + 2, 1)

    def take_damage(self, damage):
        self.health -= damage

# ============================================================================
# LEADERBOARD
# ============================================================================
class Leaderboard:
    def __init__(self, filename="/tmp/neon_leaderboard.json"):
        self.filename = filename
        self.entries = []
        self.load()

    def load(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.entries = json.load(f)
                    self.entries.sort(key=lambda x: x['score'], reverse=True)
        except:
            self.entries = []

    def save(self):
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.entries, f)
        except:
            pass

    def add_score(self, name, score, wave, kills, mode):
        rank = self.get_rank(score)
        self.entries.append({
            'name': name[:15],
            'score': score,
            'wave': wave,
            'kills': kills,
            'rank': rank,
            'mode': mode
        })
        self.entries.sort(key=lambda x: x['score'], reverse=True)
        self.entries = self.entries[:100]
        self.save()

    def get_rank(self, score):
        ranks = [("GODLIKE", 15000), ("LEGENDARY", 10000), ("ELITE", 5000), ("PRO", 2000), ("EXPERT", 1000), ("NOVICE", 0)]
        for rank, threshold in ranks:
            if score >= threshold:
                return rank
        return "NOVICE"

    def get_top(self, n=8):
        return self.entries[:n]

# ============================================================================
# UI MANAGER
# ============================================================================
class UIManager:
    def __init__(self):
        pygame.font.init()
        self.font_main = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 52, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 12)

    def draw_hud(self, surface, player, wave, enemies_alive, mode):
        pygame.draw.rect(surface, COLOR_BG_ACCENT, (20, 20, 250, 20))
        health_width = int(250 * max(0, player.health) / player.max_health)
        pygame.draw.rect(surface, COLOR_SUCCESS if player.health > 30 else COLOR_DANGER, (20, 20, health_width, 20))
        pygame.draw.rect(surface, COLOR_PRIMARY, (20, 20, 250, 20), 2)
        surface.blit(self.font_main.render(f"HP: {int(player.health)}/{player.max_health}", True, COLOR_TEXT), (30, 22))

        if player.shield > 0:
            pygame.draw.rect(surface, COLOR_BG_ACCENT, (20, 45, 250, 12))
            shield_width = int(250 * player.shield / player.max_shield)
            pygame.draw.rect(surface, COLOR_ACCENT, (20, 45, shield_width, 12))
            pygame.draw.rect(surface, COLOR_ACCENT, (20, 45, 250, 12), 1)

        surface.blit(self.font_main.render(f"W: {player.current_weapon.name} (1-5)", True, COLOR_SECONDARY), (20, 65))

        boost_y = 85
        if player.damage_boost_timer > 0:
            surface.blit(self.font_small.render(f"DMG+ {player.damage_boost_timer:.1f}s", True, COLOR_DANGER), (20, boost_y))
            boost_y += 15
        if player.speed_boost_timer > 0:
            surface.blit(self.font_small.render(f"SPD+ {player.speed_boost_timer:.1f}s", True, COLOR_WARNING), (20, boost_y))
            boost_y += 15
        if player.slow_enemies:
            surface.blit(self.font_small.render(f"SLOW {player.slow_timer:.1f}s", True, COLOR_ACCENT), (20, boost_y))

        surface.blit(self.font_main.render(f"WAVE: {wave}", True, COLOR_PRIMARY), (WIDTH - 280, 20))
        surface.blit(self.font_main.render(f"SCORE: {player.score}", True, COLOR_SUCCESS), (WIDTH - 280, 45))
        surface.blit(self.font_small.render(f"COMBO: x{player.combo} ({player.combo_timer:.1f}s)", True, COLOR_SECONDARY), (WIDTH - 280, 70))
        surface.blit(self.font_small.render(f"ENEMIES: {enemies_alive} | KILLS: {player.kills}", True, COLOR_DANGER), (WIDTH - 280, 85))
        surface.blit(self.font_small.render(f"MODE: {mode}", True, COLOR_ACCENT), (WIDTH - 280, 100))

    def draw_minimap(self, surface, player, enemies, powerups):
        minimap_size = int(150 * (WIDTH / 1400))
        minimap_x = WIDTH - minimap_size - 20
        minimap_y = 120
        
        pygame.draw.rect(surface, COLOR_BG_ACCENT, (minimap_x, minimap_y, minimap_size, minimap_size))
        pygame.draw.rect(surface, COLOR_PRIMARY, (minimap_x, minimap_y, minimap_size, minimap_size), 2)
        
        scale_x = minimap_size / WIDTH
        scale_y = minimap_size / HEIGHT
        
        pygame.draw.circle(surface, COLOR_PRIMARY, (int(minimap_x + player.x * scale_x), int(minimap_y + player.y * scale_y)), 3)
        
        for enemy in enemies:
            pygame.draw.circle(surface, enemy.color, (int(minimap_x + enemy.x * scale_x), int(minimap_y + enemy.y * scale_y)), 2)
        
        for pu in powerups:
            color = pu.power_type.value["color"]
            pygame.draw.circle(surface, color, (int(minimap_x + pu.x * scale_x), int(minimap_y + pu.y * scale_y)), 2)

    def draw_game_over(self, surface, player, wave, kills, mode):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((5, 5, 15, 240))
        surface.blit(overlay, (0, 0))

        title = self.font_large.render("GAME OVER", True, COLOR_DANGER)
        surface.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 150))

        stats = [
            f"Final Score: {player.score}",
            f"Wave Reached: {wave}",
            f"Enemies Defeated: {kills}",
            f"Game Mode: {mode}",
            f"Rank: {Leaderboard().get_rank(player.score)}",
            "",
            "Press R to Restart | ESC to Quit"
        ]
        
        for i, stat in enumerate(stats):
            surface.blit(self.font_main.render(stat, True, COLOR_TEXT), (WIDTH // 2 - 150, HEIGHT // 2 - 30 + i * 35))

# ============================================================================
# MAIN GAME
# ============================================================================
class Game:
    def __init__(self, mode="SURVIVAL"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption(f"⚡ NEON COMBAT: ULTIMATE ARENA ⚡ [{WIDTH}x{HEIGHT}]")
        self.clock = pygame.time.Clock()

        self.sound_manager = SoundManager()
        self.ui_manager = UIManager()
        self.leaderboard = Leaderboard()

        self.mode = mode
        self.reset_game()

    def reset_game(self):
        self.player = Player(WIDTH // 4, HEIGHT // 2)
        self.enemies = []
        self.bullets = []
        self.powerups = []
        self.obstacles = []
        self.particle_system = ParticleSystem()
        
        self.wave = 1
        self.wave_timer = 0
        self.wave_spawned = False
        self.game_over = False
        self.screen_shake_time = 0
        self.global_time = 0
        
        self.generate_obstacles()

    def generate_obstacles(self):
        self.obstacles = []
        num_obstacles = max(2, self.wave // 3)
        for _ in range(num_obstacles):
            x = random.uniform(WIDTH * 0.2, WIDTH * 0.8)
            y = random.uniform(HEIGHT * 0.2, HEIGHT * 0.8)
            width = random.uniform(60, 120)
            height = random.uniform(40, 100)
            obstacle_type = random.choice(["static", "moving"]) if self.wave > 5 else "static"
            self.obstacles.append(Obstacle(x, y, width, height, obstacle_type))

    def get_wave_enemies(self):
        basic_enemies = [EnemyType.DRONE, EnemyType.ROGUE, EnemyType.SWARM]
        intermediate = [EnemyType.TANK, EnemyType.SNIPER, EnemyType.CHARGER, EnemyType.SHIELDER, EnemyType.HEALER]
        advanced = [EnemyType.ASSASSIN, EnemyType.EXPLOSIVE, EnemyType.FREEZER]
        
        if self.wave <= 2:
            return basic_enemies
        elif self.wave <= 5:
            return basic_enemies + intermediate
        else:
            return basic_enemies + intermediate + advanced

    def spawn_wave(self):
        available_enemies = self.get_wave_enemies()
        enemy_count = min(3 + self.wave // 2, 12)
        
        for _ in range(enemy_count):
            enemy_type = random.choice(available_enemies)
            x = random.choice([WIDTH * 0.1, WIDTH * 0.9])
            y = random.uniform(HEIGHT * 0.1, HEIGHT * 0.9)
            self.enemies.append(Enemy(x, y, enemy_type))

        if self.wave % 5 == 0:
            boss_type = random.choice([EnemyType.BOSS_TANK, EnemyType.BOSS_NOVA])
            self.enemies.append(Enemy(WIDTH // 2, HEIGHT * 0.15, boss_type))
            self.sound_manager.play('boss')

        self.wave_spawned = True
        self.generate_obstacles()

    def process_collisions(self):
        for b in self.bullets[:]:
            if b.is_enemy:
                dist = math.hypot(b.x - self.player.x, b.y - self.player.y)
                if dist < self.player.radius + b.radius:
                    damage = self.player.take_damage(b.damage)
                    b.should_remove = True
                    self.sound_manager.play('hit')
                    self.particle_system.emit(b.x, b.y, COLOR_DANGER, count=8)
                    self.particle_system.add_damage_number(b.x, b.y, damage, COLOR_DANGER)
                    self.screen_shake_time = 0.15
            else:
                for enemy in self.enemies[:]:
                    dist = math.hypot(b.x - enemy.x, b.y - enemy.y)
                    if dist < enemy.radius + b.radius:
                        enemy.take_damage(b.damage)
                        b.should_remove = True
                        self.sound_manager.play('hit')
                        self.particle_system.emit(b.x, b.y, enemy.color, count=12)
                        self.particle_system.add_damage_number(b.x, b.y, b.damage, COLOR_SUCCESS)
                        
                        if enemy.health <= 0:
                            points = self.player.add_score(enemy.score_value)
                            self.particle_system.emit(enemy.x, enemy.y, COLOR_SUCCESS, count=25)
                            
                            if random.random() < 0.15 + (self.wave * 0.02):
                                powerup_type = random.choice(list(PowerUpType))
                                self.powerups.append(PowerUp(enemy.x, enemy.y, powerup_type))
                            
                            self.enemies.remove(enemy)
                            self.sound_manager.play('levelup')
                        
                        self.screen_shake_time = 0.1
                        break

        for pu in self.powerups[:]:
            dist = math.hypot(pu.x - self.player.x, pu.y - self.player.y)
            if dist < self.player.radius + pu.radius:
                self.player.apply_powerup(pu.power_type)
                self.sound_manager.play('powerup')
                self.particle_system.emit(pu.x, pu.y, pu.power_type.value["color"], count=20)
                self.powerups.remove(pu)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                global WIDTH, HEIGHT
                WIDTH, HEIGHT = event.size
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                pygame.display.set_caption(f"⚡ NEON COMBAT: ULTIMATE ARENA ⚡ [{WIDTH}x{HEIGHT}]")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_1:
                    self.player.change_weapon(WeaponType.PISTOL)
                elif event.key == pygame.K_2:
                    self.player.change_weapon(WeaponType.RIFLE)
                elif event.key == pygame.K_3:
                    self.player.change_weapon(WeaponType.SHOTGUN)
                elif event.key == pygame.K_4:
                    self.player.change_weapon(WeaponType.PLASMA)
                elif event.key == pygame.K_5:
                    self.player.change_weapon(WeaponType.LASER)
        return True

    def run(self):
        running = True
        while running:
            raw_dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            dt = raw_dt
            self.global_time += dt

            running = self.handle_events()

            if not self.game_over:
                self.player.update(dt, self.bullets, self.sound_manager, self.particle_system, self.obstacles)

                for enemy in self.enemies[:]:
                    enemy.update(dt, self.player, self.bullets, self.sound_manager, self.particle_system, self.obstacles, self.player.slow_enemies)

                for pu in self.powerups[:]:
                    pu.update(dt)
                    if pu.should_remove:
                        self.powerups.remove(pu)

                for obs in self.obstacles:
                    obs.update(dt)

                for b in self.bullets[:]:
                    b.update(dt)
                    if b.should_remove:
                        self.bullets.remove(b)

                self.particle_system.update(dt)
                self.process_collisions()

                if not self.wave_spawned:
                    self.wave_timer += dt
                    if self.wave_timer > 2:
                        self.spawn_wave()

                if len(self.enemies) == 0 and self.wave_spawned:
                    self.wave += 1
                    self.wave_spawned = False

                if self.player.health <= 0:
                    self.game_over = True
                    self.leaderboard.add_score("PLAYER", self.player.score, self.wave, self.player.kills, self.mode)

            cam_offset_x = 0
            cam_offset_y = 0
            if self.screen_shake_time > 0:
                self.screen_shake_time -= raw_dt
                cam_offset_x = random.randint(-4, 4)
                cam_offset_y = random.randint(-4, 4)

            self.screen.fill(COLOR_BG)
            camera_tuple = (cam_offset_x, cam_offset_y)

            for obs in self.obstacles:
                obs.draw(self.screen, camera_tuple)

            self.particle_system.draw(self.screen, camera_tuple, self.ui_manager.font_small)

            for enemy in self.enemies:
                enemy.draw(self.screen, camera_tuple)

            if self.player.health > 0:
                self.player.draw(self.screen, camera_tuple)

            for b in self.bullets:
                b.draw(self.screen, camera_tuple)

            for pu in self.powerups:
                pu.draw(self.screen, camera_tuple)

            self.ui_manager.draw_hud(self.screen, self.player, self.wave, len(self.enemies), self.mode)
            self.ui_manager.draw_minimap(self.screen, self.player, self.enemies, self.powerups)

            if self.game_over:
                self.ui_manager.draw_game_over(self.screen, self.player, self.wave, self.player.kills, self.mode)

            pygame.display.flip()

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game(mode="SURVIVAL")
    game.run()