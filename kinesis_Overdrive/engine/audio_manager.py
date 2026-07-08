"""
Synthesizes and plays all sound effects. No external audio assets needed —
everything is generated procedurally on first run and cached to disk.
"""

import math
import os
import random
import struct
import wave

import pygame

from config import HIT_SOUND_PATH, START_SOUND_PATH, LEVELUP_SOUND_PATH, MISS_SOUND_PATH

SAMPLE_RATE = 44100


def _synthesize(filename, duration, freq_start, freq_decay, noise_amount=0.0,
                 wave_shape="sine"):
    if os.path.exists(filename):
        return
    num_samples = int(SAMPLE_RATE * duration)
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        for i in range(num_samples):
            t = i / SAMPLE_RATE
            freq = freq_start * math.exp(-freq_decay * t)
            envelope = math.exp(-15 * t)
            if wave_shape == "sine":
                tone = math.sin(2 * math.pi * freq * t)
            elif wave_shape == "square":
                tone = 1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0
            else:
                tone = math.sin(2 * math.pi * freq * t)
            noise = random.uniform(-1, 1) * noise_amount
            sample = (tone + noise) * envelope
            val = max(-32768, min(32767, int(sample * 32767.0)))
            wav_file.writeframes(struct.pack("h", val))


class AudioManager:
    def __init__(self):
        _synthesize(HIT_SOUND_PATH, duration=0.2, freq_start=800,
                    freq_decay=20, noise_amount=0.4)
        _synthesize(START_SOUND_PATH, duration=0.3, freq_start=440,
                    freq_decay=6, noise_amount=0.0)
        _synthesize(LEVELUP_SOUND_PATH, duration=0.35, freq_start=660,
                    freq_decay=4, noise_amount=0.05)
        _synthesize(MISS_SOUND_PATH, duration=0.15, freq_start=180,
                    freq_decay=10, noise_amount=0.2, wave_shape="square")

        self.enabled = False
        self._sounds = {}
        try:
            pygame.mixer.init()
            self._load("hit", HIT_SOUND_PATH)
            self._load("start", START_SOUND_PATH)
            self._load("levelup", LEVELUP_SOUND_PATH)
            self._load("miss", MISS_SOUND_PATH)
            self.enabled = True
        except pygame.error:
            self.enabled = False

    def _load(self, name, path):
        if os.path.exists(path):
            snd = pygame.mixer.Sound(path)
            snd.set_volume(0.8)
            self._sounds[name] = snd

    def play(self, name):
        if not self.enabled or name not in self._sounds:
            return
        channel = pygame.mixer.find_channel(True)
        if channel is not None:
            channel.play(self._sounds[name])

    def shutdown(self):
        if self.enabled:
            pygame.mixer.quit()
