import os
import math
import numpy as np
import threading
import time

# Suppress pygame launch banners in standard streams
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class SoundManager:
    """
    Synthesizes and handles non-blocking programmatic frequencies 
    simulating reverse swooshes and temporal distortions using hardware threads.
    """
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=1)
        self.play_lock = threading.Lock()
        self.active_thread = None
        self.terminate_signal = False

    def _generate_sine_wave(self, freq_start, freq_end, duration, volume=0.4):
        sample_rate = 22050
        num_samples = int(sample_rate * duration)
        buf = np.zeros((num_samples,), dtype=np.int16)
        
        for i in range(num_samples):
            t = float(i) / sample_rate
            # Dynamic frequency sweep modulation mapping
            alpha = t / duration
            current_freq = freq_start + alpha * (freq_end - freq_start)
            
            val = idx_val = math.sin(2.0 * math.pi * current_freq * t)
            # Apply cosine fade to eliminate audio clicks
            fade = 1.0
            if i < 400:
                fade = float(i) / 400.0
            elif i > num_samples - 400:
                fade = float(num_samples - i) / 400.0
                
            buf[i] = int(val * fade * volume * 32767)
            
        return pygame.mixer.Sound(buffer=buf)

    def _play_async_loop(self, is_trigger_swoosh):
        if is_trigger_swoosh:
            # Reverse rising swoosh audio calculation
            sound = self._generate_sine_wave(120, 480, 0.4, volume=0.5)
            sound.play()
        else:
            # Continuous temporal low-frequency distortion hum
            while not self.terminate_signal:
                sound = self._generate_sine_wave(90, 75, 0.2, volume=0.3)
                channel = sound.play()
                while channel.get_busy() and not self.terminate_signal:
                    time.sleep(0.01)

    def play_rewind_trigger(self):
        with self.play_lock:
            self.stop_all()
            self.terminate_signal = False
            self.active_thread = threading.Thread(target=self._play_async_loop, args=(True,), daemon=True)
            self.active_thread.start()

    def start_ambient_hum(self):
        with self.play_lock:
            if self.active_thread and not self.terminate_signal:
                return
            self.terminate_signal = False
            self.active_thread = threading.Thread(target=self._play_async_loop, args=(False,), daemon=True)
            self.active_thread.start()

    def stop_all(self):
        self.terminate_signal = True
        pygame.mixer.stop()
        if self.active_thread and self.active_thread.is_alive():
            self.active_thread.join(timeout=0.1)