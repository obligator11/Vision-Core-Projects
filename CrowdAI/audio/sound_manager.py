import numpy as np
import threading
import time

# Enforce explicit headless audio instantiation matrix initialization wrapper layers
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class SoundManager:
    """
    Thread-isolated asynchronous sound engine that constructs sound signatures mathematically 
    in system memory to eliminate the need for external asset dependencies.
    """
    def __init__(self):
        # Configure pygame backend parameters safely 
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.mixer.init()
        
        self.current_state = "LOW"
        self.worker_active = True
        self.lock = threading.Lock()
        
        # Generate wave profiles directly inside buffer vectors
        self.warning_beep = self._synthesize_sine_wave(880, 0.12)  # High frequency caution pulse
        self.alarm_loop = self._synthesize_sine_wave(440, 0.35)   # Piercing rhythmic notice signal
        
        # Launch dedicated background execution process thread loop container
        self.audio_thread = threading.Thread(target=self._audio_orchestration_loop, args=(), daemon=True)
        self.audio_thread.start()

    def _synthesize_sine_wave(self, frequency, duration, sample_rate=44100):
        """
        Generates raw audio bytes in memory to ensure full standalone operation.
        """
        total_samples = int(sample_rate * duration)
        time_axis = np.linspace(0, duration, total_samples, endpoint=False)
        wave_vector = np.sin(2 * np.pi * frequency * time_axis)
        
        # Scale to match signed 16-bit integer boundaries
        quantized_audio = (wave_vector * 32767).astype(np.int16)
        return pygame.mixer.Sound(buffer=quantized_audio)

    def trigger_alert_state(self, state_string):
        """
        Thread-safe external configuration interface update state hook pointer.
        """
        with self.lock:
            self.current_state = state_string

    def _audio_orchestration_loop(self):
        while self.worker_active:
            with self.lock:
                active_state = self.current_state
                
            if active_state == "MEDIUM":
                self.warning_beep.play()
                time.sleep(0.65)  # Measured pulse cycle gap
            elif active_state == "HIGH CONGESTION":
                self.alarm_loop.play()
                time.sleep(0.45)  # High-frequency warning trigger cadence
            else:
                time.sleep(0.1)   # Resting polling pause frequency configuration

    def terminate(self):
        self.worker_active = False
        pygame.mixer.quit()