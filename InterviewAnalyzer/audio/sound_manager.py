import pygame
import threading
import queue
import time
import numpy as np

class SoundManager:
    """An asynchronous audio synthesis engine that runs on a separate thread. 
    It plays real-time feedback sounds without causing audio stuttering or frame drops."""
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=1)
        self.queue = queue.Queue()
        self.started = True
        self.last_played_time = 0
        self.cooldown_period = 4.0
        
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _synthesize_tone(self, frequency, duration_ms, type_wave='sine'):
        sample_rate = 22050
        num_samples = int(sample_rate * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
        
        if type_wave == 'sine':
            samples = np.sin(2 * np.pi * frequency * t)
        elif type_wave == 'square':
            samples = np.sign(np.sin(2 * np.pi * frequency * t))
        else:
            samples = np.sin(2 * np.pi * frequency * t)
            
        audio_buffer = (samples * 32767).astype(np.int16)
        return pygame.mixer.Sound(buffer=audio_buffer)

    def trigger_feedback(self, tier_status):
        current_time = time.time()
        if current_time - self.last_played_time > self.cooldown_period:
            self.queue.put(tier_status)
            self.last_played_time = current_time

    def _worker_loop(self):
        low_confidence_tone = self._synthesize_tone(frequency=261.63, duration_ms=400, type_wave='square')
        neutral_tone = self._synthesize_tone(frequency=440.00, duration_ms=200, type_wave='sine')
        high_confidence_tone = self._synthesize_tone(frequency=523.25, duration_ms=150, type_wave='sine')

        while self.started:
            try:
                tier = self.queue.get(timeout=0.1)
                if tier == "LOW_CONFIDENCE":
                    low_confidence_tone.set_volume(0.15)
                    low_confidence_tone.play()
                elif tier == "MEDIUM_CONFIDENCE":
                    neutral_tone.set_volume(0.1)
                    neutral_tone.play()
                elif tier == "HIGH_CONFIDENCE":
                    high_confidence_tone.set_volume(0.2)
                    high_confidence_tone.play()
                self.queue.task_done()
            except queue.Empty:
                continue

    def close(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        pygame.mixer.quit()