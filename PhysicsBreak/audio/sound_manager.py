import pygame
import numpy as np
import threading
import queue

class SoundManager:
    """
    Synthesizes and runs audio signals via an internal worker thread,
    preventing frame drops or pipeline blocks.
    """
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=1)
        self.sample_rate = 22050
        self.audio_queue = queue.Queue(maxsize=3)
        
        self.worker = threading.Thread(target=self._audio_loop, daemon=True)
        self.worker.start()
        
    def _generate_sine_wave(self, frequency: float, duration: float, volume: float = 0.4) -> pygame.mixer.Sound:
        """Generates raw sinusoidal wave byte arrays directly inside RAM buffers."""
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        wave = np.sin(2 * np.pi * frequency * t) * 32767
        wave = wave.astype(np.int16)
        return pygame.mixer.Sound(buffer=wave)

    def _generate_glitch_sound(self) -> pygame.mixer.Sound:
        """Synthesizes high-frequency glitch tones mimicking digital breakdown signals."""
        duration = 0.12
        num_samples = int(duration * self.sample_rate)
        # Create a frequency sweep combined with random frequency phase noise
        t = np.linspace(0, duration, num_samples, endpoint=False)
        noise = np.random.uniform(-1.0, 1.0, num_samples) * 400
        freq_sweep = 800 + (t * -4000) + noise
        wave = np.sin(2 * np.pi * freq_sweep * t) * 20000
        return pygame.mixer.Sound(buffer=wave.astype(np.int16))

    def _audio_loop(self):
        while True:
            alert_type = self.audio_queue.get()
            try:
                if alert_type == "ANOMALY":
                    snd = self._generate_glitch_sound()
                    snd.play()
                elif alert_type == "CRITICAL":
                    # Instant synchronous generation of alarm waves
                    snd = self._generate_sine_wave(880.0, 0.25, volume=0.6)
                    snd.play()
            except Exception as e:
                print(f"[AUDIO ERROR]: {e}")
            self.audio_queue.task_done()

    def trigger_alert(self, severity: str):
        """Dispatches notification triggers safely into background queues."""
        if severity in ["ANOMALY", "CRITICAL"]:
            try:
                self.audio_queue.put_nowait(severity)
            except queue.Full:
                pass # Bypasses dropped calls if standard buffers fill up