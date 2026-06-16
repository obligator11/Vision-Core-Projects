import os
import queue
import threading
import numpy as np

# Suppress Pygame initialization output messages in terminal console windows
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

class AsynchronousSoundEngine:
    """
    Processes low-latency notification sound markers inside an independent thread queue,
    synthesizing dynamic audio arrays via NumPy to prevent UI frame-drops.
    """
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        self.audio_queue = queue.Queue()
        self.sample_rate = 44100
        
        # Start the background task worker loop
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def generate_sine_wave(self, frequency: float, duration_sec: float, volume: float = 0.4) -> pygame.mixer.Sound:
        num_samples = int(self.sample_rate * duration_sec)
        time_axis = np.linspace(0, duration_sec, num_samples, endpoint=False)
        # Mathematical compilation of raw wave signals
        amplitude_array = np.sin(2 * np.pi * frequency * time_axis) * 32767
        stereo_signal = np.vstack((amplitude_array, amplitude_array)).T.astype(np.int16)
        return pygame.mixer.Sound(buffer=stereo_signal)

    def trigger_warning_beep(self) -> None:
        """High lag trigger signal."""
        self.audio_queue.put(('warning', 880, 0.15, 0.5))

    def trigger_success_tone(self) -> None:
        """System lag improvement tone trigger."""
        self.audio_queue.put(('success', 523.25, 0.25, 0.3))

    def _worker_loop(self) -> None:
        while True:
            try:
                msg_type, freq, duration, vol = self.audio_queue.get(timeout=1.0)
                if msg_type == 'warning':
                    sound = self.generate_sine_wave(freq, duration, vol)
                    sound.play()
                elif msg_type == 'success':
                    # Synthesize an arpeggio sequence for immediate clear tone execution
                    sound1 = self.generate_sine_wave(freq, duration * 0.5, vol)
                    sound2 = self.generate_sine_wave(freq * 1.25, duration, vol)
                    sound1.play()
                    pygame.time.wait(70)
                    sound2.play()
                self.audio_queue.task_done()
            except queue.Empty:
                continue