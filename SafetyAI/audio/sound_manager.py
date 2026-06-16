"""
SafetyAI - Asynchronous Non-Blocking Hardware Audio Mixer Thread
Uses Pygame to generate pure math sine wave frequencies entirely at runtime.
"""
import threading
import time
import numpy as np
import pygame

class SoundManager:
    def __init__(self):
        # Configure the OS hardware mixer buffer stream engine pipeline safely
        pygame.mixer.pre_init(44100, -16, 1, 512)
        pygame.mixer.init()
        
        self.current_risk_level = "SAFE"
        self.is_running = True
        
        # Sythesize native pure array tones directly to eliminate file dependency issues
        self.beep_warning = self._generate_sine_sound(880.0, 0.15, volume=0.4)  # 880Hz warning beep
        self.beep_danger = self._generate_sine_sound(1200.0, 0.45, volume=0.95) # 1.2kHz piercing alarm
        
        # Thread isolation initialization
        self.audio_thread = threading.Thread(target=self._audio_execution_loop, daemon=True)
        self.audio_thread.start()
        
    def _generate_sine_sound(self, frequency: float, duration: float, volume=0.5) -> pygame.mixer.Sound:
        """
        Generates and signs a raw mathematical waveform array directly into RAM.
        """
        sample_rate = 44100
        total_samples = int(sample_rate * duration)
        
        # Calculate angular velocity phase values
        t = np.linspace(0, duration, total_samples, endpoint=False)
        wave = np.sin(2 * np.pi * frequency * t) * 32767
        signed_16bit_buffer = wave.astype(np.int16)
        
        sound = pygame.mixer.Sound(buffer=signed_16bit_buffer)
        sound.set_volume(volume)
        return sound
        
    def set_risk_level(self, level: str):
        """
        Thread-safe injection vector updates the targeting state buffer safely.
        """
        self.current_risk_level = level
        
    def _audio_execution_loop(self):
        """
        Background worker processing audio loop loops.
        Prevents audio block processing overhead from dropping visual frames.
        """
        while self.is_running:
            state = self.current_risk_level
            
            if state == "SAFE":
                time.sleep(0.1)
                
            elif state == "WARNING":
                self.beep_warning.play()
                # Periodic notification pacing loop window
                time.sleep(1.8)
                
            elif state == "DANGER":
                self.beep_danger.play()
                # Rapid piercing continuous alert loop cadence window
                time.sleep(0.55)
                
            else:
                time.sleep(0.1)
                
    def terminate(self):
        """
        Safely halts audio output threads and releases hardware access bindings.
        """
        self.is_running = False
        pygame.mixer.quit()