import os
import time
import pygame
from threading import Thread
from config import SystemConfig

class SoundManager:
    """Asynchronous, non-blocking audio engine utilizing multi-channel execution controllers."""
    
    def __init__(self) -> None:
        pygame.mixer.init()
        self.running: bool = True
        self.current_status: str = "SAFE"
        
        # Load audio resources or construct synthetic failbacks structures mappings if missing
        if os.path.exists(SystemConfig.ALERT_SOUND_FILE):
            self.alert_sound = pygame.mixer.Sound(SystemConfig.ALERT_SOUND_FILE)
        else:
            # Fallback initialization to ignore crash paths configurations errors
            self.alert_sound = None
            print(f"[WARN] Audio source configuration path file absolute error: '{SystemConfig.ALERT_SOUND_FILE}' missing.")

        self.thread = Thread(target=self._audio_processing_loop, name="AsynchronousAudioEngineThread", args=())
        self.thread.daemon = True
        self.thread.start()

    def update_status(self, risk_status: str) -> None:
        """Sets active monitoring thresholds targets state parameters references strings variables."""
        self.current_status = risk_status

    def _audio_processing_loop(self) -> None:
        """Main internal threat run loop parsing playing frequency profiles dependencies tracks safely."""
        last_warning_time = 0.0
        danger_channel = pygame.mixer.Channel(0) if pygame.mixer.get_init() else None
        
        while self.running:
            if self.alert_sound is None:
                time.sleep(0.1)
                continue
                
            now = time.time()
            if self.current_status == "DANGER":
                if danger_channel and not danger_channel.get_busy():
                    # Play continuously inside high importance priority channel routes loops configurations
                    danger_channel.play(self.alert_sound, loops=-1)
            elif self.current_status == "WARNING":
                if danger_channel and danger_channel.get_busy():
                    danger_channel.stop()
                if now - last_warning_time >= SystemConfig.WARNING_ALERT_INTERVAL:
                    # Pulsed alert channel tracking models implementation loops
                    pygame.mixer.Channel(1).play(self.alert_sound)
                    last_warning_time = now
            else:
                # SAFE status ensures all channel entities terminate playing execution calls instantly
                if danger_channel and danger_channel.get_busy():
                    danger_channel.stop()
                    
            time.sleep(0.05)

    def close(self) -> None:
        """Terminates internal audio streams allocations context configurations safely."""
        self.running = False
        pygame.mixer.quit()