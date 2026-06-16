import pyaudio
import numpy as np
import threading
import queue
import re
import collections
from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class AudioNLPEngine:
    """Asynchronously streams audio, performs local transcription using Faster-Whisper,
    tracks vocal fillers, and runs real-time sentiment analysis via VADER."""
    def __init__(self, model_size="base"):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        
        # Initialize specialized processing engines
        print(f"[INIT] Loading Faster-Whisper Model ({model_size})...")
        self.whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.vader = SentimentIntensityAnalyzer()
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.lock = threading.Lock()
        
        # Exposed Thread-Safe Metrics Payload
        self.shared_payload = {
            "transcript": "Listening...",
            "filler_count": 0,
            "sentiment_score": 100.0,  # Starts normalized at perfect baseline
            "sentiment_label": "NEUTRAL",
            "speech_detected": False
        }
        
        # Internal Analysis Accumulators
        self.filler_words_pattern = re.compile(r'\b(um|uh|like|ah|eh|er)\b', re.IGNORECASE)
        self.accumulated_fillers = 0
        self.audio_history_buffer = collections.deque(maxlen=self.RATE * 7) # Rolling 7-second sequence block

    def start(self):
        self.is_running = True
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.analysis_thread = threading.Thread(target=self._nlp_analysis_loop, daemon=True)
        
        self.capture_thread.start()
        self.analysis_thread.start()
        return self

    def _capture_loop(self):
        while self.is_running:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                self.audio_queue.put(audio_chunk)
            except Exception:
                continue

    def _nlp_analysis_loop(self):
        # Local window sliding control metrics
        processing_block_size = self.RATE * 3 # Chunk segments parsed in 3-second chunks
        local_buffer = []

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                local_buffer.extend(chunk)
                
                if len(local_buffer) >= processing_block_size:
                    audio_data = np.array(local_buffer, dtype=np.float32) / 32768.0
                    local_buffer = [] # Reset accumulation line
                    
                    # Run Fast In-Memory Transcription Segment
                    segments, info = self.whisper.transcribe(audio_data, beam_size=1)
                    text_segment = "".join([seg.text for seg in segments]).strip()
                    
                    if text_segment:
                        # Extract Filler Word Matches via Pattern Analysis
                        matches = self.filler_words_pattern.findall(text_segment)
                        self.accumulated_fillers += len(matches)
                        
                        # Process Structural Linguistic Sentiment Scores via VADER
                        sentiment_payload = self.vader.polarity_scores(text_segment)
                        compound_val = sentiment_payload["compound"] # Scaled value from -1.0 to +1.0
                        
                        # Convert linguistic polarity into a clean 0-100 rating scale
                        normalized_sentiment = (compound_val + 1.0) * 50.0 
                        
                        if compound_val >= 0.15:
                            label = "POSITIVE / CONFIDENT"
                        elif compound_val <= -0.15:
                            label = "NEGATIVE / HESITANT"
                        else:
                            label = "NEUTRAL"
                            
                        with self.lock:
                            self.shared_payload["transcript"] = text_segment
                            self.shared_payload["filler_count"] = self.accumulated_fillers
                            self.shared_payload["sentiment_score"] = float(normalized_sentiment)
                            self.shared_payload["sentiment_label"] = label
                            self.shared_payload["speech_detected"] = True
            except queue.Empty:
                continue

    def get_metrics(self):
        with self.lock:
            return self.shared_payload.copy()

    def stop(self):
        self.is_running = False
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception:
            pass