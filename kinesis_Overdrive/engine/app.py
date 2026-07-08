"""
KinesisApp — the top-level orchestrator.

Owns every subsystem (camera, pose tracking, gesture recognition, audio,
scoring, spawner, renderer) and drives the single main loop that ties them
together through the GameState machine. Gesture-first: an open palm starts
and restarts rounds, a fist pauses/resumes. Keyboard is a fallback, never
required.
"""

import time

import cv2

import config
from engine.camera import Camera
from engine.pose_tracker import PoseTracker
from engine.gesture_recognizer import GestureRecognizer, GestureHoldTracker, OPEN_PALM, FIST
from engine.audio_manager import AudioManager
from engine.score_manager import ScoreManager
from engine.state_machine import StateMachine, GameState
from entities.particle import ParticleSystem
from entities.target import TargetSpawner
from entities.trail import HandTrail
from ui.renderer import Renderer


class KinesisApp:
    def __init__(self):
        self.camera = Camera().start()
        self.w, self.h = self.camera.width, self.camera.height

        self.pose_tracker = PoseTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.audio = AudioManager()
        self.score_manager = ScoreManager()
        self.renderer = Renderer(self.w, self.h)

        self.difficulty_key = config.DEFAULT_DIFFICULTY
        self.difficulty = config.DIFFICULTIES[self.difficulty_key]
        self.spawner = TargetSpawner(self.w, self.h, self.difficulty)
        self.particles = ParticleSystem()

        self.left_trail = HandTrail(config.COLOR_LEFT_TRAIL, self.h)
        self.right_trail = HandTrail(config.COLOR_RIGHT_TRAIL, self.h)

        self.state = StateMachine(GameState.MENU)

        self.palm_hold = GestureHoldTracker(OPEN_PALM, config.PALM_HOLD_SECONDS)
        self.fist_hold = GestureHoldTracker(FIST, config.FIST_HOLD_SECONDS)

        self.round_start_time = None
        self.countdown_start_time = None
        self.pause_started_at = None
        self.total_paused_duration = 0.0

        self._running = True
        self._last_gestures = []

    # -- Round lifecycle -----------------------------------------------
    def _begin_countdown(self, now):
        self.state.transition(GameState.COUNTDOWN, now)
        self.countdown_start_time = now

    def _begin_round(self, now):
        self.audio.play("start")
        self.state.transition(GameState.PLAYING, now)
        self.round_start_time = now
        self.total_paused_duration = 0.0
        self.score_manager.reset()
        self.spawner.set_difficulty(self.difficulty)
        self.spawner.clear()

    def _end_round(self, now):
        self.state.transition(GameState.GAME_OVER, now)
        self.score_manager.save_score(self.difficulty.name)

    def _time_remaining(self, now):
        if self.round_start_time is None:
            return config.ROUND_SECONDS
        elapsed = (now - self.round_start_time) - self.total_paused_duration
        return max(0.0, config.ROUND_SECONDS - elapsed)

    # -- Input handling ---------------------------------------------------
    def _handle_key(self, key, now):
        if key == ord('q'):
            self._running = False
        elif key == ord('p'):
            if self.state.state == GameState.PLAYING:
                self.pause_started_at = now
                self.state.transition(GameState.PAUSED, now)
            elif self.state.state == GameState.PAUSED:
                self._resume_from_pause(now)
        elif key in (ord('1'), ord('2'), ord('3')) and self.state.state == GameState.MENU:
            mapping = {ord('1'): "EASY", ord('2'): "NORMAL", ord('3'): "INSANE"}
            self.difficulty_key = mapping[key]
            self.difficulty = config.DIFFICULTIES[self.difficulty_key]

    def _resume_from_pause(self, now):
        if self.pause_started_at is not None:
            self.total_paused_duration += (now - self.pause_started_at)
            self.pause_started_at = None
        self.state.transition(GameState.PLAYING, now)

    def _handle_gestures(self, now):
        palm_progress, palm_confirmed = self.palm_hold.update(self._last_gestures, now)
        fist_progress, fist_confirmed = self.fist_hold.update(self._last_gestures, now)

        if self.state.state == GameState.MENU and palm_confirmed:
            self._begin_countdown(now)
        elif self.state.state == GameState.GAME_OVER and palm_confirmed:
            self._begin_countdown(now)
        elif self.state.state == GameState.PLAYING and fist_confirmed:
            self.pause_started_at = now
            self.state.transition(GameState.PAUSED, now)
        elif self.state.state == GameState.PAUSED and fist_confirmed:
            self._resume_from_pause(now)

        return palm_progress, fist_progress

    # -- Per-frame gameplay update -----------------------------------------
    def _update_playing(self, now, pose):
        if pose:
            self.left_trail.update(pose["left_wrist"], now)
            self.right_trail.update(pose["right_wrist"], now)

            for trail in (self.left_trail, self.right_trail):
                pos = trail.latest_pos
                if pos is None:
                    continue
                hit_target = self.spawner.check_hit(pos)
                if hit_target is not None:
                    speed = trail.speed_mph()
                    self.audio.play("hit")
                    self.particles.burst(hit_target.x, hit_target.y, hit_target.color)
                    self.score_manager.register_hit(speed, self.difficulty.score_multiplier, now)

        misses = self.spawner.update(now)
        if misses:
            self.audio.play("miss")
            self.score_manager.register_miss(now)
        self.score_manager.maybe_decay_combo(now)

        self.particles.update_and_prune()

        if self._time_remaining(now) <= 0:
            self._end_round(now)

    # -- Main loop ----------------------------------------------------------
    def run(self):
        print("[SYSTEM] Project Kinesis: NEXT online. Show an open palm to begin. 'q' to quit.")
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

        try:
            while self._running:
                frame_start = time.time()
                now = frame_start

                raw = self.camera.read()
                pose = self.pose_tracker.process(raw) if raw is not None else None
                self._last_gestures = self.gesture_recognizer.process(raw) if raw is not None else []

                palm_progress, fist_progress = self._handle_gestures(now)

                if self.state.state == GameState.COUNTDOWN:
                    elapsed = now - self.countdown_start_time
                    remaining = config.COUNTDOWN_SECONDS - elapsed
                    if remaining <= 0:
                        self._begin_round(now)
                elif self.state.state == GameState.PLAYING:
                    self._update_playing(now, pose)
                    self.particles.update_and_prune()

                ctx = {
                    "trails": [self.left_trail, self.right_trail],
                    "spawner": self.spawner,
                    "particles": self.particles,
                    "score_manager": self.score_manager,
                    "difficulty": self.difficulty,
                    "time_remaining": self._time_remaining(now),
                    "show_impact": (now - self.score_manager.last_hit_at) < 1.5,
                    "palm_progress": palm_progress,
                    "fist_progress": fist_progress,
                    "countdown_remaining": (config.COUNTDOWN_SECONDS - (now - self.countdown_start_time))
                                            if self.countdown_start_time else 0,
                    "leaderboard": self.score_manager.load_leaderboard(),
                }

                display_frame = self.renderer.render(raw, self.state.state, ctx)
                cv2.imshow(config.WINDOW_NAME, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_key(key, now)

                elapsed = time.time() - frame_start
                time.sleep(max(0, (1.0 / config.TARGET_FPS) - elapsed))
        finally:
            self.shutdown()

    def shutdown(self):
        self.camera.stop()
        self.pose_tracker.close()
        self.gesture_recognizer.close()
        self.audio.shutdown()
        cv2.destroyAllWindows()
