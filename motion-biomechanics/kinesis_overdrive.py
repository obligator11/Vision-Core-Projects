"""
SHADOWSTRIKE ULTRA — Cinematic AI Boxing Experience
Smooth 60 FPS build — all overlay copies eliminated, pose threaded
"""

import cv2
import numpy as np
import pygame
import mediapipe as mp
import math
import time
import random
import threading
import wave
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class AudioEngine:
    SAMPLE_RATE = 44100
    SOUNDS: dict = {}

    @classmethod
    def _write_wav(cls, path, samples):
        samples = np.clip(samples, -1.0, 1.0)
        int16 = (samples * 32767).astype(np.int16)
        with wave.open(path, 'w') as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(cls.SAMPLE_RATE); wf.writeframes(int16.tobytes())

    @classmethod
    def _envelope(cls, sig, attack=0.005, decay=0.1, sustain=0.3, release=0.15):
        n = len(sig); env = np.zeros(n)
        a = int(attack*cls.SAMPLE_RATE); d = int(decay*cls.SAMPLE_RATE)
        r = int(release*cls.SAMPLE_RATE); s_len = max(0, n-a-d-r)
        if a: env[:a] = np.linspace(0,1,a)
        if d: env[a:a+d] = np.linspace(1,sustain,d)
        if s_len: env[a+d:a+d+s_len] = sustain
        if r: env[-r:] = np.linspace(sustain,0,r)
        return sig * env

    @classmethod
    def _noise(cls, duration, amp=0.5):
        return amp*(np.random.rand(int(cls.SAMPLE_RATE*duration))*2-1)

    @classmethod
    def _low_pass(cls, sig, cutoff=800):
        alpha = cutoff/cls.SAMPLE_RATE; out = np.zeros_like(sig); prev = 0.0
        for i,x in enumerate(sig): out[i]=prev+alpha*(x-prev); prev=out[i]
        return out

    @classmethod
    def generate_all(cls, folder="."):
        sounds = {}
        SR = cls.SAMPLE_RATE

        # whoosh
        dur=0.18; t=np.linspace(0,dur,int(SR*dur))
        freq=np.linspace(600,120,len(t))
        sig=cls._envelope(0.6*np.sin(2*np.pi*freq*t)+cls._noise(dur,0.25)[:len(t)],
                          attack=0.002,decay=0.08,sustain=0.1,release=0.09)
        p=os.path.join(folder,"whoosh.wav"); cls._write_wav(p,sig); sounds["whoosh"]=p

        # light hit
        dur=0.22; t=np.linspace(0,dur,int(SR*dur))
        freq=np.linspace(300,80,len(t))
        sig=cls._envelope(0.7*np.sin(2*np.pi*freq*t)+cls._low_pass(cls._noise(dur,0.5),1200)[:len(t)],
                          attack=0.001,decay=0.06,sustain=0.2,release=0.12)
        p=os.path.join(folder,"light_hit.wav"); cls._write_wav(p,sig); sounds["light_hit"]=p

        # heavy hit
        dur=0.35; t=np.linspace(0,dur,int(SR*dur))
        freq=np.linspace(140,40,len(t))
        body=0.8*np.sin(2*np.pi*freq*t); sub=0.6*np.sin(2*np.pi*(freq*0.5)*t)
        sig=cls._envelope(body+sub+cls._low_pass(cls._noise(dur,0.6),600)[:len(body)],
                          attack=0.001,decay=0.12,sustain=0.25,release=0.20)
        p=os.path.join(folder,"heavy_hit.wav"); cls._write_wav(p,sig); sounds["heavy_hit"]=p

        # power hit
        dur=0.45; t=np.linspace(0,dur,int(SR*dur))
        freq=np.linspace(90,25,len(t))
        body=0.9*np.sin(2*np.pi*freq*t); sub=0.7*np.sin(2*np.pi*(freq*0.5)*t)
        sig=cls._envelope(body+sub+cls._low_pass(cls._noise(dur,0.8),400)[:len(body)],
                          attack=0.001,decay=0.15,sustain=0.3,release=0.25)
        p=os.path.join(folder,"power_hit.wav"); cls._write_wav(p,sig); sounds["power_hit"]=p

        # combo up
        dur=0.30; sig=np.zeros(int(SR*dur)); step=int(SR*dur/3)
        for i,f in enumerate([523,659,784]):
            t_s=np.linspace(0,step/SR,step)
            tone=cls._envelope(0.6*np.sin(2*np.pi*f*t_s),attack=0.005,decay=0.04,sustain=0.4,release=0.05)
            s=i*step; sig[s:s+step]+=tone
        sig=cls._envelope(sig,attack=0.01,decay=0.1,sustain=0.5,release=0.1)
        p=os.path.join(folder,"combo_up.wav"); cls._write_wav(p,sig); sounds["combo_up"]=p

        return sounds

    def __init__(self, folder="."):
        pygame.mixer.pre_init(self.SAMPLE_RATE,-16,1,512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)
        for name,path in self.generate_all(folder).items():
            try: self.SOUNDS[name]=pygame.mixer.Sound(path)
            except Exception as e: print(f"[Audio] {name}: {e}")

    def play(self, name, volume=1.0):
        snd=self.SOUNDS.get(name)
        if not snd: return
        snd.set_volume(max(0.0,min(1.0,volume+random.uniform(-0.04,0.04))))
        snd.play()


# ─────────────────────────────────────────────────────────────────────────────
# PARTICLE SYSTEM  (no per-particle frame.copy — single overlay pass)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Particle:
    x: float; y: float
    vx: float; vy: float
    life: float
    color: Tuple[int,int,int]
    size: float
    kind: str = "spark"   # spark | smoke


class EffectsEngine:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.particles: List[Particle] = []
        self.shockwaves = []          # [cx,cy,r,max_r,life,color]
        self.screen_shake = [0.0, 0.0]
        self.shake_offset = (0, 0)
        self.slow_mo_until = 0.0
        self.zoom_factor = 1.0
        self.zoom_cx = W//2
        self.zoom_cy = H//2
        self.flash_color = None
        self.flash_until = 0.0
        self.trails: deque = deque(maxlen=24)

        # ── Pre-bake static overlays so we never allocate per frame ──────────
        self._flash_buf   = np.zeros((H, W, 3), dtype=np.uint8)
        # Vignette mask (float32, pre-computed once)
        Y = np.linspace(-1,1,H)[:,None]
        X = np.linspace(-1,1,W)[None,:]
        self._vignette = np.clip(1.0-(X**2+Y**2)*0.6,0,1).astype(np.float32)[:,:,None]
        # Scanline mask
        self._scanline = np.ones((H,W,3), dtype=np.float32)
        self._scanline[::4,:] = 0.92   # dim every 4th row slightly
        # Working overlay buffer reused every frame
        self._overlay = np.empty((H, W, 3), dtype=np.uint8)

    # ── triggers ─────────────────────────────────────────────────────────────
    def trigger_hit(self, cx, cy, speed, power=False, direction=(0,-1)):
        count   = 50 if power else 22
        palette = [(255,60,0),(255,160,0),(255,220,80)] if power else \
                  [(0,200,255),(80,140,255),(255,255,255)]
        for _ in range(count):
            ang=random.uniform(0,math.tau); spd=random.uniform(3,12)*(1.5 if power else 1.0)
            col=random.choice(palette)
            self.particles.append(Particle(
                x=cx,y=cy,
                vx=math.cos(ang)*spd + direction[0]*random.uniform(2,5),
                vy=math.sin(ang)*spd + direction[1]*random.uniform(2,5),
                life=1.0, color=col,
                size=random.uniform(2,6) if power else random.uniform(1,3.5),
                kind="spark"
            ))
        ring_col=(255,80,0) if power else (0,220,255)
        self.shockwaves.append([cx,cy,0,110 if power else 65,1.0,ring_col])
        for _ in range(6 if power else 3):
            self.particles.append(Particle(
                x=cx+random.uniform(-12,12), y=cy+random.uniform(-12,12),
                vx=random.uniform(-0.8,0.8), vy=random.uniform(-1.5,0),
                life=1.0, color=(160,160,160),
                size=random.uniform(5,13), kind="smoke"
            ))

    def trigger_screen_shake(self, magnitude, duration=0.25):
        if magnitude > self.screen_shake[0]:
            self.screen_shake = [magnitude, duration]

    def trigger_slow_mo(self, duration=0.35):
        self.slow_mo_until = time.time()+duration
        self.zoom_factor = 1.12

    def trigger_flash(self, color=(255,30,30), duration=0.12):
        self.flash_color = color; self.flash_until = time.time()+duration

    def add_trail(self, x, y, color):
        self.trails.append((x, y, time.time(), color))

    # ── update (no allocations) ───────────────────────────────────────────────
    def update(self, dt):
        now = time.time()

        # shake
        if self.screen_shake[1] > 0:
            self.screen_shake[1] -= dt
            mag = max(1, int(self.screen_shake[0]*(self.screen_shake[1]/0.25)))
            self.shake_offset = (random.randint(-mag,mag), random.randint(-mag,mag))
        else:
            self.shake_offset = (0,0); self.screen_shake[0]=0

        # zoom decay
        if now > self.slow_mo_until and self.zoom_factor > 1.0:
            self.zoom_factor = max(1.0, self.zoom_factor - dt*3.0)

        # particles — update in-place, filter once
        alive = []
        for p in self.particles:
            p.life -= dt*(2.2 if p.kind=="smoke" else 3.2)
            if p.life <= 0: continue
            p.x+=p.vx; p.y+=p.vy
            if p.kind=="spark": p.vy+=0.35; p.vx*=0.93
            else:               p.size*=1.025
            alive.append(p)
        self.particles = alive

        # shockwaves
        self.shockwaves = [
            [cx,cy,r+mr*dt*4.5,mr,life-dt*4.0,col]
            for cx,cy,r,mr,life,col in self.shockwaves
            if life-dt*4.0 > 0 and r < mr
        ]

    # ── draw  (minimal copies — one shared overlay buffer) ───────────────────
    def draw(self, frame):
        now = time.time()
        ov  = self._overlay   # reuse this buffer — never allocate

        # ── trails (batch: draw all to one overlay, blend once) ──────────────
        trail_pts = [(x,y,t,c) for x,y,t,c in self.trails if now-t < 0.35]
        if trail_pts:
            np.copyto(ov, frame)
            for x,y,t,c in trail_pts:
                age   = now-t
                alpha = max(0.0, 1.0-age/0.35)
                r     = max(1, int(7*alpha))
                glow  = (min(255,int(c[0]*1.4)), min(255,int(c[1]*1.4)), min(255,int(c[2]*1.4)))
                cv2.circle(ov,(int(x),int(y)),r+3,glow,-1)
                cv2.circle(ov,(int(x),int(y)),r,c,-1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

        # ── sparks — draw directly (no copy needed, just colored circles) ────
        for p in self.particles:
            if p.kind != "spark": continue
            a   = max(0.0, p.life)
            col = (int(p.color[0]*a), int(p.color[1]*a), int(p.color[2]*a))
            sz  = max(1, int(p.size*a))
            cx_, cy_ = int(p.x), int(p.y)
            glow= (min(255,int(p.color[0]*1.4)), min(255,int(p.color[1]*1.4)), min(255,int(p.color[2]*1.4)))
            cv2.circle(frame,(cx_,cy_),sz+2,glow,-1)
            cv2.circle(frame,(cx_,cy_),sz,col,-1)

        # ── smoke — batch all smoke into one overlay pass ────────────────────
        smoke = [p for p in self.particles if p.kind=="smoke"]
        if smoke:
            np.copyto(ov, frame)
            for p in smoke:
                a = max(0.0, p.life)*0.28
                if a < 0.01: continue
                col=(int(p.color[0]*a),int(p.color[1]*a),int(p.color[2]*a))
                cv2.circle(ov,(int(p.x),int(p.y)),max(1,int(p.size)),
                           (p.color[0],p.color[1],p.color[2]),-1)
            cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)

        # ── shockwaves — batch into one overlay ──────────────────────────────
        if self.shockwaves:
            np.copyto(ov, frame)
            for cx,cy,r,mr,life,col in self.shockwaves:
                a = max(0.0,life)*0.65
                cv2.circle(ov,(int(cx),int(cy)),max(1,int(r)),col,max(1,int(3*life)))
            cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        # ── flash — single blend using pre-allocated buffer ───────────────────
        if self.flash_color and now < self.flash_until:
            rem   = self.flash_until-now
            alpha = min(0.52,(rem/0.12)*0.52)
            self._flash_buf[:] = (self.flash_color[2], self.flash_color[1], self.flash_color[0])
            cv2.addWeighted(self._flash_buf, alpha, frame, 1-alpha, 0, frame)

    def apply_vignette(self, frame):
        """Pre-baked vignette — one multiply, no allocation."""
        np.multiply(frame, self._vignette, out=frame.astype(np.float32)).clip(0,255).astype(np.uint8, copy=False)
        # In-place friendly version:
        tmp = frame.astype(np.float32)
        np.multiply(tmp, self._vignette, out=tmp)
        np.clip(tmp, 0, 255, out=tmp)
        frame[:] = tmp.astype(np.uint8)

    def apply_scanlines(self, frame):
        """Pre-baked scanline — one multiply."""
        tmp = frame.astype(np.float32)
        np.multiply(tmp, self._scanline, out=tmp)
        frame[:] = tmp.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER TRACKING  (pose runs on its own thread)
# ─────────────────────────────────────────────────────────────────────────────
class PlayerTracking:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
            model_complexity=1
        )
        self.wrist_history = {"L": deque(maxlen=6), "R": deque(maxlen=6)}
        self.speeds        = {"L": 0.0, "R": 0.0}
        self._landmarks    = None
        self._wrists       = {"L": None, "R": None}

        # Thread state
        self._lock         = threading.Lock()
        self._pending_rgb  = None
        self._result_ready = threading.Event()
        self._running      = True
        self._thread       = threading.Thread(target=self._pose_loop, daemon=True)
        self._thread.start()

    def _pose_loop(self):
        while self._running:
            self._result_ready.wait(timeout=0.05)
            self._result_ready.clear()
            with self._lock:
                rgb = self._pending_rgb
            if rgb is None:
                continue
            results = self.pose.process(rgb)
            wrists  = {"L": None, "R": None}
            lmarks  = None
            if results.pose_landmarks:
                lmarks = results.pose_landmarks
                lm = lmarks.landmark
                h, w = rgb.shape[:2]
                L = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
                R = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                wrists["L"] = (int(L.x*w), int(L.y*h))
                wrists["R"] = (int(R.x*w), int(R.y*h))
            with self._lock:
                self._wrists    = wrists
                self._landmarks = lmarks
                # update speed history
                for side, pos in wrists.items():
                    if pos:
                        hist = self.wrist_history[side]
                        if hist:
                            dx = pos[0]-hist[-1][0]; dy = pos[1]-hist[-1][1]
                            self.speeds[side] = math.hypot(dx,dy)
                        hist.append(pos)

    def submit(self, rgb_frame):
        """Push a new frame for pose processing (non-blocking)."""
        with self._lock:
            self._pending_rgb = rgb_frame
        self._result_ready.set()

    def get_state(self):
        with self._lock:
            return dict(self._wrists), self._landmarks, dict(self.speeds)

    def get_punch_direction(self, side):
        with self._lock:
            hist = self.wrist_history[side]
            if len(hist) < 2: return (0,-1)
            dx = hist[-1][0]-hist[-2][0]; dy = hist[-1][1]-hist[-2][1]
            mag = max(1, math.hypot(dx,dy))
            return (dx/mag, dy/mag)

    def draw_skeleton(self, frame, lmarks, color=(0,255,150)):
        if not lmarks: return
        h,w = frame.shape[:2]
        lm  = lmarks.landmark
        CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),
                (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]
        for a,b in CONN:
            pa=lm[a]; pb=lm[b]
            if pa.visibility>0.4 and pb.visibility>0.4:
                p1=(int(pa.x*w),int(pa.y*h)); p2=(int(pb.x*w),int(pb.y*h))
                cv2.line(frame,p1,p2,(0,80,60),4)
                cv2.line(frame,p1,p2,color,2)
        for idx in [15,16,11,12,23,24]:
            p=lm[idx]
            if p.visibility>0.4:
                cx_,cy_=int(p.x*w),int(p.y*h)
                cv2.circle(frame,(cx_,cy_),6,(255,255,255),-1)
                cv2.circle(frame,(cx_,cy_),4,color,-1)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# ENEMY AI
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Enemy:
    cx: float; cy: float; radius: int
    hp: int; max_hp: int
    phase: float=0.0; speed: float=1.5; direction: int=1
    alive: bool=True; hit_flash: float=0.0
    bob_amp: float=12.0; bob_freq: float=1.8
    kind: str="head"; score_value: int=10
    cy_draw: float=0.0

class EnemyAI:
    COLORS = {"head":(0,200,255),"body":(255,60,200)}

    def __init__(self,W,H):
        self.fw=W; self.fh=H
        self.enemies: List[Enemy]=[]
        self.spawn_timer=0.0; self.wave=1; self.total_killed=0

    def reset(self):
        self.enemies.clear(); self.spawn_timer=0.0; self.wave=1; self.total_killed=0

    def spawn_wave(self):
        count=min(2+self.wave,5)
        for _ in range(count):
            kind=random.choice(["head","body"])
            r=random.randint(28,44) if kind=="head" else random.randint(22,32)
            cx=random.randint(self.fw//4, 3*self.fw//4)
            cy=random.randint(self.fh//5, self.fh//2)
            hp=1+(self.wave//2)
            self.enemies.append(Enemy(
                cx=cx,cy=cy,radius=r,hp=hp,max_hp=hp,
                phase=random.uniform(0,math.tau),
                speed=random.uniform(0.8,1.4+self.wave*0.12),
                direction=random.choice([-1,1]),kind=kind,
                bob_amp=random.uniform(8,18),bob_freq=random.uniform(1.5,2.5),
                score_value=(15 if kind=="head" else 10)*self.wave,cy_draw=cy
            ))

    def update(self,dt):
        self.spawn_timer-=dt
        if self.spawn_timer<=0 and not self.enemies:
            self.spawn_wave(); self.wave+=1; self.spawn_timer=2.0
        for e in self.enemies:
            if not e.alive: continue
            e.phase+=dt*e.bob_freq
            e.cx+=e.speed*e.direction
            if e.cx<e.radius+40 or e.cx>self.fw-e.radius-40: e.direction*=-1
            e.cy_draw=e.cy+math.sin(e.phase)*e.bob_amp
            if e.hit_flash>0: e.hit_flash=max(0.0,e.hit_flash-dt*7)
        self.enemies=[e for e in self.enemies if e.alive]

    def check_hit(self,px,py,speed):
        for e in self.enemies:
            if not e.alive: continue
            if math.hypot(px-e.cx, py-e.cy_draw) < e.radius*1.4:
                return e
        return None

    def damage_enemy(self,e,speed):
        e.hp-=1+(1 if speed>35 else 0); e.hit_flash=1.0
        if e.hp<=0: e.alive=False; self.total_killed+=1; return True,e.score_value
        return False,max(3,e.score_value//3)

    def draw(self,frame):
        for e in self.enemies:
            if not e.alive: continue
            cx,cy=int(e.cx),int(e.cy_draw)
            base=self.COLORS[e.kind]
            col=tuple(int(base[i]*(1-e.hit_flash)+(255 if i==2 else 50)*e.hit_flash) for i in range(3)) \
                if e.hit_flash>0 else base
            # glow rings (direct draw — no copy)
            for gr,a in [(e.radius+18,0.06),(e.radius+8,0.13),(e.radius+3,0.28)]:
                ov=frame.copy()
                cv2.circle(ov,(cx,cy),gr,col,-1)
                cv2.addWeighted(ov,a,frame,1-a,0,frame)
            cv2.circle(frame,(cx,cy),e.radius,col,-1)
            if e.kind=="head":
                er=max(2,e.radius//5)
                for ex_ in [-e.radius//3,e.radius//3]:
                    cv2.circle(frame,(cx+ex_,cy-e.radius//5),er,(0,0,0),-1)
                    cv2.circle(frame,(cx+ex_+1,cy-e.radius//5-1),max(1,er//2),(255,255,255),-1)
            else:
                cv2.line(frame,(cx-e.radius//2,cy),(cx+e.radius//2,cy),(0,0,0),2)
                cv2.line(frame,(cx,cy-e.radius//2),(cx,cy+e.radius//2),(0,0,0),2)
                cv2.circle(frame,(cx,cy),e.radius//3,(0,0,0),2)
            if e.max_hp>1:
                bx=cx-e.radius; by=cy-e.radius-14; bw=e.radius*2
                cv2.rectangle(frame,(bx,by),(bx+bw,by+5),(40,40,40),-1)
                cv2.rectangle(frame,(bx,by),(bx+int(bw*e.hp/e.max_hp),by+5),(0,255,100),-1)
            if e.max_hp>1 and e.hp==1:
                pulse=int(abs(math.sin(time.time()*6))*4)
                cv2.circle(frame,(cx,cy),e.radius+pulse,(0,0,255),2)


# ─────────────────────────────────────────────────────────────────────────────
# COMBO SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
class ComboSystem:
    WINDOW=2.2
    def __init__(self):
        self.count=0; self.last_hit=0.0; self.multiplier=1.0
        self.display_text=""; self.display_until=0.0; self.peak=0

    def hit(self,audio):
        now=time.time()
        if now-self.last_hit>self.WINDOW: self.count=0
        self.count+=1; self.last_hit=now; self.peak=max(self.peak,self.count)
        self.multiplier=1.0+0.25*math.log(self.count,1.5) if self.count>=2 else 1.0
        if self.count>=3:
            self.display_text=f"COMBO \xd7{self.count}"; self.display_until=now+1.2
            if self.count%3==0: audio.play("combo_up",0.9)
        return self.multiplier

    def update(self):
        if time.time()-self.last_hit>self.WINDOW:
            self.count=0; self.multiplier=1.0

    def draw(self,frame):
        now=time.time()
        if not self.display_text or now>self.display_until: return
        h,w=frame.shape[:2]
        rem=self.display_until-now; alpha=min(1.0,rem*4)
        fs=1.1+0.04*self.count; thick=max(2,2+self.count//4)
        font=cv2.FONT_HERSHEY_DUPLEX
        (tw,_),_=cv2.getTextSize(self.display_text,font,fs,thick)
        tx=w//2-tw//2; ty=h//2+55
        col=(255,120,0) if self.count>=7 else (0,220,255)
        for gr in [14,6]:
            ov=frame.copy()
            cv2.putText(ov,self.display_text,(tx,ty),font,fs,col,thick+gr,cv2.LINE_AA)
            cv2.addWeighted(ov,0.1*alpha,frame,1-0.1*alpha,0,frame)
        cv2.putText(frame,self.display_text,(tx,ty),font,fs,(255,255,255),thick,cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────
class HUD:
    def __init__(self,W,H):
        self.W=W; self.H=H; self.score=0; self.disp_score=0.0
        self.current_speed=0.0; self.wave=1; self.kills=0

    def update(self,dt):
        self.disp_score+=(self.score-self.disp_score)*min(1.0,dt*10)

    def _glow(self,frame,text,pos,font,scale,color,thick,gcol=None):
        if gcol:
            ov=frame.copy()
            cv2.putText(ov,text,pos,font,scale,gcol,thick+5,cv2.LINE_AA)
            cv2.addWeighted(ov,0.18,frame,0.82,0,frame)
        cv2.putText(frame,text,pos,font,scale,color,thick,cv2.LINE_AA)

    def draw(self,frame,combo):
        w,h=self.W,self.H
        font=cv2.FONT_HERSHEY_DUPLEX; font2=cv2.FONT_HERSHEY_SIMPLEX
        # dark panels
        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(280,112),(0,0,0),-1)
        cv2.rectangle(ov,(w-252,0),(w,92),(0,0,0),-1)
        cv2.addWeighted(ov,0.52,frame,0.48,0,frame)
        # score
        self._glow(frame,f"{int(self.disp_score):,}",(18,48),font,1.55,(0,255,180),2,(0,130,90))
        self._glow(frame,"SCORE",(18,22),font2,0.58,(100,255,200),1)
        self._glow(frame,f"WAVE {self.wave}",(18,78),font2,0.62,(255,200,0),1)
        self._glow(frame,f"KO  {self.kills}",(18,100),font2,0.52,(180,180,180),1)
        # speed bar
        bx=w-240; by=18; bw=220; bh=18
        ratio=min(1.0,self.current_speed/60.0)
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(30,30,30),-1)
        fw_=int(bw*ratio)
        for i in range(fw_):
            t_=i/max(1,bw); r_=min(255,int(t_*510)); g_=min(255,int((1-t_)*510))
            cv2.line(frame,(bx+i,by),(bx+i,by+bh),(0,g_,r_),1)
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(0,200,255),1)
        self._glow(frame,"SPEED",(bx,by-4),font2,0.48,(0,200,255),1)
        if combo.multiplier>1.0:
            mc=f"\xd7{combo.multiplier:.1f}"
            col=(255,120,0) if combo.count>=7 else (0,220,255)
            self._glow(frame,mc,(w-200,65),font,1.05,col,2,col)
        # accent lines
        acc=(0,200,255)
        cv2.line(frame,(0,115),(280,115),acc,1)
        cv2.line(frame,(280,0),(280,115),acc,1)
        cv2.line(frame,(w-252,95),(w,95),acc,1)
        cv2.line(frame,(w-252,0),(w-252,95),acc,1)


# ─────────────────────────────────────────────────────────────────────────────
# INTRO
# ─────────────────────────────────────────────────────────────────────────────
class IntroSequence:
    def __init__(self,duration=2.8):
        self.start=time.time(); self.duration=duration; self.done=False

    def draw(self,frame):
        now=time.time(); elapsed=now-self.start
        if elapsed>self.duration: self.done=True; return False
        h,w=frame.shape[:2]; t=elapsed/self.duration
        if t<0.15:
            a=1.0-t/0.15
            ov=np.full_like(frame,(255,255,255))
            cv2.addWeighted(ov,a*0.8,frame,1-a*0.8,0,frame)
        # grid
        ga=max(0,0.22*(1-t))
        if ga>0.01:
            gf=frame.copy()
            for x in range(0,w,40): cv2.line(gf,(x,0),(x,h),(0,80,60),1)
            for y_ in range(0,h,40): cv2.line(gf,(0,y_),(w,y_),(0,80,60),1)
            cv2.addWeighted(gf,ga,frame,1-ga,0,frame)
        font=cv2.FONT_HERSHEY_DUPLEX; title="AI BOXING CHALLENGE"
        fs=1.8
        (tw,_),_=cv2.getTextSize(title,font,fs,3)
        tx=w//2-tw//2; ty=h//2-20
        gx=random.randint(-5,5) if t<0.5 else 0
        off=max(0,int((1-t)*7))
        cv2.putText(frame,title,(tx-off+gx,ty),font,fs,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(frame,title,(tx+gx,ty),font,fs,(0,255,0),3,cv2.LINE_AA)
        cv2.putText(frame,title,(tx+off+gx,ty),font,fs,(255,0,0),3,cv2.LINE_AA)
        a_t=min(1.0,t*4)
        ov2=frame.copy(); cv2.putText(ov2,title,(tx,ty),font,fs,(255,255,255),2,cv2.LINE_AA)
        cv2.addWeighted(ov2,a_t,frame,1-a_t,0,frame)
        if t>0.5:
            sub="PUNCH THE TARGETS — COMBO FOR MULTIPLIER"; fs2=0.6
            (sw,_),_=cv2.getTextSize(sub,font,fs2,1); a_s=min(1.0,(t-0.5)*4)
            ov3=frame.copy(); cv2.putText(ov3,sub,(w//2-sw//2,ty+55),font,fs2,(0,220,255),1,cv2.LINE_AA)
            cv2.addWeighted(ov3,a_s,frame,1-a_s,0,frame)
        beat=math.sin(elapsed*math.pi*4)
        if beat>0.95:
            fl=frame.copy(); cv2.rectangle(fl,(0,0),(w,h),(0,60,40),-1)
            cv2.addWeighted(fl,0.18,frame,0.82,0,frame)
        cv2.rectangle(frame,(0,h-4),(int(w*(1-t)),h),(0,200,255),-1)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GAME
# ─────────────────────────────────────────────────────────────────────────────
class ShadowStrikeUltra:
    LIGHT_HIT_SPD=14; HEAVY_HIT_SPD=28; POWER_HIT_SPD=45; WHOOSH_SPD=18
    SLOW_MO_FACTOR=0.28

    def __init__(self):
        self.W,self.H=1280,720
        self.cap=cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.H)
        self.cap.set(cv2.CAP_PROP_FPS,60)
        # Use MJPG for higher FPS from webcam
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        pygame.init()
        self.audio   = AudioEngine()
        self.tracker = PlayerTracking()
        self.enemies = EnemyAI(self.W,self.H)
        self.effects = EffectsEngine(self.W,self.H)
        self.combo   = ComboSystem()
        self.hud     = HUD(self.W,self.H)
        self.intro   = IntroSequence()

        self.score=0; self.last_time=time.time()
        self.whoosh_cd=0.0; self.power_text_until=0.0
        self.hit_label=None   # (text, until, x, y)

        # Hit cooldown per side to avoid multi-trigger on same punch
        self.hit_cd={"L":0.0,"R":0.0}

        # Threaded capture
        self._frame_lock=threading.Lock()
        self._latest=None; self._running=True
        self._cap_thread=threading.Thread(target=self._capture_loop,daemon=True)
        self._cap_thread.start()

        self.fps_deque=deque(maxlen=60)

        # Pre-allocate zoom buffer
        self._zoom_buf=np.empty((self.H,self.W,3),dtype=np.uint8)

    def _capture_loop(self):
        while self._running:
            ret,frame=self.cap.read()
            if ret:
                frame=cv2.flip(frame,1)
                with self._frame_lock:
                    self._latest=frame

    def _get_frame(self):
        with self._frame_lock:
            return self._latest.copy() if self._latest is not None else None

    def _apply_zoom(self, frame, factor, cx, cy):
        """Fast zoom — crop + INTER_LINEAR resize, in-place into pre-alloc buf."""
        if factor<=1.001: return frame
        h,w=frame.shape[:2]
        nw=int(w/factor); nh=int(h/factor)
        x1=max(0,min(w-nw,cx-nw//2)); y1=max(0,min(h-nh,cy-nh//2))
        cropped=frame[y1:y1+nh, x1:x1+nw]
        cv2.resize(cropped,(w,h),dst=self._zoom_buf,interpolation=cv2.INTER_LINEAR)
        return self._zoom_buf

    def _apply_shake(self, frame, sx, sy):
        """Numpy roll — no warpAffine allocation."""
        if sx==0 and sy==0: return frame
        return np.roll(np.roll(frame, sy, axis=0), sx, axis=1)

    def _draw_power_text(self,frame):
        if time.time()>self.power_text_until: return
        h,w=frame.shape[:2]; rem=self.power_text_until-time.time()
        alpha=min(1.0,rem*5); font=cv2.FONT_HERSHEY_DUPLEX; text="POWER HIT!"
        fs=2.0; (tw,_),_=cv2.getTextSize(text,font,fs,3)
        tx=w//2-tw//2; ty=h//3
        ov=frame.copy()
        for gr in [14,6]:
            cv2.putText(ov,text,(tx,ty),font,fs,(0,0,255),3+gr,cv2.LINE_AA)
        cv2.addWeighted(ov,0.13*alpha,frame,1-0.13*alpha,0,frame)
        cv2.putText(frame,text,(tx,ty),font,fs,(255,60,60),3,cv2.LINE_AA)
        cv2.putText(frame,text,(tx,ty),font,fs,(255,200,200),1,cv2.LINE_AA)

    def _draw_hit_label(self,frame):
        if not self.hit_label: return
        text,until,lx,ly=self.hit_label
        now=time.time()
        if now>until: self.hit_label=None; return
        rem=until-now; alpha=min(1.0,rem*5)
        ly_=ly-int((1.0-rem/0.7)*28)
        ov=frame.copy()
        cv2.putText(ov,text,(lx,ly_),cv2.FONT_HERSHEY_DUPLEX,0.85,(255,255,100),2,cv2.LINE_AA)
        cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)

    def _draw_fps(self,frame):
        if len(self.fps_deque)<2: return
        fps=len(self.fps_deque)/(self.fps_deque[-1]-self.fps_deque[0]+1e-9)
        col=(0,255,0) if fps>=50 else (0,200,255) if fps>=30 else (0,60,255)
        cv2.putText(frame,f"FPS {fps:.0f}",(self.W-100,self.H-12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)

    def run(self):
        cv2.namedWindow("ShadowStrike ULTRA",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ShadowStrike ULTRA",self.W,self.H)

        # Submit first frame immediately
        first=self._get_frame()
        if first is not None:
            self.tracker.submit(cv2.cvtColor(first,cv2.COLOR_BGR2RGB))

        while self._running:
            now=time.time()
            frame=self._get_frame()
            if frame is None:
                if cv2.waitKey(1)&0xFF==ord('q'): break
                continue

            dt_raw=now-self.last_time; self.last_time=now
            self.fps_deque.append(now)

            # Submit to pose thread (non-blocking)
            self.tracker.submit(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

            # Slow-mo dt
            in_slow=(time.time()<self.effects.slow_mo_until)
            dt=dt_raw*self.SLOW_MO_FACTOR if in_slow else dt_raw

            # ── intro ────────────────────────────────────────────────────────
            if not self.intro.done:
                self.intro.draw(frame)
                self.effects.apply_scanlines(frame)
                cv2.imshow("ShadowStrike ULTRA",frame)
                if cv2.waitKey(1)&0xFF==ord('q'): break
                continue

            # ── get pose results (latest, non-blocking) ───────────────────────
            wrists,lmarks,speeds=self.tracker.get_state()

            # ── dim + cyberpunk tint ─────────────────────────────────────────
            frame=cv2.convertScaleAbs(frame,alpha=0.83,beta=-8)

            # ── skeleton ─────────────────────────────────────────────────────
            self.tracker.draw_skeleton(frame,lmarks)

            # ── enemy update ──────────────────────────────────────────────────
            self.enemies.update(dt)
            self.hud.wave=self.enemies.wave; self.hud.kills=self.enemies.total_killed

            # ── hit detection ─────────────────────────────────────────────────
            for side in ("L","R"):
                pos=wrists.get(side)
                if not pos: continue
                spd=speeds.get(side,0.0)
                self.hud.current_speed=max(self.hud.current_speed*0.88,spd)

                # Trail
                if spd>8:
                    tc=(255,80,0) if spd>self.POWER_HIT_SPD else \
                       (0,180,255) if spd>self.HEAVY_HIT_SPD else (0,255,150)
                    self.effects.add_trail(pos[0],pos[1],tc)

                # Whoosh
                if spd>self.WHOOSH_SPD and now>self.whoosh_cd:
                    self.audio.play("whoosh",min(0.7,0.3+(spd-self.WHOOSH_SPD)/40))
                    self.whoosh_cd=now+0.15

                # Hit — with cooldown so one punch doesn't trigger 10 frames
                if spd>self.LIGHT_HIT_SPD and now>self.hit_cd[side]:
                    hit_e=self.enemies.check_hit(pos[0],pos[1],spd)
                    if hit_e:
                        self.hit_cd[side]=now+0.25   # 250ms cooldown per hand
                        direction=self.tracker.get_punch_direction(side)
                        killed,base_sc=self.enemies.damage_enemy(hit_e,spd)
                        ex,ey=int(hit_e.cx),int(hit_e.cy_draw)
                        mult=self.combo.hit(self.audio)
                        earned=int(base_sc*mult)
                        self.score+=earned; self.hud.score=self.score
                        is_power=spd>self.POWER_HIT_SPD
                        self.effects.trigger_hit(ex,ey,spd,power=is_power,direction=direction)
                        if is_power:
                            self.audio.play("power_hit",1.0)
                            self.effects.trigger_screen_shake(18,0.30)
                            self.effects.trigger_slow_mo(0.32)
                            self.effects.trigger_flash((80,0,0),0.13)
                            self.effects.zoom_cx=ex; self.effects.zoom_cy=ey
                            self.power_text_until=now+0.85
                            label=f"+{earned} POWER!"
                        elif spd>self.HEAVY_HIT_SPD:
                            self.audio.play("heavy_hit",0.88)
                            self.effects.trigger_screen_shake(9,0.18)
                            self.effects.trigger_flash((0,20,50),0.07)
                            label=f"+{earned}"
                        else:
                            self.audio.play("light_hit",0.7)
                            self.effects.trigger_screen_shake(3,0.09)
                            label=f"+{earned}"
                        self.hit_label=(label,now+0.7,ex-30,ey-40)
                        if killed:
                            for _ in range(18):
                                a=random.uniform(0,math.tau); s=random.uniform(5,16)
                                col=random.choice([(255,220,0),(255,140,0),(255,60,0)])
                                self.effects.particles.append(Particle(
                                    x=ex,y=ey,vx=math.cos(a)*s,vy=math.sin(a)*s,
                                    life=1.0,color=col,size=random.uniform(3,8),kind="spark"
                                ))

            # ── updates ──────────────────────────────────────────────────────
            self.combo.update()
            self.effects.update(dt_raw)

            # ── zoom ─────────────────────────────────────────────────────────
            if self.effects.zoom_factor>1.001:
                frame=self._apply_zoom(frame,self.effects.zoom_factor,
                                       self.effects.zoom_cx,self.effects.zoom_cy)

            # ── shake ─────────────────────────────────────────────────────────
            sx,sy=self.effects.shake_offset
            if sx or sy:
                frame=self._apply_shake(frame,sx,sy)

            # ── draw ─────────────────────────────────────────────────────────
            self.enemies.draw(frame)
            self.effects.draw(frame)

            # wrist indicators
            for side,pos in wrists.items():
                if not pos: continue
                spd=speeds.get(side,0.0)
                col=(255,60,0) if spd>self.POWER_HIT_SPD else \
                    (0,180,255) if spd>self.HEAVY_HIT_SPD else (0,255,150)
                r=min(20,max(6,int(5+spd*0.28)))
                ov=frame.copy()
                cv2.circle(ov,pos,r+7,col,-1)
                cv2.addWeighted(ov,0.18,frame,0.82,0,frame)
                cv2.circle(frame,pos,r,col,-1)
                cv2.circle(frame,pos,r+2,(255,255,255),1)

            # cinematic pass
            self.effects.apply_scanlines(frame)
            self.effects.apply_vignette(frame)

            # UI
            self.hud.update(dt_raw); self.hud.draw(frame,self.combo)
            self.combo.draw(frame)
            self._draw_power_text(frame)
            self._draw_hit_label(frame)

            # slow-mo label
            if in_slow:
                cv2.putText(frame,"SLOW MOTION",(self.W//2-105,self.H-40),
                            cv2.FONT_HERSHEY_DUPLEX,0.88,(0,220,255),2,cv2.LINE_AA)

            self._draw_fps(frame)
            cv2.imshow("ShadowStrike ULTRA",frame)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            elif key==ord('r'):
                self.score=0; self.hud.score=0
                self.enemies.reset(); self.combo=ComboSystem()
                self.intro=IntroSequence(duration=1.5)

        self._running=False
        self.tracker.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print(f"\n{'='*40}\n  SCORE: {self.score:,}  |  PEAK COMBO: x{self.combo.peak}  |  KO: {self.enemies.total_killed}\n{'='*40}")


if __name__=="__main__":
    print("""
  ╔══════════════════════════════════════════════╗
  ║     SHADOWSTRIKE ULTRA  — Loading...         ║
  ║     Q to quit | R to restart                 ║
  ╚══════════════════════════════════════════════╝
""")
    ShadowStrikeUltra().run()