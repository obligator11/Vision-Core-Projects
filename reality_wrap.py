import cv2
import numpy as np
import mediapipe as mp
import torch
import time
import math
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from PIL import Image

# ==========================================
# PHASE 3 & 4: THE ENGINE & THE WARP (LCM)
# ==========================================
print("[SYS] Initializing Reality Warp Generative Engine...")

model_id = "Lykon/dreamshaper-7"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

pipe = AutoPipelineForImage2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Inject Latent Consistency Model Scheduler for 2-4 step inference
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lcm_lora_id)

# Bypass torch.compile to prevent infinite hanging, use channels_last for speed
pipe.unet.to(memory_format=torch.channels_last)
pipe.set_progress_bar_config(disable=True)
print("[SYS] Generative Engine Online.")

# ==========================================
# PHASE 1 & 2: THE TRIGGER & THE ANCHOR (MediaPipe)
# ==========================================
mp_hands = mp.solutions.hands
mp_selfie = mp.solutions.selfie_segmentation

hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.8
)
# Model 1 is the general landscape model (fastest)
segmentation = mp_selfie.SelfieSegmentation(model_selection=1)

# Variables for tracking
current_bg = None
cooldown_timer = 0
COOLDOWN_FRAMES = 30 # Prevent triggering multiple times a second

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detect_gesture(hand_landmarks):
    """
    Kinematic Triggers calculating normalized vector proximities.
    """
    # Finger tips
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Snap Detection: Thumb and Middle finger are extremely close
    snap_dist = calculate_distance(thumb_tip, middle_tip)
    if snap_dist < 0.05:
        return "SNAP"

    # Open Palm Detection: All tips are far from the wrist
    tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    avg_dist_to_wrist = sum([calculate_distance(t, wrist) for t in tips]) / 5.0
    if avg_dist_to_wrist > 0.4:
        return "PALM"

    return None

def generate_background(frame, prompt):
    """
    Passes the segmented background to the LCM pipeline at a safe resolution.
    """
    # Remove '8k' from prompt just to be safe
    prompt = prompt.replace("8k", "1080p")
    print(f"[WARP] Activating Style: {prompt}")
    
    # CRITICAL FIX: Shrink the frame to 512x512 so your GPU doesn't crash
    safe_size_frame = cv2.resize(frame, (512, 512))
    pil_img = Image.fromarray(cv2.cvtColor(safe_size_frame, cv2.COLOR_BGR2RGB))
    
    start_time = time.time()
    result = pipe(
        prompt=prompt,
        image=pil_img,
        num_inference_steps=3,       
        guidance_scale=1.5,          
        strength=0.8                 
    ).images[0]
    
    gen_time = time.time() - start_time
    print(f"[WARP] Generation Complete in {gen_time:.3f}s")
    
    # Resize the AI image back up to match your original webcam size
    result_cv2 = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    return cv2.resize(result_cv2, (frame.shape[1], frame.shape[0]))

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[SYS] Camera Online. Awaiting Kinematic Triggers...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Mirror display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. The Anchor (Segmentation)
    seg_results = segmentation.process(rgb_frame)
    # Create an alpha mask from the segmentation output
    condition = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.5

    # 2. The Trigger (Hand Tracking)
    hand_results = hands.process(rgb_frame)
    triggered_gesture = None
    
    if cooldown_timer > 0:
        cooldown_timer -= 1
    elif hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            gesture = detect_gesture(hand_landmarks)
            if gesture:
                triggered_gesture = gesture
                cooldown_timer = COOLDOWN_FRAMES
                break # Only process one gesture at a time

    # 3. The Warp (Trigger Generative AI)
    if triggered_gesture == "SNAP":
        current_bg = generate_background(frame, "a highly detailed cyberpunk laboratory, neon lighting, highly detailed")
    elif triggered_gesture == "PALM":
        current_bg = generate_background(frame, "a futuristic glowing neon grid, synthwave, digital cyberspace background")

    # 4. The Render (Compositing)
    if current_bg is not None:
        # Ensure sizes match (SD might return slightly different dims based on padding)
        current_bg_resized = cv2.resize(current_bg, (frame.shape[1], frame.shape[0]))
        # Composite: Original Body + AI Background
        output_frame = np.where(condition, frame, current_bg_resized)
        
        # Add HUD Text
        cv2.putText(output_frame, "[ Reality Warp: ACTIVE ]", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        output_frame = frame
        cv2.putText(output_frame, "[ Reality Warp: STANDBY ]", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show Output
    cv2.imshow('Sayyam AI Lab: Reality Warp', output_frame)

    if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()