import cv2
import torch
import numpy as np
import mediapipe as mp
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
if device.type == 'cuda':
    midas.half()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

terminal_anchored = False
is_dragging = False
xray_mode = False
global_cooldown = 0
flash_alpha = 0.0

anchor_x, anchor_y, anchor_z = 0, 0, 0.0
drag_offset_x, drag_offset_y = 0, 0
terminal_w, terminal_h = 350, 200

THEMES = [(0, 200, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0)]
theme_idx = 0

def create_hud(w, h, active_drag=False, theme_color=(0, 200, 255)):
    hud = np.zeros((h, w, 3), dtype=np.uint8)
    bg_color = (40, 50, 20) if active_drag else (30, 20, 0)
    border_color = (0, 255, 0) if active_drag else theme_color
    
    cv2.rectangle(hud, (0, 0), (w, h), bg_color, -1)
    cv2.rectangle(hud, (2, 2), (w-2, h-2), border_color, 2)
    cv2.putText(hud, "SAYYAM AI LAB", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme_color, 2)
    cv2.putText(hud, "[SYS: LOOKING GLASS V12]", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    status_text = "STATUS: DRAGGING..." if active_drag else "Z-DEPTH LOCK: ACTIVE"
    status_color = (0, 255, 0) if active_drag else theme_color
    cv2.putText(hud, status_text, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    for i in range(10, w, 20): cv2.line(hud, (i, 160), (i, h), (100, 100, 100), 1)
    return hud

hud_static = create_hud(terminal_w, terminal_h, False, THEMES[theme_idx])
hud_active = create_hud(terminal_w, terminal_h, True, THEMES[theme_idx])

cv2.namedWindow("Looking Glass", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Looking Glass", 1280, 720)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
prev_time = time.time()

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def dist3d(p1, p2, w, h):
    dx = (p1.x - p2.x) * w
    dy = (p1.y - p2.y) * h
    dz = (p1.z - p2.z) * w 
    return math.sqrt(dx*dx + dy*dy + dz*dz)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(img_rgb).to(device)
    if device.type == 'cuda':
        input_batch = input_batch.half()
        
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=(h_frame, w_frame), mode="bilinear", align_corners=False
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    if xray_mode:
        depth_map_f32 = depth_map.astype(np.float32)
        depth_normalized = cv2.normalize(depth_map_f32, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        frame = cv2.addWeighted(frame, 0.4, depth_colormap, 0.6, 0)
        cv2.putText(frame, "X-RAY TOPOGRAPHY ENABLED", (w_frame//2 - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    results = hands.process(img_rgb)
    if global_cooldown > 0: global_cooldown -= 1
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb = hand_landmarks.landmark[4]
            index, pip_index = hand_landmarks.landmark[8], hand_landmarks.landmark[6]
            middle, pip_middle = hand_landmarks.landmark[12], hand_landmarks.landmark[10]
            ring, pip_ring = hand_landmarks.landmark[16], hand_landmarks.landmark[14]
            pinky, pip_pinky = hand_landmarks.landmark[20], hand_landmarks.landmark[18]
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]
            
            hand_size = dist3d(wrist, middle_mcp, w_frame, h_frame)
            
            tx, ty = int(thumb.x * w_frame), int(thumb.y * h_frame)
            ix, iy = int(index.x * w_frame), int(index.y * h_frame)
            cx, cy = clamp((tx + ix) // 2, 0, w_frame - 1), clamp((ty + iy) // 2, 0, h_frame - 1)
            
            index_curled = dist3d(index, wrist, w_frame, h_frame) < dist3d(pip_index, wrist, w_frame, h_frame)
            middle_curled = dist3d(middle, wrist, w_frame, h_frame) < dist3d(pip_middle, wrist, w_frame, h_frame)
            ring_curled = dist3d(ring, wrist, w_frame, h_frame) < dist3d(pip_ring, wrist, w_frame, h_frame)
            pinky_curled = dist3d(pinky, wrist, w_frame, h_frame) < dist3d(pip_pinky, wrist, w_frame, h_frame)
            
            all_curled = index_curled and middle_curled and ring_curled and pinky_curled
            
            pinch_threshold = hand_size * 0.7 if is_dragging else hand_size * 0.35
            is_pinching = dist3d(thumb, index, w_frame, h_frame) < pinch_threshold and not all_curled
            
            thumb_to_palm = dist3d(thumb, middle_mcp, w_frame, h_frame)
            
            is_fist = all_curled and (thumb_to_palm < hand_size * 0.7) and not is_pinching
            is_thumbs_up = all_curled and (thumb_to_palm > hand_size * 0.7) and (thumb.y < middle_mcp.y)
            is_thumbs_down = all_curled and (thumb_to_palm > hand_size * 0.7) and (thumb.y > middle_mcp.y)
            
            is_peace = not index_curled and not middle_curled and ring_curled and pinky_curled and not is_pinching
            is_rock_on = not index_curled and middle_curled and ring_curled and not pinky_curled and not is_pinching
            
            if is_fist and terminal_anchored:
                cv2.putText(frame, "HUD DESTROYED", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                terminal_anchored = False
                is_dragging = False

            elif is_thumbs_up and global_cooldown == 0:
                xray_mode = True
                global_cooldown = 30
                
            elif is_thumbs_down and global_cooldown == 0:
                xray_mode = False
                global_cooldown = 30
                
            elif is_peace and global_cooldown == 0:
                theme_idx = (theme_idx + 1) % len(THEMES)
                hud_static = create_hud(terminal_w, terminal_h, False, THEMES[theme_idx])
                hud_active = create_hud(terminal_w, terminal_h, True, THEMES[theme_idx])
                cv2.putText(frame, "THEME CYCLE", (tx, ty - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, THEMES[theme_idx], 3)
                global_cooldown = 30

            elif is_rock_on and global_cooldown == 0:
                flash_alpha = 1.0 
                filename = f"AR_Snapshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame) 
                print(f"[SYSTEM] Snapshot saved to disk: {filename}")
                global_cooldown = 40
                
            elif is_pinching:
                cv2.circle(frame, (cx, cy), 15, THEMES[theme_idx], cv2.FILLED)
                if not terminal_anchored:
                    anchor_x = clamp(cx - terminal_w // 2, 0, w_frame - terminal_w)
                    anchor_y = clamp(cy - terminal_h // 2, 0, h_frame - terminal_h)
                    anchor_z = depth_map[cy, cx]
                    terminal_anchored = True
                else:
                    grab_margin = 60 
                    if not is_dragging and (anchor_x - grab_margin <= cx <= anchor_x + terminal_w + grab_margin) and (anchor_y - grab_margin <= cy <= anchor_y + terminal_h + grab_margin):
                        is_dragging = True
                        drag_offset_x = cx - anchor_x
                        drag_offset_y = cy - anchor_y
            else:
                is_dragging = False 
                
            if is_dragging:
                anchor_x = clamp(cx - drag_offset_x, 0, w_frame - terminal_w)
                anchor_y = clamp(cy - drag_offset_y, 0, h_frame - terminal_h)
                anchor_z = depth_map[cy, cx] 

    if terminal_anchored:
        end_x = min(anchor_x + terminal_w, w_frame)
        end_y = min(anchor_y + terminal_h, h_frame)
        render_w, render_h = end_x - anchor_x, end_y - anchor_y
        
        if render_w > 0 and render_h > 0:
            current_hud = hud_active if is_dragging else hud_static
            hud_slice = current_hud[0:render_h, 0:render_w]
            frame_roi = frame[anchor_y:end_y, anchor_x:end_x]
            depth_roi = depth_map[anchor_y:end_y, anchor_x:end_x]
            
            occlusion_mask = (depth_roi < anchor_z * 1.1).astype(np.uint8)
            occlusion_mask_3d = np.repeat(occlusion_mask[:, :, np.newaxis], 3, axis=2)
            
            blended_roi = cv2.addWeighted(frame_roi, 0.2, hud_slice, 0.8, 0)
            final_roi = np.where(occlusion_mask_3d == 1, blended_roi, frame_roi)
            frame[anchor_y:end_y, anchor_x:end_x] = final_roi

    if flash_alpha > 0:
        white_overlay = np.full(frame.shape, 255, dtype=np.uint8)
        frame = cv2.addWeighted(frame, 1 - flash_alpha, white_overlay, flash_alpha, 0)
        flash_alpha = max(0, flash_alpha - 0.1) 

    curr_time = time.time()
    cv2.putText(frame, f'FPS: {int(1 / (curr_time - prev_time))} | RTX ACCEL', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, THEMES[theme_idx], 2)
    prev_time = curr_time

    cv2.imshow("Looking Glass", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()