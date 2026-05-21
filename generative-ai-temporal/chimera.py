import cv2
import torch
import numpy as np
import multiprocessing as mp
import threading
import sys
from PIL import Image
from ultralytics import YOLO
from diffusers import AutoPipelineForImage2Image, LCMScheduler

def lcm_worker(frame_queue, texture_queue, prompt_queue):
    pipe = AutoPipelineForImage2Image.from_pretrained("Lykon/dreamshaper-7", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    current_prompt = "Ancient Mayan gold chalice, glowing runes"
    
    while True:
        if not prompt_queue.empty():
            current_prompt = prompt_queue.get()
            
        if not frame_queue.empty():
            crop = frame_queue.get()
            if crop is None:
                break
            
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            gen = pipe(prompt=current_prompt, image=pil_crop, num_inference_steps=4, guidance_scale=8.0).images[0]
            gen_cv = cv2.cvtColor(np.array(gen), cv2.COLOR_RGB2BGR)
            
            if texture_queue.empty():
                texture_queue.put(gen_cv)

def cli_listener(prompt_queue):
    while True:
        try:
            new_prompt = input()
            if new_prompt.strip():
                prompt_queue.put(new_prompt.strip())
        except EOFError:
            break

def run_chimera():
    mp.set_start_method('spawn', force=True)
    
    frame_q = mp.Queue(maxsize=1)
    texture_q = mp.Queue(maxsize=1)
    prompt_q = mp.Queue()
    
    worker_process = mp.Process(target=lcm_worker, args=(frame_q, texture_q, prompt_q), daemon=True)
    worker_process.start()
    
    cli_thread = threading.Thread(target=cli_listener, args=(prompt_q,), daemon=True)
    cli_thread.start()
    
    model = YOLO('yolo11l-seg.pt')
    
    # THE FIX: Generate a list of all 80 COCO classes, but exclude 0 (Person)
    allowed_classes = [i for i in range(80) if i != 0]
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Project Chimera', cv2.WINDOW_NORMAL)
    
    current_texture = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        dark_bg = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
        display_frame = dark_bg
        
        # Apply the exclusion filter here
        results = model(frame, classes=allowed_classes, verbose=False)
        
        if results and len(results[0].boxes) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0 and frame_q.empty():
                frame_q.put(crop)
                
            if not texture_q.empty():
                current_texture = texture_q.get()
                
            if current_texture is not None:
                tex_resized = cv2.resize(current_texture, (x2 - x1, y2 - y1))
                
                src_pts = np.float32([
                    [0, 0], 
                    [tex_resized.shape[1], 0], 
                    [tex_resized.shape[1], tex_resized.shape[0]], 
                    [0, tex_resized.shape[0]]
                ])
                dst_pts = np.float32([
                    [x1, y1], 
                    [x2, y1], 
                    [x2, y2], 
                    [x1, y2]
                ])
                
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped_tex = cv2.warpPerspective(tex_resized, matrix, (frame.shape[1], frame.shape[0]))
                
                bin_mask = (mask > 0.5).astype(np.uint8)
                inv_mask = 1 - bin_mask
                
                fg = cv2.bitwise_and(warped_tex, warped_tex, mask=bin_mask)
                bg = cv2.bitwise_and(dark_bg, dark_bg, mask=inv_mask)
                
                blended_fg = cv2.addWeighted(cv2.bitwise_and(frame, frame, mask=bin_mask), 0.2, fg, 0.8, 0)
                display_frame = cv2.add(bg, blended_fg)
                
        cv2.imshow('Project Chimera', display_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_chimera()