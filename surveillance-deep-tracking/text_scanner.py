

import sys
import time

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720

OCR_EVERY_N_FRAMES = 8          # throttle OCR (it's slow) - camera preview stays smooth
WORD_CONF_THRESHOLD = 60        # 0-100, general text mode confidence cutoff

PLATE_MIN_ASPECT = 2.0
PLATE_MAX_ASPECT = 5.5
PLATE_MIN_AREA = 1500
PLATE_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Uncomment and edit if tesseract isn't automatically on your PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)

try:
    pytesseract.get_tesseract_version()
except Exception as e:
    print("WARNING: pytesseract could not find the Tesseract engine on this system.")
    print("Install it (see the docstring at the top of this file) or set")
    print("pytesseract.pytesseract.tesseract_cmd to its full path.")
    print(f"Details: {e}")


# ----------------------------------------------------------------------------
# GENERAL TEXT MODE
# ----------------------------------------------------------------------------
def run_general_ocr(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=Output.DICT)

    results = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
        if text and conf >= WORD_CONF_THRESHOLD:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            results.append({"text": text, "conf": conf, "bbox": (x, y, w, h)})
    return results


def draw_general_results(frame, results):
    for r in results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 90), 2)
        label = f"{r['text']} ({r['conf']}%)"
        cv2.putText(frame, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 90), 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# PLATE MODE
# ----------------------------------------------------------------------------
def find_plate_candidates(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blurred, 30, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < PLATE_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / h
        if PLATE_MIN_ASPECT <= aspect <= PLATE_MAX_ASPECT:
            candidates.append((x, y, w, h))
    return candidates


def ocr_plate_region(frame_bgr, bbox):
    x, y, w, h = bbox
    roi = frame_bgr[y:y + h, x:x + w]
    if roi.size == 0:
        return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = f'--psm 7 -c tessedit_char_whitelist={PLATE_CHAR_WHITELIST}'
    text = pytesseract.image_to_string(thresh, config=config)
    return text.strip()


def draw_plate_results(frame, candidates, texts):
    for (x, y, w, h), text in zip(candidates, texts):
        color = (0, 200, 255) if text else (120, 120, 120)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if text:
            cv2.putText(frame, text, (x, max(15, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    mode = "general"  # or "plate"
    frame_count = 0
    cached_general = []
    cached_plate_candidates = []
    cached_plate_texts = []
    last_ocr_time = 0.0

    print("LIVE TEXT SCANNER running. Press G=general text, P=plate mode, Q=quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % OCR_EVERY_N_FRAMES == 0:
            t0 = time.time()
            if mode == "general":
                cached_general = run_general_ocr(frame)
            else:
                cached_plate_candidates = find_plate_candidates(frame)
                cached_plate_texts = [ocr_plate_region(frame, bbox)
                                       for bbox in cached_plate_candidates]
            last_ocr_time = time.time() - t0

        out = frame.copy()
        if mode == "general":
            draw_general_results(out, cached_general)
        else:
            draw_plate_results(out, cached_plate_candidates, cached_plate_texts)

        mode_label = "GENERAL TEXT MODE" if mode == "general" else "PLATE MODE"
        cv2.putText(out, mode_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"last OCR pass: {last_ocr_time * 1000:0.0f} ms",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(out, "[G] General  [P] Plate  [Q] Quit",
                    (20, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Live Text Scanner", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('g'):
            mode = "general"
        elif key == ord('p'):
            mode = "plate"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()