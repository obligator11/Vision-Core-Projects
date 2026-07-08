

import cv2
import numpy as np
import time

CLOAK_COLOR = "red"   # default; switch with '1' (red) / '2' (green) at runtime

# HSV ranges for each supported cloak color
COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "green": [
        (np.array([35, 80, 40]), np.array([85, 255, 255])),
    ],
}


def capture_background(cap, num_frames=40):
    """Average several frames to get a clean, stable background plate."""
    print("Capturing background... step OUT of the frame now!")
    backgrounds = []
    for i in range(num_frames):
        ok, frame = cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            backgrounds.append(frame)
        time.sleep(0.02)
    if not backgrounds:
        return None
    background = np.median(backgrounds, axis=0).astype(np.uint8)
    print("Background captured! You can step back in now.")
    return background


def get_cloak_mask(hsv_frame, color):
    ranges = COLOR_RANGES[color]
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv_frame, lower, upper)
        mask = m if mask is None else (mask | m)

    # Clean up the mask: remove noise, fill small holes
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def main():
    global CLOAK_COLOR
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # warm up camera
    for _ in range(30):
        cap.read()

    background = capture_background(cap)
    if background is None:
        print("Failed to capture background.")
        return

    print(f"Cloak color: {CLOAK_COLOR.upper()}  |  Press '1' red, '2' green, 'b' recapture bg, 'q' quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = get_cloak_mask(hsv, CLOAK_COLOR)
        mask_inv = cv2.bitwise_not(mask)

        # Areas where cloak IS detected -> show background
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        # Areas where cloak is NOT detected -> show current frame (you, normally)
        visible_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

        final = cv2.addWeighted(cloak_area, 1, visible_area, 1, 0)

        cv2.putText(final, f"Cloak: {CLOAK_COLOR.upper()}  (1=red 2=green b=recapture bg q=quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Invisible Cloak", final)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            CLOAK_COLOR = "red"
        elif key == ord('2'):
            CLOAK_COLOR = "green"
        elif key == ord('b'):
            background = capture_background(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()