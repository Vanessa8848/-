from maix import app, camera, display, image, time, uart
import cv2
import numpy as np
import struct
import math

CAM_W = 512
CAM_H = 320
MIN_AREA = 1600
MAX_AREA = 22000
MAX_ASPECT = 3.5

REAL_WIDTH_MM = 300.0
REAL_HEIGHT_MM = 200.0
FOCAL_LENGTH = 1000.0 * 89.0 / 300.0

# ===== 激光参数：保持你调好的值 =====
LASER_MOUNT_X_MM = 0.0
LASER_MOUNT_Y_MM = 30.0
REF_DIST_MM = 1000.0
LASER_OFFSET_X_AT_REF = -75.0    # ← 保持你调好的值
LASER_OFFSET_Y_AT_REF = 220.0    # ← 保持你调好的值

ser = uart.UART("/dev/ttyS0", 115200)

def send_offset(err_x, err_y):
    data = struct.pack("<hh", int(err_x), int(err_y))
    ck = 0
    for b in data:
        ck ^= b
    ser.write(b"\xAA\x55" + data + bytes([ck]))

def order_points(points):
    pts = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(4)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def laser_offset_px(w_px, h_px):
    D = FOCAL_LENGTH * REAL_WIDTH_MM / w_px
    theta_x = math.atan((LASER_OFFSET_X_AT_REF - LASER_MOUNT_X_MM) / REF_DIST_MM)
    theta_y = math.atan((LASER_OFFSET_Y_AT_REF - LASER_MOUNT_Y_MM) / REF_DIST_MM)
    offset_x_mm = LASER_MOUNT_X_MM + D * math.tan(theta_x)
    offset_y_mm = LASER_MOUNT_Y_MM + D * math.tan(theta_y)
    offset_x_px = int(offset_x_mm * FOCAL_LENGTH / D)
    offset_y_px = int(offset_y_mm * FOCAL_LENGTH / D)
    return offset_x_px, offset_y_px

def find_best_rectangle(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    edged = cv2.Canny(closed, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    best = None
    max_area = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if area < MIN_AREA:
            continue
        rect = order_points(approx.reshape(4, 2))

        vec1 = rect[1] - rect[0]
        vec2 = rect[2] - rect[1]
        n = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if n < 1e-6:
            continue
        angle_cos = np.dot(vec1, vec2) / n
        if abs(angle_cos) > 0.3:
            continue

        w = np.linalg.norm(rect[1] - rect[0])
        h = np.linalg.norm(rect[2] - rect[1])
        if w < 1 or h < 1:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > MAX_ASPECT:
            continue

        if area > max_area:
            max_area = area
            best = {
                "area": area,
                "approx": rect.astype("float32"),
            }
    return best

def run_find_rects():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    prev_err_x = 0
    prev_err_y = 0
    have_prev = False
    lost_frames = 0
    MAX_LOST = 15          # 连续丢15帧以内，保留上一帧误差

    while not app.need_exit():
        img = cam.read()
        img_cv = image.image2cv(img, False, False)
        best = find_best_rectangle(img_cv)

        if not best:
            lost_frames += 1
            if have_prev and lost_frames <= MAX_LOST:
                send_offset(prev_err_x, prev_err_y)   # ← 保留上一帧误差继续发
            img.draw_string(8, 8, "no rect", image.COLOR_RED, scale=1)
            disp.show(img)
            continue

        lost_frames = 0
        ordered = best["approx"]

        ox = int(np.mean(ordered[:, 0]))
        oy = int(np.mean(ordered[:, 1]))
        w_px = int(np.linalg.norm(ordered[1] - ordered[0]))
        h_px = int(np.linalg.norm(ordered[2] - ordered[1]))

        screen_center = (img.width() // 2, img.height() // 2)
        offset_x_px, offset_y_px = laser_offset_px(w_px, h_px)
        aim_x = screen_center[0] + offset_x_px
        aim_y = screen_center[1] + offset_y_px

        err_x = ox - aim_x
        err_y = oy - aim_y
        send_offset(err_x, err_y)

        prev_err_x = err_x      # 保存这一帧误差
        prev_err_y = err_y
        have_prev = True

        pts = ordered.astype("int32")
        for i in range(4):
            x1, y1 = int(pts[i][0]), int(pts[i][1])
            x2, y2 = int(pts[(i + 1) % 4][0]), int(pts[(i + 1) % 4][1])
            img.draw_line(x1, y1, x2, y2, image.COLOR_GREEN, thickness=2)
        img.draw_circle(ox, oy, 4, image.COLOR_RED, thickness=-1)

        disp.show(img)

if __name__ == "__main__":
    try:
        print("start ok")
        run_find_rects()
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print("========== ERROR ==========")
        print(err_msg)
        print("========== END ERROR ==========")