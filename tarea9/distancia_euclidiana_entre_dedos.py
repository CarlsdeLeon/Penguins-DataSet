import cv2
import mediapipe as mp
import math
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

DIST_MIN = 20
DIST_MAX = 200
TOUCH_THRESHOLD = 30

def euclidean(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def map_range(value, in_min, in_max, out_min, out_max):
    ratio = (value - in_min) / (in_max - in_min)
    return int(max(out_min, min(out_max, ratio * (out_max - out_min) + out_min)))

def dedo_levantado(lm, tip, dip, pip, mcp):
    return (lm[tip].y < lm[dip].y and
            lm[dip].y < lm[pip].y and
            lm[pip].y < lm[mcp].y)

def dibujar_barra(frame, nivel, x=30, y=80, ancho=30, alto=300):
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (150, 150, 150), 2)

    fill_h = int((nivel / 100) * alto)
    fill_y = y + alto - fill_h

    if nivel < 20:
        color = (0, 0, 255)
    elif nivel < 50:
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(frame, (x, fill_y), (x + ancho, y + alto), color, -1)

    cv2.putText(frame, "100", (x + ancho + 5, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, " 50", (x + ancho + 5, y + alto // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, "  0", (x + ancho + 5, y + alto),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"{nivel}%", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

latest_result: HandLandmarkerResult = None

def guardar_resultado(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

MODEL_PATH = "./tarea 5/hand_landmarker.task"

options = HandLandmarkerOpts(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=guardar_resultado
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

DEDOS = {
    'Pulgar': (4, 3, 2, 1),
    'Indice': (8, 7, 6, 5),
    'Medio':  (12, 11, 10, 9),
    'Anular': (16, 15, 14, 13),
    'Menique':(20, 19, 18, 17),
}

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        nivel = 0

        if latest_result and latest_result.hand_landmarks:
            lm = latest_result.hand_landmarks[0]

            for start, end in HAND_CONNECTIONS:
                x1, y1 = int(lm[start].x * w), int(lm[start].y * h)
                x2, y2 = int(lm[end].x * w), int(lm[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

            for punto in lm:
                cx, cy = int(punto.x * w), int(punto.y * h)
                cv2.circle(frame, (cx, cy), 4, (200, 200, 200), -1)

            thumb_x = int(lm[4].x * w)
            thumb_y = int(lm[4].y * h)
            index_x = int(lm[8].x * w)
            index_y = int(lm[8].y * h)

            dist = euclidean(thumb_x, thumb_y, index_x, index_y)
            nivel = map_range(dist, DIST_MIN, DIST_MAX, 0, 100)

            if dist < TOUCH_THRESHOLD:
                color_linea = (0, 0, 255)
            elif dist < 80:
                color_linea = (0, 165, 255)
            else:
                color_linea = (0, 255, 0)

            grosor = 4 if dist < TOUCH_THRESHOLD else 2

            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), color_linea, grosor)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)

            mid_x = (thumb_x + index_x) // 2
            mid_y = (thumb_y + index_y) // 2
            cv2.putText(frame, f"{int(dist)}px",
                        (mid_x + 8, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_linea, 2)

            if dist < TOUCH_THRESHOLD:
                cv2.putText(frame, "! CONTACTO !",
                            (w // 2 - 90, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 3)

            dedos_levantados = [
                nombre for nombre, (tip, dip, pip, mcp) in DEDOS.items()
                if dedo_levantado(lm, tip, dip, pip, mcp)
            ]

            cv2.putText(frame,
                        f'Dedos: {", ".join(dedos_levantados) or "ninguno"}',
                        (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 180), 2)

        dibujar_barra(frame, nivel)

        cv2.putText(frame, "LM4=PULGAR  LM8=INDICE",
                    (w - 240, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow('Air Controller — MediaPipe Tasks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()