import sys
import os
import cv2
import numpy as np
import pygame
import mediapipe as mp

MUSIC_FILE = "music.mp3"
GAZE_THRESHOLD = 0.08
DEBOUNCE_FRAMES = 5
SHOW_LANDMARKS = True
WINDOW_NAME = "GazeRocker"
MODEL_PATH = "face_landmarker.task"

COLOR_UP   = (100, 220, 100)
COLOR_DOWN = (60,  80,  220)
COLOR_HUD  = (255, 255, 255)
IRIS_COLOR = (0,   200, 255)

LEFT_EYE_TOP     = 159
LEFT_EYE_BOTTOM  = 145
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473


class FaceMeshWrapper:

    def __init__(self):
        mp_version = tuple(int(x) for x in mp.__version__.split(".")[:2])
        self._version = mp_version
        print(f"  mediapipe {mp.__version__} detected", end=" — ")

        if mp_version < (0, 10):
            self._init_legacy()
        else:
            self._init_tasks()

    # ── Legacy (0.9.x) ────────────────────────────────────────────────────────
    def _init_legacy(self):
        print("using mp.solutions.face_mesh")
        mp_face = mp.solutions.face_mesh
        self._mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._mode = "legacy"

    def _process_legacy(self, rgb_frame):
        result = self._mesh.process(rgb_frame)
        if result.multi_face_landmarks:
            return result.multi_face_landmarks[0].landmark
        return None

    def _init_tasks(self):
        if not os.path.exists(MODEL_PATH):
            print()
            print()
            print("=" * 60)
            print("  mediapipe 0.10+ requires a model file.")
            print()
            print("  OPTION A (recommended — easiest fix):")
            print("    pip install mediapipe==0.9.3")
            print("    Then re-run this script.")
            print()
            print("  OPTION B (stay on 0.10):")
            print(f"    Download face_landmarker.task and place it")
            print(f"    in the same folder as gaze_rocker.py")
            print()
            print("    Download URL:")
            print("    https://storage.googleapis.com/mediapipe-models/")
            print("    face_landmarker/face_landmarker/float16/1/face_landmarker.task")
            print("=" * 60)
            sys.exit(1)

        print("using FaceLandmarker tasks API")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_face_blendshapes=False,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp_image   = mp.Image
        self._mode       = "tasks"

    def _process_tasks(self, rgb_frame):
        mp_img = self._mp_image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_img)
        if result.face_landmarks:
            return result.face_landmarks[0]   # list of NormalizedLandmark
        return None

    def process(self, rgb_frame):
        if self._mode == "legacy":
            return self._process_legacy(rgb_frame)
        return self._process_tasks(rgb_frame)

    def close(self):
        if self._mode == "legacy":
            self._mesh.close()
        else:
            self._landmarker.close()


def get_iris_ratio(landmarks, eye_top_idx, eye_bottom_idx, iris_idx, frame_h):
    top    = landmarks[eye_top_idx].y * frame_h
    bottom = landmarks[eye_bottom_idx].y * frame_h
    iris_y = landmarks[iris_idx].y * frame_h
    eye_h  = bottom - top
    if eye_h < 1:
        return None
    return (iris_y - top) / eye_h - 0.5


def draw_hud(frame, gaze_value, state, threshold):
    h, w = frame.shape[:2]
    bar_color = COLOR_DOWN if state == "PLAYING" else COLOR_UP

    label = "> PLAYING" if state == "PLAYING" else "|| PAUSED"
    cv2.rectangle(frame, (10, 10), (210, 50), bar_color, -1)
    cv2.putText(frame, label, (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    bx   = w - 30
    bt   = 60
    bb   = h - 70
    brng = bb - bt
    cv2.rectangle(frame, (bx - 8, bt), (bx + 8, bb), (60, 60, 60), -1)

    ty = int(bt + brng * (-threshold + 0.5))
    ty = max(bt, min(bb, ty))
    cv2.line(frame, (bx - 14, ty), (bx + 14, ty), (0, 255, 255), 2)
    cv2.putText(frame, "thresh", (bx - 50, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    if gaze_value is not None:
        gy = int(bt + brng * (gaze_value + 0.5))
        gy = max(bt, min(bb, gy))
        cv2.circle(frame, (bx, gy), 8, bar_color, -1)

    gv_str = f"{gaze_value:+.4f}" if gaze_value is not None else " n/a  "
    cv2.putText(frame, f"gaze  : {gv_str}", (10, h - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD, 1, cv2.LINE_AA)
    cv2.putText(frame, f"thresh: {threshold:+.4f}", (10, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q=quit  +=raise  -=lower  L=markers", (10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA)


def init_music(path):
    pygame.mixer.init()
    if not os.path.exists(path):
        print(f"\n  Music file not found: {path}")
        print(f"  Place {path} in the same folder as gaze_rocker.py\n")
        return False
    pygame.mixer.music.load(path)
    pygame.mixer.music.set_volume(0.8)
    return True


def main():
    music_ok   = init_music(MUSIC_FILE)
    face_mesh  = FaceMeshWrapper()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state          = "PAUSED"
    debounce_count = 0
    pending_state  = None
    threshold      = GAZE_THRESHOLD
    show_landmarks = SHOW_LANDMARKS

    print()
    print("=" * 52)
    print("  GazeRocker  |  Sensor -> Threshold -> Actuator")
    print("=" * 52)
    print(f"  Threshold : {threshold:+.4f}  (look down past this -> music)")
    print(f"  Music     : {MUSIC_FILE}")
    print("  Keys: Q quit  |  + raise threshold  |  - lower  |  L toggle markers")
    print("  TIP : watch 'gaze' value on screen while looking")
    print("        straight ahead — set threshold just above it")
    print("=" * 52)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks  = face_mesh.process(rgb)
        frame = cv2.flip(frame, 1)
        gaze_value = None

        if landmarks is not None:
            if show_landmarks:
                for idx in [LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER]:
                    cx = int(landmarks[idx].x * w)
                    cy = int(landmarks[idx].y * h)
                    cv2.circle(frame, (cx, cy), 5, IRIS_COLOR, -1)

            l = get_iris_ratio(landmarks, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,
                               LEFT_IRIS_CENTER,  h)
            r = get_iris_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                               RIGHT_IRIS_CENTER, h)
            ratios = [v for v in [l, r] if v is not None]
            if ratios:
                gaze_value = float(np.mean(ratios))

            if gaze_value is not None:
                desired = "PLAYING" if gaze_value < -threshold else "PAUSED"

                if desired != state:
                    if desired == pending_state:
                        debounce_count += 1
                    else:
                        pending_state  = desired
                        debounce_count = 1

                    if debounce_count >= DEBOUNCE_FRAMES:
                        state = desired
                        debounce_count = 0
                        if music_ok:
                            if state == "PLAYING":
                                if not pygame.mixer.music.get_busy():
                                    pygame.mixer.music.play(-1)
                                else:
                                    pygame.mixer.music.unpause()
                                print(f"  > PLAY   (gaze={gaze_value:+.4f})")
                            else:
                                pygame.mixer.music.pause()
                                print(f"  || PAUSE  (gaze={gaze_value:+.4f})")
                else:
                    debounce_count = 0
                    pending_state  = None

        draw_hud(frame, gaze_value, state, threshold)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            threshold = round(threshold + 0.005, 4)
            print(f"  Threshold -> {threshold:+.4f}")
        elif key == ord('-'):
            threshold = round(threshold - 0.005, 4)
            print(f"  Threshold -> {threshold:+.4f}")
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"  Eye markers {'ON' if show_landmarks else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    if music_ok:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    face_mesh.close()
    print("GazeRocker closed.")


if __name__ == "__main__":
    main()