# GazeRocker 🎸

**BUILDCORED ORCAS — Day 1 · Week 1: Body as Input**

> Look down → rock plays. Look up → silence. Your face is the switch.

---

## What It Does

GazeRocker reads your webcam, tracks where your irises are pointing using MediaPipe FaceMesh, and converts that into a single signed number — your **gaze value**. When that number drops below a threshold (you look down), pygame starts looping your track. When you look back up, it pauses. Everything runs locally with no cloud, no API keys, no hardware.

A live HUD overlays the webcam feed showing:
- Playback state badge (PLAYING / PAUSED)
- A vertical gauge with your live gaze dot and the threshold line
- Numeric readout of `gaze` and `thresh` values
- Optional iris markers (toggle with `L`)

---

## Hardware Concept: Tilt Sensor → Comparator → Relay

This project is a direct software mirror of a hardware tilt-switch circuit.

| Hardware | This Build |
|---|---|
| Tilt sensor (e.g. SW-520D, MPU-6050) | Webcam + MediaPipe iris landmarks |
| Analog comparator (e.g. LM393) | `gaze_value < -threshold` |
| Relay / transistor switch | `pygame.mixer.music.play / pause` |
| Actuator (speaker, motor, LED) | OS audio output |

```
[Webcam] → FaceMesh iris ratio → comparator(threshold) → pygame play/pause
   ↑                                      ↑                      ↑
 sensor                             digital logic            actuator
```

The debounce counter (5 frames before a state flip commits) is the software equivalent of a **Schmitt trigger** — it prevents rapid toggling when the signal hovers near the threshold boundary. In hardware this is achieved with hysteresis resistors on the comparator feedback pin.

---

## How the Gaze Value Is Calculated

MediaPipe's `refine_landmarks=True` exposes iris-centre landmarks (468, 473). For each eye, the script computes:

```
ratio  = (iris_y - top_lid_y) / eye_height   # 0 = top lid, 1 = bottom lid
signed = ratio - 0.5                          # 0 = neutral, neg = up, pos = down
```

Left and right eyes are averaged. The comparator fires `PLAYING` when:

```python
gaze_value < -threshold   # iris moved toward top lid = looking down
```

The sign inversion exists because MediaPipe's eyelid reference landmarks shift partially with the eye as you look down, causing the iris-to-lid ratio to behave opposite to the naive expectation.

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| `opencv-python` | any | Webcam capture, frame rendering, HUD drawing |
| `mediapipe` | 0.9.x or 0.10.x | FaceMesh — 478 landmarks including iris centres |
| `pygame` | any | Audio mixer: `play(-1)`, `pause`, `unpause` |
| `numpy` | any | Average left/right iris ratios |

---

## Setup

```bash
pip install opencv-python mediapipe pygame numpy
```

> **mediapipe 0.10+ note:** the `mp.solutions` API was removed in 0.10. The script detects your version automatically. If you're on 0.10+, either:
>
> **Option A (easiest):** downgrade
> ```bash
> pip install mediapipe==0.9.3
> ```
>
> **Option B:** download the model file and place it next to `gaze_rocker.py` as `face_landmarker.task`:
> ```
> https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
> ```

Place your music file in the same folder as the script and name it `music.mp3` (or edit `MUSIC_FILE` at the top of `gaze_rocker.py`).

```bash
python gaze_rocker.py
```

---

## Tuning the Threshold

The default threshold is `0.08`. Every face and webcam position is different — here's how to dial it in:

1. Run the script and look straight at your screen
2. Read the `gaze:` value from the HUD — that's your **resting baseline** (typically `-0.02` to `+0.05`)
3. Use `+` / `-` to move the threshold until the cyan line sits just above your resting dot
4. A deliberate downward glance (chin toward chest, eyes down) should now cross it cleanly

---

## Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `+` or `=` | Raise threshold (harder to trigger) |
| `-` | Lower threshold (easier to trigger) |
| `L` | Toggle iris marker dots on / off |

---

## Configuration (top of script)

| Constant | Default | Description |
|---|---|---|
| `MUSIC_FILE` | `"music.mp3"` | Path to your audio file |
| `GAZE_THRESHOLD` | `0.08` | Trigger sensitivity |
| `DEBOUNCE_FRAMES` | `5` | Frames before state commits |
| `SHOW_LANDMARKS` | `True` | Show iris dots at startup |
| `MODEL_PATH` | `"face_landmarker.task"` | Model file path (0.10+ only) |

---

## Files

```
day01_gaze_rocker/
├── gaze_rocker.py        # Main script
├── music.mp3             # Your track (not included)
├── face_landmarker.task  # Only needed for mediapipe 0.10+
└── README.md             # This file
```

---

## Known Limitations

- Requires reasonable lighting — FaceMesh accuracy drops significantly in low light or with strong backlight behind you
- `refine_landmarks=True` adds ~2–3 ms per frame; acceptable on most laptops
- Single face only (`max_num_faces=1`)
- Blink frames return `None` and are skipped without affecting playback state
- Threshold is sensitive to head tilt — works best when your head is roughly level

---

## What I Learned

The core insight is that **threshold-based control is the simplest possible closed-loop system** — no history, no PID, just one comparison. Every real sensor-to-actuator pipeline starts here before adding complexity.

Three bugs encountered and fixed during the build:

1. **`mp.solutions` removed in mediapipe 0.10+** — solved by writing a version-detecting wrapper class that routes to the correct API at runtime
2. **Threshold too low** — the resting forward gaze already sat above `0.015`, so music played at neutral and paused on upward tilt. Fixed by raising the default to `0.08` and adding calibration guidance
3. **Gaze sign inverted** — MediaPipe's eyelid landmarks shift partially with eye movement, making the iris-to-lid ratio behave opposite to the naive geometric expectation. Fixed by flipping the comparator: `gaze_value < -threshold`

---

## ORCAS v2.0 Preview

In v2.0 this pattern reappears with a real tilt sensor (SW-520D ball-tilt switch or MPU-6050 IMU) on a Raspberry Pi Pico W. The Pico reads a GPIO HIGH/LOW, normalises it to a float, crosses a threshold, and fires a relay that switches a physical amplifier. The `gaze_value` pipeline here maps directly onto the `adc_reading → normalised float` pipeline you'd write for the IMU — same logic, different medium.

---

*BUILDCORED ORCAS · 30 Days · 30 Projects · Zero Hardware Required*
