#!/usr/bin/env python3
"""
motion_cam.py - a webcam that only records when something moves.

Learns what the empty room looks like, then watches for anything that doesn't
belong. When movement starts it saves a clip that begins a few seconds BEFORE
the trigger, so you see what led up to it, keeps recording until things go
quiet, and writes a line to a CSV log.

Install:   pip install opencv-python numpy
Watch:     python motion_cam.py
Sensitive: python motion_cam.py --min-area 300 --sensitivity 20
One area:  python motion_cam.py --zone 200,100,400,300
Review:    python motion_cam.py --input door.mp4 --output marked.mp4

Keys (click the preview window first):
    space  arm / disarm recording      s  snapshot
    [ ]    smaller / larger minimum size for something to count
    - =    more / less sensitive
    m      show the motion mask instead of the picture
    b      forget the background and learn it again
    q/Esc  quit
"""

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


class MotionDetector:
    """Background model plus clean-up, returning boxes around what moved."""

    def __init__(self, sensitivity=25.0, min_area=800, zone=None, warmup=20):
        self.min_area = int(min_area)
        self.zone = zone
        self.warmup = int(warmup)
        self.seen = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.set_sensitivity(sensitivity)

    def set_sensitivity(self, value):
        """Lower threshold = twitchier. Rebuilds the background model."""
        self.sensitivity = float(np.clip(value, 5.0, 120.0))
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=self.sensitivity, detectShadows=False
        )
        self.seen = 0

    def reset(self):
        self.set_sensitivity(self.sensitivity)

    def update(self, frame):
        """Returns (settled, boxes, mask). settled is False while learning."""
        view = frame
        ox, oy = 0, 0
        if self.zone:
            x, y, w, h = self.zone
            view = frame[y : y + h, x : x + w]
            ox, oy = x, y

        blur = cv2.GaussianBlur(view, (5, 5), 0)
        mask = self.bg.apply(blur)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        self.seen += 1
        if self.seen <= self.warmup:
            return False, [], mask  # still working out what "empty" looks like

        found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x + ox, y + oy, w, h, int(cv2.contourArea(c))))
        boxes.sort(key=lambda b: -b[4])
        return True, boxes, mask


class EventLog:
    """One CSV row per motion event."""

    COLUMNS = ["started", "ended", "seconds", "largest_area", "clip"]

    def __init__(self, path):
        self.path = path
        fresh = not os.path.exists(path)
        self.fh = open(path, "a", newline="")
        self.writer = csv.writer(self.fh)
        if fresh:
            self.writer.writerow(self.COLUMNS)
            self.fh.flush()

    def add(self, started, ended, area, clip):
        self.writer.writerow(
            [
                datetime.fromtimestamp(started).isoformat(timespec="seconds"),
                datetime.fromtimestamp(ended).isoformat(timespec="seconds"),
                f"{ended - started:.1f}",
                area,
                clip,
            ]
        )
        self.fh.flush()

    def close(self):
        self.fh.close()


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_writer(path, fps, size):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def annotate(frame, boxes, settled, armed, recording, detector, quiet_left):
    out = frame.copy()
    if detector.zone:
        x, y, w, h = detector.zone
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 180, 60), 1)
    for x, y, w, h, area in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (60, 220, 90), 2)
        cv2.putText(out, str(area), (x, y - 6), FONT, 0.45, (60, 220, 90), 1)
    if not settled:
        text, colour = "learning the background...", (90, 200, 255)
    elif recording:
        text, colour = f"RECORDING  stops in {quiet_left:.0f}s", (0, 0, 255)
        cv2.circle(out, (26, 26), 9, (0, 0, 255), -1)
    elif not armed:
        text, colour = "disarmed", (160, 160, 160)
    elif boxes:
        text, colour = "movement", (60, 220, 90)
    else:
        text, colour = "all quiet", (200, 200, 200)
    cv2.putText(out, text, (46, 32), FONT, 0.6, colour, 2)
    cv2.putText(
        out,
        f"min area {detector.min_area}   threshold {detector.sensitivity:.0f}",
        (12, out.shape[0] - 12),
        FONT,
        0.45,
        (200, 200, 200),
        1,
    )
    return out


def review_file(args, detector):
    """Run the detector over an existing video and mark it up."""
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(args.output or f"marked_{stamp()}.mp4", fps, (w, h))

    events, active, n = [], None, 0
    quiet_frames = int(args.quiet * fps)
    since_motion = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        settled, boxes, mask = detector.update(frame)
        t = n / fps
        if settled and boxes:
            since_motion = 0
            area = boxes[0][4]
            if active is None:
                active = [t, t, area]
            else:
                active[1], active[2] = t, max(active[2], area)
        else:
            since_motion += 1
            if active and since_motion > quiet_frames:
                events.append(tuple(active))
                active = None
        writer.write(annotate(frame, boxes, settled, True, active is not None, detector, 0))
        n += 1
    if active:
        events.append(tuple(active))
    writer.release()
    cap.release()

    print(f"{n} frames, {n / fps:.1f}s, {len(events)} motion events")
    for i, (start, end, area) in enumerate(events, 1):
        print(f"  {i:2d}. {start:7.2f}s to {end:7.2f}s  ({end - start:5.2f}s) area {area}")
    return events


def run_live(args, detector):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Try --camera 1, and check "
            "that no other app is using the webcam."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, probe = cap.read()
    if not ok:
        raise SystemExit("The camera opened but sent no frames.")
    size = (probe.shape[1], probe.shape[0])

    log = EventLog(args.log)
    preroll = deque(maxlen=max(1, int(args.preroll * args.fps)))
    writer, clip_path = None, ""
    started_at, last_motion, biggest = 0.0, 0.0, 0
    armed, show_mask = True, False

    win = "Motion Cam  (space arm  [ ] size  - = sensitivity  b relearn  q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, *size)
    print(__doc__)
    print(f"Watching {size[0]}x{size[1]}, logging to {args.log}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            now = time.time()
            settled, boxes, mask = detector.update(frame)
            preroll.append(frame.copy())

            if settled and boxes and armed:
                last_motion = now
                biggest = max(biggest, boxes[0][4])
                if writer is None:
                    clip_path = os.path.join(args.outdir, f"motion_{stamp()}.mp4")
                    writer = open_writer(clip_path, args.fps, size)
                    started_at = now - len(preroll) / args.fps
                    for old in preroll:  # the seconds before the trigger
                        writer.write(old)
                    cv2.imwrite(clip_path.replace(".mp4", ".png"), frame)
                    print(f"Motion -> {clip_path}")

            quiet_left = 0.0
            if writer is not None:
                quiet_left = args.quiet - (now - last_motion)
                writer.write(frame)
                if quiet_left <= 0:
                    writer.release()
                    writer = None
                    log.add(started_at, now, biggest, clip_path)
                    print(f"Saved {clip_path} ({now - started_at:.1f}s)")
                    biggest = 0

            if show_mask:
                view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                if detector.zone:  # the mask only covers the zone
                    canvas = np.zeros_like(frame)
                    x, y, w, h = detector.zone
                    canvas[y : y + h, x : x + w] = view
                    view = canvas
            else:
                view = annotate(
                    frame, boxes, settled, armed, writer is not None, detector, quiet_left
                )
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                armed = not armed
                print("armed" if armed else "disarmed")
            elif key == ord("s"):
                name = os.path.join(args.outdir, f"snap_{stamp()}.png")
                cv2.imwrite(name, frame)
                print(f"Snapshot -> {name}")
            elif key == ord("["):
                detector.min_area = max(50, detector.min_area - 200)
            elif key == ord("]"):
                detector.min_area += 200
            elif key in (ord("-"), ord("_")):
                detector.set_sensitivity(detector.sensitivity + 5)
            elif key in (ord("="), ord("+")):
                detector.set_sensitivity(detector.sensitivity - 5)
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("b"):
                detector.reset()
                print("relearning the background")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            log.add(started_at, time.time(), biggest, clip_path)
            print(f"Saved {clip_path}")
        log.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_zone(text):
    try:
        x, y, w, h = (int(v) for v in text.split(","))
        return x, y, w, h
    except ValueError:
        raise argparse.ArgumentTypeError("zone must be x,y,width,height")


def main():
    p = argparse.ArgumentParser(description="Motion-triggered webcam recorder")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=20.0, help="recording frame rate")
    p.add_argument("--min-area", type=int, default=800, help="pixels before it counts")
    p.add_argument("--sensitivity", type=float, default=25.0, help="lower = twitchier")
    p.add_argument("--zone", type=parse_zone, help="x,y,width,height to watch")
    p.add_argument("--preroll", type=float, default=3.0, help="seconds kept before")
    p.add_argument("--quiet", type=float, default=4.0, help="seconds before stopping")
    p.add_argument("--warmup", type=int, default=20, help="frames spent learning")
    p.add_argument("--outdir", default=".", help="where clips and stills go")
    p.add_argument("--log", default="motion_log.csv")
    p.add_argument("--input", help="review a video file instead of the camera")
    p.add_argument("--output", help="annotated output for --input")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    detector = MotionDetector(args.sensitivity, args.min_area, args.zone, args.warmup)
    if args.input:
        review_file(args, detector)
    else:
        run_live(args, detector)


if __name__ == "__main__":
    main()
