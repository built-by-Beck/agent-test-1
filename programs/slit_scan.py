#!/usr/bin/env python3
"""
slit_scan.py - a camera where every part of the picture is a different moment.

Instead of showing you one instant, it builds each frame out of many. In slit
mode one thin strip of live camera is added at the edge and everything else
slides over, so the image reads left-to-right as time. Wave a hand and it
stretches into a ribbon; stand still and you look normal.

Install:   pip install opencv-python numpy
Run:       python slit_scan.py
Convert:   python slit_scan.py --input clip.mp4 --output smear.mp4 --mode bands

Modes:
    slit    time runs across the frame, newest at the right edge
    slitv   time runs down the frame, newest at the bottom
    bands   the frame in horizontal stripes, each from a different moment
    echo    moving things leave a fading trail behind them
    paint   light painting: the brightest thing ever seen at each pixel

Keys (click the preview window first):
    m      next mode               c  clear what's built up
    [ ]    shorter / longer time spread
    , .    thinner / wider slit (slit and slitv modes)
    r      start / stop recording  s  snapshot
    f      mirror the camera
    q/Esc  quit
"""

import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
MODES = ["slit", "slitv", "bands", "echo", "paint"]


class TimeSmear:
    """Holds whatever the current mode has built up so far."""

    def __init__(self, size, mode="slit", bands=48, decay=0.88, slit=2, at="center"):
        self.w, self.h = size
        self.mode = mode
        self.bands = int(np.clip(bands, 4, 160))
        self.decay = float(np.clip(decay, 0.5, 0.99))
        self.slit = int(np.clip(slit, 1, 64))
        self.at = at
        self.history = deque(maxlen=self.bands)
        self.clear()

    def _source(self, span, total):
        """Where the live strip is read from: the start, middle, or end."""
        if self.at == "left":
            return 0
        if self.at == "right":
            return total - span
        return (total - span) // 2

    def clear(self):
        self.canvas = np.zeros((self.h, self.w, 3), np.uint8)
        self.history.clear()
        self.filled = 0

    def cycle_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        self.clear()

    def set_bands(self, n):
        self.bands = int(np.clip(n, 4, 160))
        self.history = deque(self.history, maxlen=self.bands)

    def set_slit(self, n):
        self.slit = int(np.clip(n, 1, 64))

    def push(self, frame):
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)

        if self.mode == "slit":
            # Everything shifts left; a strip of now goes in at the right.
            s = min(self.slit, self.w)
            x = self._source(s, self.w)
            self.canvas[:, :-s] = self.canvas[:, s:]
            self.canvas[:, -s:] = frame[:, x : x + s]
            self.filled = min(self.w, self.filled + s)

        elif self.mode == "slitv":
            s = min(self.slit, self.h)
            y = self._source(s, self.h)
            self.canvas[:-s] = self.canvas[s:]
            self.canvas[-s:] = frame[y : y + s]
            self.filled = min(self.h, self.filled + s)

        elif self.mode == "bands":
            # Each stripe is taken from a different frame in the recent past,
            # oldest at the top, so a moving arm staircases down the picture.
            self.history.append(frame)
            n = len(self.history)
            edges = np.linspace(0, self.h, n + 1).astype(int)
            for i in range(n):
                y0, y1 = edges[i], edges[i + 1]
                if y1 > y0:
                    self.canvas[y0:y1] = self.history[i][y0:y1]

        elif self.mode == "echo":
            faded = (self.canvas.astype(np.float32) * self.decay).astype(np.uint8)
            self.canvas = np.maximum(faded, frame)

        elif self.mode == "paint":
            self.canvas = np.maximum(self.canvas, frame)

        return self.canvas

    def label(self):
        if self.mode in ("slit", "slitv"):
            return f"{self.mode}  slit {self.slit}px at {self.at}"
        if self.mode == "bands":
            return f"bands  {self.bands} stripes"
        if self.mode == "echo":
            return f"echo  decay {self.decay:.2f}"
        return "paint  (brightest wins)"


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_writer(path, fps, size):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def convert_file(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
    smear = TimeSmear((w, h), args.mode, args.bands, args.decay, args.slit, args.slit_at)
    out = args.output or f"smear_{stamp()}.mp4"
    writer = open_writer(out, fps, (w, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(smear.push(frame).copy())
        n += 1
    writer.release()
    cap.release()
    print(f"Wrote {n} frames to {out}  (mode: {args.mode})")


def run_live(args):
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
    size = (probe.shape[1] // 2 * 2, probe.shape[0] // 2 * 2)

    smear = TimeSmear(size, args.mode, args.bands, args.decay, args.slit, args.slit_at)
    writer, rec_start, rec_frames, rec_path = None, 0.0, 0, ""
    mirror = True

    win = "Slit Scan  (m mode  [ ] spread  , . slit  r record  s snap  q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, *size)
    print(__doc__)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            out = smear.push(frame)

            if writer is not None:
                due = int((time.time() - rec_start) * args.fps) + 1
                while rec_frames < due:
                    writer.write(out)
                    rec_frames += 1

            view = out.copy()
            cv2.putText(view, smear.label(), (12, 28), FONT, 0.6, (90, 255, 160), 2)
            if writer is not None:
                cv2.circle(view, (size[0] - 28, 26), 9, (0, 0, 255), -1)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                smear.cycle_mode()
            elif key == ord("c"):
                smear.clear()
            elif key == ord("["):
                smear.set_bands(smear.bands - 4)
                smear.decay = max(0.5, smear.decay - 0.03)
            elif key == ord("]"):
                smear.set_bands(smear.bands + 4)
                smear.decay = min(0.99, smear.decay + 0.03)
            elif key == ord(","):
                smear.set_slit(smear.slit - 1)
            elif key == ord("."):
                smear.set_slit(smear.slit + 1)
            elif key == ord("f"):
                mirror = not mirror
            elif key == ord("s"):
                name = f"slit_{stamp()}.png"
                cv2.imwrite(name, out)
                print(f"Snapshot -> {name}")
            elif key == ord("r"):
                if writer is None:
                    rec_path = args.output or f"slit_{stamp()}.mp4"
                    writer = open_writer(rec_path, args.fps, size)
                    rec_start, rec_frames = time.time(), 0
                    print(f"Recording -> {rec_path}")
                else:
                    writer.release()
                    writer = None
                    print(f"Saved {rec_path}")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved {rec_path}")
        cap.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Slit-scan time-smear camera")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mode", choices=MODES, default="slit")
    p.add_argument("--slit", type=int, default=2, help="strip width in pixels")
    p.add_argument(
        "--slit-at",
        choices=["left", "center", "right"],
        default="center",
        help="where the live strip is read from",
    )
    p.add_argument("--bands", type=int, default=48, help="stripes in bands mode")
    p.add_argument("--decay", type=float, default=0.88, help="trail length in echo")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--input", help="convert a video file instead")
    p.add_argument("--output", help="output path (default: timestamped)")
    args = p.parse_args()

    if args.input:
        convert_file(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
