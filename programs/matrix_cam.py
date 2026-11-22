#!/usr/bin/env python3
"""
matrix_cam.py - webcam video rendered as letters, numbers and symbols.

Every "pixel" of the output is a character. Bright areas get dense, bright
characters; dark areas get sparse, dim ones, so the picture stays readable.

Install:   pip install opencv-python numpy
Run:       python matrix_cam.py
Convert:   python matrix_cam.py --input clip.mp4 --output clip_matrix.mp4

Keys (click the preview window first):
    r      start / stop recording
    s      save a snapshot (PNG)
    m      cycle colour mode: matrix -> colour -> white -> amber
    x      toggle the falling "digital rain"
    [ ]    smaller / bigger characters
    f      mirror the image
    q/Esc  quit
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

CHARS = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "!@#$%&*+=-:;.,<>?/\\|()[]{}^~'\""
)
MODES = ["matrix", "colour", "white", "amber"]
TINTS = {  # BGR
    "matrix": (70, 255, 90),
    "white": (255, 255, 255),
    "amber": (0, 190, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_atlas(cell_h):
    """Draw every character once into a small tile, sorted dark -> dense.

    Returns (atlas, cell_w) where atlas has shape (n_chars, cell_h, cell_w).
    """
    cell_w = max(4, int(round(cell_h * 0.6)))
    scale = cell_h * 0.8 / 22.0  # Hershey Simplex is ~22px tall at scale 1
    thick = 1 if cell_h < 20 else 2
    tiles = [np.zeros((cell_h, cell_w), np.uint8)]  # blank = darkest
    for ch in CHARS:
        tile = np.zeros((cell_h, cell_w), np.uint8)
        (tw, th), _ = cv2.getTextSize(ch, FONT, scale, thick)
        x = (cell_w - tw) // 2
        y = (cell_h + th) // 2
        cv2.putText(tile, ch, (x, y), FONT, scale, 255, thick, cv2.LINE_AA)
        tiles.append(tile)
    tiles.sort(key=lambda t: t.mean())
    return np.stack(tiles), cell_w


class MatrixRenderer:
    def __init__(self, out_w, out_h, cell_h=12, mode="matrix", rain=True):
        self.out_w, self.out_h = out_w, out_h
        self.mode = mode
        self.rain_on = rain
        self.rng = np.random.default_rng()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.set_cell(cell_h)

    def set_cell(self, cell_h):
        self.cell_h = int(np.clip(cell_h, 6, 40))
        self.atlas, self.cell_w = build_atlas(self.cell_h)
        self.rows = self.out_h // self.cell_h
        self.cols = self.out_w // self.cell_w
        n = len(self.atlas)
        # Per-cell character "wobble" so the glyphs flicker like the film.
        self.jitter_max = max(1, n // 12)
        self.jitter = self.rng.integers(
            -self.jitter_max, self.jitter_max + 1, (self.rows, self.cols)
        )
        # Digital rain state: one falling head per column.
        self.rain = np.zeros((self.rows, self.cols), np.float32)
        self.heads = self.rng.uniform(-self.rows, self.rows, self.cols)
        self.speeds = self.rng.uniform(0.3, 1.1, self.cols)

    def cycle_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]

    def _step_rain(self):
        self.rain *= 0.90
        self.heads += self.speeds
        done = self.heads > self.rows + 8
        k = int(done.sum())
        if k:
            self.heads[done] = self.rng.uniform(-self.rows * 0.6, 0, k)
            self.speeds[done] = self.rng.uniform(0.3, 1.1, k)
        r = self.heads.astype(int)
        ok = (r >= 0) & (r < self.rows)
        self.rain[r[ok], np.nonzero(ok)[0]] = 1.0

    def render(self, frame):
        rows, cols, ch, cw = self.rows, self.cols, self.cell_h, self.cell_w
        small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
        gray = self.clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        lum = gray.astype(np.float32) / 255.0

        # Re-roll the wobble on ~8% of cells each frame.
        flick = self.rng.random((rows, cols)) < 0.08
        k = int(flick.sum())
        self.jitter[flick] = self.rng.integers(
            -self.jitter_max, self.jitter_max + 1, k
        )

        n = len(self.atlas)
        idx = (lum * (n - 1)).astype(int) + self.jitter
        # Keep truly dark cells blank-ish so the image keeps its contrast.
        idx = np.clip(idx, np.where(lum < 0.06, 0, 1), n - 1)

        glyphs = self.atlas[idx]  # (rows, cols, ch, cw)
        mask = glyphs.transpose(0, 2, 1, 3).reshape(rows * ch, cols * cw)

        # Per-cell colour: brightness follows the camera image.
        level = 0.18 + 0.82 * lum
        if self.mode == "colour":
            cell = small.astype(np.float32)
            peak = cell.max(axis=2, keepdims=True) + 1.0
            cell = cell / peak * 255.0 * level[..., None]
        else:
            cell = level[..., None] * np.array(TINTS[self.mode], np.float32)

        if self.rain_on:
            self._step_rain()
            glow = self.rain[..., None]
            cell = cell * (1.0 + 0.9 * glow)
            head = (self.rain > 0.95)[..., None]
            cell = np.where(head, 255.0, cell)  # white-hot leading character

        cell = np.clip(cell, 0, 255).astype(np.uint8)
        colour = cv2.resize(
            cell, (cols * cw, rows * ch), interpolation=cv2.INTER_NEAREST
        )
        img = cv2.multiply(
            colour, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), scale=1 / 255.0
        )

        # Centre on a fixed-size canvas so the video size never changes.
        canvas = np.zeros((self.out_h, self.out_w, 3), np.uint8)
        y0 = (self.out_h - img.shape[0]) // 2
        x0 = (self.out_w - img.shape[1]) // 2
        canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
        return canvas


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_writer(path, fps, size):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def convert_file(args, renderer):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    out = args.output or f"matrix_{stamp()}.mp4"
    writer = open_writer(out, fps, (args.width, args.height))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(renderer.render(frame))
        count += 1
    writer.release()
    cap.release()
    print(f"Wrote {count} frames to {out}")


def run_camera(args, renderer):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Try --camera 1, and check "
            "that no other app is using the webcam."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "Matrix Cam  (r=record  m=mode  x=rain  [ ]=size  s=snap  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    writer, rec_start, rec_frames, rec_path = None, 0.0, 0, ""
    mirror = True
    print(__doc__)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped sending frames.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            out = renderer.render(frame)

            if writer is not None:
                # Write as many copies as real time demands, so the recording
                # plays at true speed even if the loop runs slow or fast.
                due = int((time.time() - rec_start) * args.fps) + 1
                while rec_frames < due:
                    writer.write(out)
                    rec_frames += 1

            view = out
            if writer is not None:
                view = out.copy()
                cv2.circle(view, (26, 26), 9, (0, 0, 255), -1)
                cv2.putText(view, "REC", (44, 33), FONT, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                if writer is None:
                    rec_path = args.output or f"matrix_{stamp()}.mp4"
                    writer = open_writer(
                        rec_path, args.fps, (args.width, args.height)
                    )
                    rec_start, rec_frames = time.time(), 0
                    print(f"Recording -> {rec_path}")
                else:
                    writer.release()
                    writer = None
                    print(f"Saved {rec_path}")
            elif key == ord("s"):
                name = f"matrix_{stamp()}.png"
                cv2.imwrite(name, out)
                print(f"Snapshot -> {name}")
            elif key == ord("m"):
                renderer.cycle_mode()
            elif key == ord("x"):
                renderer.rain_on = not renderer.rain_on
            elif key == ord("["):
                renderer.set_cell(renderer.cell_h - 2)
            elif key == ord("]"):
                renderer.set_cell(renderer.cell_h + 2)
            elif key == ord("f"):
                mirror = not mirror
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved {rec_path}")
        cap.release()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Matrix-style text webcam recorder")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--input", help="convert a video file instead of the webcam")
    p.add_argument("--output", help="output .mp4 path (default: timestamped)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--cell", type=int, default=12, help="character height, px")
    p.add_argument("--mode", choices=MODES, default="matrix")
    p.add_argument("--no-rain", action="store_true")
    args = p.parse_args()

    renderer = MatrixRenderer(
        args.width, args.height, args.cell, args.mode, not args.no_rain
    )
    if args.input:
        convert_file(args, renderer)
    else:
        run_camera(args, renderer)


if __name__ == "__main__":
    main()
