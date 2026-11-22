#!/usr/bin/env python3
"""
screen_fx.py - record your screen through a live chain of video effects.

Grabs the desktop (a whole monitor, a region, or a zoomed box that follows the
mouse), pushes every frame through an effects chain you switch on and off while
it runs, shows a live preview, and records to MP4 or GIF.

Install:   pip install mss opencv-python numpy pillow
Run:       python screen_fx.py
Monitors:  python screen_fx.py --list-monitors
Region:    python screen_fx.py --region 100,100,1280,720
Convert:   python screen_fx.py --input clip.mp4 --output clip_fx.mp4 --preset vhs

Mirror warning: the preview window is itself on screen, so pointing the capture
at the monitor showing it gives you the endless tunnel effect. Capture a region
that excludes the preview, use a second monitor, or run with --no-preview
(which has no window and no keys, so it records straight away until Ctrl+C).

Keys (click the preview window first):
    1-9    toggle an effect (the HUD lists which is which)
    - =    less / more of the effect you last toggled
    p      cycle presets      c  clear all effects
    z      cursor zoom on/off , .  zoom out / in
    r      start / stop recording MP4
    g      save the last few seconds as a GIF
    s      save a PNG snapshot
    h      hide / show the HUD
    q/Esc  quit
"""

import argparse
import ctypes
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BAYER8 = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        np.float32,
    )
    + 0.5
) / 64.0


# --------------------------------------------------------------------------
# Effects.  Each one owns an amount (0..1) and is toggled by its own key.
# --------------------------------------------------------------------------


class Effect:
    name = "effect"

    def __init__(self, amount=0.5):
        self.amount = float(amount)
        self.on = False

    def nudge(self, delta):
        self.amount = float(np.clip(self.amount + delta, 0.05, 1.0))

    def apply(self, img, t, rng):
        return img


class Pixelate(Effect):
    name = "pixelate"

    def apply(self, img, t, rng):
        h, w = img.shape[:2]
        block = 1 + int(self.amount * 28)
        if block < 2:
            return img
        small = cv2.resize(
            img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA
        )
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class PixelSort(Effect):
    """Sort runs of pixels in each row by brightness, within a threshold band."""

    name = "pixelsort"

    def apply(self, img, t, rng):
        h, w = img.shape[:2]
        # Sorting is the slowest effect here, so it runs at half height.
        work = cv2.resize(img, (w, h // 2), interpolation=cv2.INTER_AREA)
        luma = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        lo = int(40 + (1.0 - self.amount) * 110)
        mask = luma > lo
        rows = np.nonzero(mask.any(axis=1))[0]
        for y in rows:
            cols = np.nonzero(mask[y])[0]
            if len(cols) < 3:
                continue
            order = cols[np.argsort(luma[y, cols], kind="stable")]
            work[y, cols] = work[y, order]
        return cv2.resize(work, (w, h), interpolation=cv2.INTER_LINEAR)


class Outline(Effect):
    name = "outline"

    def apply(self, img, t, rng):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lo = int(140 - self.amount * 110)
        edges = cv2.Canny(gray, lo, lo * 2)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        glow = np.zeros_like(img)
        glow[:, :, 1] = edges  # wireframe green, with a touch of cyan
        glow[:, :, 0] = edges // 3
        keep = 1.0 - 0.55 * self.amount
        return cv2.addWeighted(img, keep, glow, 1.0, 0)


class Posterize(Effect):
    """Quantise the colours, with an ordered dither so bands break up."""

    name = "dither"

    def __init__(self, amount=0.5):
        super().__init__(amount)
        self._tile = None

    def apply(self, img, t, rng):
        h, w = img.shape[:2]
        if self._tile is None or self._tile.shape[:2] != (h, w):
            self._tile = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w, None].copy()
        levels = max(2, int(10 - self.amount * 8))
        step = 255.0 / (levels - 1)
        out = np.floor(img.astype(np.float32) / step + self._tile) * step
        return np.clip(out, 0, 255).astype(np.uint8)


class Bloom(Effect):
    name = "bloom"

    def apply(self, img, t, rng):
        thresh = int(255 - self.amount * 150)
        bright = cv2.threshold(img, thresh, 255, cv2.THRESH_TOZERO)[1]
        k = 1 + 2 * int(4 + self.amount * 14)
        blur = cv2.GaussianBlur(bright, (k, k), 0)
        return cv2.addWeighted(img, 1.0, blur, 0.55 + self.amount, 0)


class RGBSplit(Effect):
    name = "rgbsplit"

    def apply(self, img, t, rng):
        shift = 1 + int(self.amount * 18)
        wob = int(np.sin(t * 2.3) * shift * 0.4)
        b, g, r = cv2.split(img)
        b = np.roll(b, -shift, axis=1)
        r = np.roll(r, shift, axis=1)
        g = np.roll(g, wob, axis=0)
        return cv2.merge((b, g, r))


class VHS(Effect):
    """Wavy rows, chroma bleed, noise, and a tracking band drifting up."""

    name = "vhs"

    def __init__(self, amount=0.5):
        super().__init__(amount)
        self._cache = None

    def _grids(self, h, w):
        """Row numbers and the base column grid, built once per frame size."""
        if self._cache is None or self._cache[0] != (h, w):
            rows = np.arange(h, dtype=np.float32)
            cols = np.tile(np.arange(w, dtype=np.float32), (h, 1))
            map_y = np.repeat(rows[:, None], w, axis=1)
            self._cache = ((h, w), rows, cols, map_y)
        return self._cache[1:]

    def apply(self, img, t, rng):
        h, w = img.shape[:2]
        rows, cols, map_y = self._grids(h, w)
        amp = self.amount * 9.0
        wobble = (
            np.sin(rows * 0.06 + t * 5.0) * amp
            + np.sin(rows * 0.015 - t * 1.7) * amp * 0.6
        )
        map_x = cols + wobble[:, None].astype(np.float32)
        out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        k = 1 + 2 * int(1 + self.amount * 6)
        ycc[:, :, 1] = cv2.blur(ycc[:, :, 1], (k, 1))
        ycc[:, :, 2] = np.roll(cv2.blur(ycc[:, :, 2], (k, 1)), 3, axis=1)
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

        band_h = max(4, int(h * 0.04))
        y0 = int((-t * 140) % (h + band_h)) - band_h
        y0 = max(0, min(h - band_h, y0))
        band = out[y0 : y0 + band_h]
        noise = rng.integers(0, 70, band.shape, dtype=np.int16)
        out[y0 : y0 + band_h] = np.clip(band.astype(np.int16) + noise, 0, 255).astype(
            np.uint8
        )

        # Grain, drawn small and blown up so it reads as chunky tape noise.
        span = max(1, int(14 * self.amount))
        small = rng.integers(0, 2 * span + 1, (h // 3 + 1, w // 3 + 1), dtype=np.uint8)
        grain = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        grain = cv2.cvtColor(grain, cv2.COLOR_GRAY2BGR)
        return cv2.subtract(cv2.add(out, grain), (span, span, span, 0))


class Glitch(Effect):
    """Slabs of the picture torn sideways, with their colours knocked loose."""

    name = "glitch"

    def apply(self, img, t, rng):
        h, w = img.shape[:2]
        out = img.copy()
        slabs = 1 + int(self.amount * 16)
        for _ in range(slabs):
            sh = rng.integers(4, max(6, int(h * 0.09)))
            y0 = int(rng.integers(0, max(1, h - sh)))
            dx = int(rng.integers(-1, 2) * rng.integers(5, int(20 + self.amount * 110)))
            slab = np.roll(out[y0 : y0 + sh], dx, axis=1)
            if rng.random() < 0.45:
                ch = int(rng.integers(0, 3))
                slab[:, :, ch] = np.roll(slab[:, :, ch], int(rng.integers(-24, 25)), 1)
            out[y0 : y0 + sh] = slab
        return out


class CRT(Effect):
    """Scanlines, a vignette, and a gentle bulge like a glass tube."""

    name = "crt"

    def __init__(self, amount=0.5):
        super().__init__(amount)
        self._cache = None

    def _maps(self, h, w):
        key = (h, w, round(self.amount, 2))
        if self._cache and self._cache[0] == key:
            return self._cache[1]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / (w - 1)) * 2 - 1
        ny = (yy / (h - 1)) * 2 - 1
        r2 = nx * nx + ny * ny
        bulge = 1.0 + 0.12 * self.amount * r2
        map_x = ((nx * bulge + 1) / 2 * (w - 1)).astype(np.float32)
        map_y = ((ny * bulge + 1) / 2 * (h - 1)).astype(np.float32)
        vignette = np.clip(1.15 - 0.55 * self.amount * r2, 0, 1).astype(np.float32)
        lines = np.ones((h, 1), np.float32)
        lines[::2] = 1.0 - 0.55 * self.amount
        self._cache = (key, (map_x, map_y, vignette[:, :, None] * lines[:, :, None]))
        return self._cache[1]

    def apply(self, img, t, rng):
        h, w = img.shape[:2]
        map_x, map_y, shade = self._maps(h, w)
        out = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        return np.clip(out.astype(np.float32) * shade, 0, 255).astype(np.uint8)


def build_chain():
    """The chain in running order; list position is the hotkey (1-9)."""
    return [
        Pixelate(0.35),
        PixelSort(0.5),
        Outline(0.6),
        Posterize(0.5),
        Bloom(0.5),
        RGBSplit(0.4),
        VHS(0.5),
        Glitch(0.4),
        CRT(0.55),
    ]


PRESETS = {
    "clean": {},
    "vhs": {"vhs": 0.6, "rgbsplit": 0.3, "crt": 0.4},
    "crt": {"crt": 0.8, "bloom": 0.4},
    "hacker": {"outline": 0.7, "bloom": 0.6, "crt": 0.35},
    "broken": {"glitch": 0.6, "rgbsplit": 0.6, "vhs": 0.4},
    "lofi": {"pixelate": 0.4, "dither": 0.6},
    "melt": {"pixelsort": 0.6, "bloom": 0.3},
}
PRESET_NAMES = list(PRESETS)


def apply_preset(chain, name):
    wanted = PRESETS[name]
    for fx in chain:
        fx.on = fx.name in wanted
        if fx.on:
            fx.amount = wanted[fx.name]


# --------------------------------------------------------------------------
# Screen capture
# --------------------------------------------------------------------------


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def cursor_pos():
    """Mouse position in desktop coordinates, or None if we can't tell."""
    if sys.platform == "win32":
        pt = _Point()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return pt.x, pt.y
        return None
    try:
        from Xlib import display  # optional; only used for cursor zoom on Linux

        p = display.Display().screen().root.query_pointer()
        return p.root_x, p.root_y
    except Exception:
        return None


def open_mss():
    """Start screen capture, with a readable message when there's no desktop."""
    try:
        import mss  # imported here so --input needs no screen at all

        # Newer mss spells it MSS; older releases only have mss().
        return (getattr(mss, "MSS", None) or mss.mss)()
    except Exception as exc:
        raise SystemExit(
            f"Could not start screen capture: {exc}\n"
            "This needs a desktop session, and `pip install mss`."
        )


class ScreenGrabber:
    """Pulls frames from a monitor or a fixed region, with optional mouse zoom."""

    def __init__(self, monitor=1, region=None, scale=1.0):
        self.sct = open_mss()
        mons = self.sct.monitors
        if region:
            x, y, w, h = region
            self.area = {"left": x, "top": y, "width": w, "height": h}
        else:
            if not 0 <= monitor < len(mons):
                raise SystemExit(
                    f"No monitor {monitor}. Run --list-monitors to see them."
                )
            self.area = dict(mons[monitor])
        self.zoom_on = False
        self.zoom = 2.0
        w = int(self.area["width"] * scale) // 2 * 2
        h = int(self.area["height"] * scale) // 2 * 2
        self.size = (max(2, w), max(2, h))

    def grab(self):
        area = self.area
        if self.zoom_on:
            pos = cursor_pos()
            zw = max(16, int(area["width"] / self.zoom))
            zh = max(16, int(area["height"] / self.zoom))
            if pos is None:
                cx = area["left"] + area["width"] // 2
                cy = area["top"] + area["height"] // 2
            else:
                cx, cy = pos
            left = int(np.clip(cx - zw // 2, area["left"], area["left"] + area["width"] - zw))
            top = int(np.clip(cy - zh // 2, area["top"], area["top"] + area["height"] - zh))
            area = {"left": left, "top": top, "width": zw, "height": zh}
        shot = np.asarray(self.sct.grab(area))[:, :, :3]
        if (shot.shape[1], shot.shape[0]) != self.size:
            shot = cv2.resize(shot, self.size, interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(shot)

    def close(self):
        self.sct.close()


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_writer(path, fps, size):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def save_gif(frames, path, fps):
    from PIL import Image  # only needed when you actually press g

    if not frames:
        print("Nothing buffered for a GIF yet.")
        return
    h, w = frames[0].shape[:2]
    scale = min(1.0, 640.0 / w)
    pics = [
        Image.fromarray(
            cv2.cvtColor(
                cv2.resize(f, (int(w * scale), int(h * scale)), cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
        ).quantize(colors=128)
        for f in frames
    ]
    pics[0].save(
        path,
        save_all=True,
        append_images=pics[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"GIF -> {path}  ({len(pics)} frames)")


def draw_hud(img, chain, preset, fps, recording, grabber):
    pad = np.zeros((img.shape[0], 250, 3), np.uint8)
    view = np.hstack([img, pad])
    x = img.shape[1] + 12
    cv2.putText(view, f"{fps:5.1f} fps", (x, 28), FONT, 0.6, (200, 255, 200), 1)
    cv2.putText(view, f"preset: {preset}", (x, 52), FONT, 0.5, (160, 200, 255), 1)
    zoom = f"zoom x{grabber.zoom:.1f}" if grabber.zoom_on else "zoom off"
    cv2.putText(view, zoom, (x, 74), FONT, 0.5, (160, 200, 255), 1)
    for i, fx in enumerate(chain):
        y = 106 + i * 24
        colour = (90, 255, 140) if fx.on else (110, 110, 110)
        bar = "#" * int(fx.amount * 10) if fx.on else ""
        cv2.putText(view, f"{i + 1} {fx.name:9s}{bar}", (x, y), FONT, 0.45, colour, 1)
    if recording:
        cv2.circle(view, (26, 26), 9, (0, 0, 255), -1)
        cv2.putText(view, "REC", (44, 33), FONT, 0.7, (0, 0, 255), 2)
    return view


# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------


def run_chain(chain, frame, t, rng):
    for fx in chain:
        if fx.on:
            frame = fx.apply(frame, t, rng)
    return frame


def convert_file(args, chain):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale) // 2 * 2
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale) // 2 * 2
    out = args.output or f"fx_{stamp()}.mp4"
    writer = open_writer(out, fps, (w, h))
    rng = np.random.default_rng()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != (w, h):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(run_chain(chain, frame, n / fps, rng))
        n += 1
    writer.release()
    cap.release()
    print(f"Wrote {n} frames to {out}")


def run_live(args, chain):
    grabber = ScreenGrabber(args.monitor, args.region, args.scale)
    rng = np.random.default_rng()
    preset = args.preset
    gif_buf = deque(maxlen=int(args.gif_seconds * args.gif_fps))
    gif_next = 0.0
    writer, rec_start, rec_frames, rec_path = None, 0.0, 0, ""
    selected = 0
    show_hud = not args.no_hud
    fps_avg = 0.0
    last = time.time()

    win = "Screen FX  (1-9 effects  p preset  r record  g gif  q quit)"
    if not args.no_preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1600, grabber.size[0] + 250), grabber.size[1])
    print(__doc__)
    print(f"Capturing {grabber.size[0]}x{grabber.size[1]}  presets: {PRESET_NAMES}")

    if args.no_preview:
        # With no window there are no keys, so just record until Ctrl+C.
        rec_path = args.output or f"screen_{stamp()}.mp4"
        writer = open_writer(rec_path, args.fps, grabber.size)
        rec_start, rec_frames = time.time(), 0
        print(f"Recording -> {rec_path}   (Ctrl+C to stop)")

    try:
        while True:
            now = time.time()
            frame = run_chain(chain, grabber.grab(), now, rng)

            dt = now - last
            last = now
            if dt > 0:
                fps_avg = fps_avg * 0.9 + (1.0 / dt) * 0.1

            if writer is not None:
                due = int((now - rec_start) * args.fps) + 1
                while rec_frames < due:
                    writer.write(frame)
                    rec_frames += 1
            if now >= gif_next:
                gif_buf.append(frame.copy())
                gif_next = now + 1.0 / args.gif_fps

            if args.no_preview:
                time.sleep(0.001)
                continue

            view = frame
            if show_hud:
                view = draw_hud(
                    frame, chain, preset, fps_avg, writer is not None, grabber
                )
            cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                pass
            elif key in (ord("q"), 27):
                break
            elif ord("1") <= key <= ord("9"):
                i = key - ord("1")
                if i < len(chain):
                    chain[i].on = not chain[i].on
                    selected = i
                    preset = "custom"
            elif key in (ord("="), ord("+")):
                chain[selected].nudge(0.1)
            elif key in (ord("-"), ord("_")):
                chain[selected].nudge(-0.1)
            elif key == ord("p"):
                i = PRESET_NAMES.index(preset) if preset in PRESET_NAMES else -1
                preset = PRESET_NAMES[(i + 1) % len(PRESET_NAMES)]
                apply_preset(chain, preset)
            elif key == ord("c"):
                for fx in chain:
                    fx.on = False
                preset = "clean"
            elif key == ord("z"):
                grabber.zoom_on = not grabber.zoom_on
            elif key == ord(","):
                grabber.zoom = max(1.2, grabber.zoom - 0.2)
            elif key == ord("."):
                grabber.zoom = min(8.0, grabber.zoom + 0.2)
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("s"):
                name = f"screen_{stamp()}.png"
                cv2.imwrite(name, frame)
                print(f"Snapshot -> {name}")
            elif key == ord("g"):
                save_gif(list(gif_buf), f"screen_{stamp()}.gif", args.gif_fps)
            elif key == ord("r"):
                if writer is None:
                    rec_path = args.output or f"screen_{stamp()}.mp4"
                    writer = open_writer(rec_path, args.fps, grabber.size)
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
        grabber.close()
        cv2.destroyAllWindows()


def list_monitors():
    with open_mss() as sct:
        for i, m in enumerate(sct.monitors):
            tag = "all monitors" if i == 0 else f"monitor {i}"
            print(f"  {i}: {tag:14s} {m['width']}x{m['height']} at {m['left']},{m['top']}")


def parse_region(text):
    try:
        x, y, w, h = (int(v) for v in text.split(","))
        return x, y, w, h
    except ValueError:
        raise argparse.ArgumentTypeError("region must be x,y,width,height")


def main():
    p = argparse.ArgumentParser(description="Screen recorder with live effects")
    p.add_argument("--monitor", type=int, default=1, help="0 = all, 1 = first")
    p.add_argument("--region", type=parse_region, help="x,y,width,height")
    p.add_argument("--scale", type=float, default=1.0, help="shrink for speed")
    p.add_argument("--preset", choices=PRESET_NAMES, default="clean")
    p.add_argument("--input", help="apply the chain to a video file instead")
    p.add_argument("--output", help="output path (default: timestamped)")
    p.add_argument("--fps", type=float, default=30.0, help="recording frame rate")
    p.add_argument("--gif-seconds", type=float, default=6.0)
    p.add_argument("--gif-fps", type=float, default=12.0)
    p.add_argument("--no-preview", action="store_true", help="record with no window")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--list-monitors", action="store_true")
    args = p.parse_args()

    if args.list_monitors:
        list_monitors()
        return
    chain = build_chain()
    apply_preset(chain, args.preset)
    if args.input:
        convert_file(args, chain)
    else:
        run_live(args, chain)


if __name__ == "__main__":
    main()
