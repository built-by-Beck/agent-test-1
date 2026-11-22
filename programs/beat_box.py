#!/usr/bin/env python3
"""
beat_box.py - a step-sequencer drum machine that synthesises its own sounds.

Nothing is sampled. Every drum is built from sine sweeps and shaped noise at
start-up, laid onto a 16-step grid, and played as a looping bar you edit while
it runs.

Install:   pip install sounddevice numpy
Play:      python beat_box.py
Straight to a file (no sound card needed):
           python beat_box.py --preset boombap --export beat.wav --bars 8
Devices:   python beat_box.py --list-devices

Commands (type one and press Enter):
    k 1 5 9 13     toggle those steps on a track, 1-16
                   k kick  s snare  h hat  o openhat  c clap  t tom  b bass
    k              toggle every step on that track off/on
    tempo 96       beats per minute          swing 0.25   shuffle 16ths
    note C2        root note for the bass    vol k 0.8    track volume
    preset trap    four, boombap, trap, break, house, empty
    clear          wipe the grid             clear k      wipe one track
    grid           reprint the pattern       play / stop
    export out.wav 4      save 4 bars        save p.json / load p.json
    q              quit
"""

import argparse
import json
import sys
import wave

import numpy as np

STEPS = 16
TRACKS = {  # letter: display name
    "k": "kick",
    "s": "snare",
    "h": "hat",
    "o": "openhat",
    "c": "clap",
    "t": "tom",
    "b": "bass",
}
NOTE_BASE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

PRESETS = {
    "four": {
        "k": [1, 5, 9, 13],
        "h": [3, 7, 11, 15],
        "s": [5, 13],
        "b": [1, 9],
    },
    "boombap": {
        "k": [1, 7, 11],
        "s": [5, 13],
        "h": [1, 3, 5, 7, 9, 11, 13, 15],
        "b": [1, 11],
    },
    "trap": {
        "k": [1, 7, 8, 11],
        "s": [9],
        "h": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16],
        "b": [1, 7, 11],
    },
    "break": {
        "k": [1, 11],
        "s": [5, 13, 16],
        "h": [1, 3, 5, 7, 9, 11, 13, 15],
        "o": [8],
        "t": [15],
    },
    "house": {
        "k": [1, 5, 9, 13],
        "o": [3, 7, 11, 15],
        "c": [5, 13],
        "b": [1, 4, 9, 12],
    },
    "empty": {},
}


# --------------------------------------------------------------------------
# Synthesis.  Each voice is rendered once at start-up and reused.
# --------------------------------------------------------------------------


def envelope(n, attack, decay, curve=4.0):
    """Short rise then an exponential fall, as a float32 curve of length n."""
    a = max(1, int(n * attack))
    d = max(1, n - a)
    rise = np.linspace(0.0, 1.0, a, dtype=np.float32)
    fall = np.exp(-curve * np.linspace(0.0, 1.0, d)).astype(np.float32)
    return np.concatenate([rise, fall * (1.0 - decay * 0.0)])[:n]


def noise(sr, dur, rng):
    return rng.standard_normal(int(sr * dur)).astype(np.float32)


def highpass(x):
    """Cheap one-pole-ish high pass: the difference between samples."""
    return np.diff(x, prepend=x[:1]).astype(np.float32)


def lowpass(x, k):
    """Box smoothing, k samples wide. Bigger k = duller."""
    if k < 2:
        return x
    kernel = np.ones(k, np.float32) / k
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def make_kick(sr, rng):
    dur = 0.55
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    sweep = 45.0 + 105.0 * np.exp(-28.0 * t)  # 150 Hz down to 45 Hz
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    body = np.sin(phase) * envelope(len(t), 0.002, 0, 6.0)
    click = noise(sr, 0.012, rng) * np.linspace(1, 0, int(sr * 0.012), dtype=np.float32)
    out = body.copy()
    out[: len(click)] += click * 0.35
    return np.tanh(out * 1.4).astype(np.float32)


def make_snare(sr, rng):
    dur = 0.22
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    tone = (np.sin(2 * np.pi * 190 * t) + np.sin(2 * np.pi * 278 * t)) * 0.4
    body = highpass(noise(sr, dur, rng)) * 0.9 + tone
    return (body * envelope(len(t), 0.001, 0, 11.0) * 0.55).astype(np.float32)


def make_hat(sr, rng, dur=0.055, tone=0.0):
    n = int(sr * dur)
    x = highpass(highpass(noise(sr, dur, rng)))
    if tone:
        t = np.arange(n, dtype=np.float32) / sr
        x = x + np.sin(2 * np.pi * tone * t) * 0.15
    return (x * envelope(n, 0.0008, 0, 9.0 if dur < 0.1 else 3.0) * 0.3).astype(
        np.float32
    )


def make_clap(sr, rng):
    dur = 0.3
    n = int(sr * dur)
    out = np.zeros(n, np.float32)
    burst = highpass(noise(sr, 0.03, rng))
    for i, gain in enumerate((0.9, 0.8, 1.0)):
        start = int(sr * 0.011 * i)
        out[start : start + len(burst)] += burst * gain
    tail = highpass(noise(sr, dur, rng)) * envelope(n, 0.02, 0, 7.0) * 0.35
    return ((out + tail) * envelope(n, 0.001, 0, 2.2) * 0.45).astype(np.float32)


def make_tom(sr, rng):
    dur = 0.35
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    sweep = 110.0 + 70.0 * np.exp(-14.0 * t)
    phase = 2 * np.pi * np.cumsum(sweep) / sr
    body = np.sin(phase) * envelope(len(t), 0.002, 0, 5.0)
    return (body * 0.7).astype(np.float32)


def make_bass(sr, freq):
    """A short filtered saw, used for every bass step at the chosen note."""
    dur = 0.28
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float32) / sr
    saw = 2.0 * ((t * freq) % 1.0) - 1.0
    body = lowpass(saw.astype(np.float32), max(2, int(sr / (freq * 7))))
    return (body * envelope(n, 0.004, 0, 5.0) * 0.5).astype(np.float32)


def normalise(x, peak):
    """Scale a voice so its loudest sample sits exactly at `peak`."""
    high = float(np.abs(x).max())
    return (x * (peak / high)).astype(np.float32) if high > 1e-6 else x


def note_to_freq(name, a4=440.0):
    """'C2', 'f#3', 'Bb1' -> frequency in Hz."""
    text = name.strip().lower()
    if not text or text[0] not in NOTE_BASE:
        raise ValueError(f"not a note name: {name}")
    semi = NOTE_BASE[text[0]]
    i = 1
    while i < len(text) and text[i] in "#b":
        semi += 1 if text[i] == "#" else -1
        i += 1
    octave = int(text[i:]) if text[i:] else 3
    return a4 * 2.0 ** ((semi - 9) / 12.0 + (octave - 4))


# --------------------------------------------------------------------------
# The kit and the pattern
# --------------------------------------------------------------------------


class Kit:
    # How loud each voice sits relative to the others, before track volumes.
    LEVELS = {"k": 0.95, "s": 0.70, "h": 0.32, "o": 0.28, "c": 0.50, "t": 0.62, "b": 0.6}

    def __init__(self, sr, root="c2", seed=7):
        self.sr = sr
        rng = np.random.default_rng(seed)
        raw = {
            "k": make_kick(sr, rng),
            "s": make_snare(sr, rng),
            "h": make_hat(sr, rng),
            "o": make_hat(sr, rng, dur=0.30, tone=0.0),
            "c": make_clap(sr, rng),
            "t": make_tom(sr, rng),
        }
        self.sounds = {k: normalise(v, self.LEVELS[k]) for k, v in raw.items()}
        self.volumes = {key: 1.0 for key in TRACKS}
        self.set_root(root)

    def set_root(self, name):
        self.root = name
        bass = make_bass(self.sr, note_to_freq(name))
        self.sounds["b"] = normalise(bass, self.LEVELS["b"])


class Pattern:
    def __init__(self):
        self.grid = {key: [False] * STEPS for key in TRACKS}
        self.tempo = 96.0
        self.swing = 0.0

    def toggle(self, track, steps):
        if not steps:  # bare track letter flips the whole row
            on = any(self.grid[track])
            self.grid[track] = [not on] * STEPS if not on else [False] * STEPS
            return
        for s in steps:
            if 1 <= s <= STEPS:
                self.grid[track][s - 1] = not self.grid[track][s - 1]

    def load_preset(self, name):
        self.grid = {key: [False] * STEPS for key in TRACKS}
        for track, steps in PRESETS[name].items():
            for s in steps:
                self.grid[track][s - 1] = True

    def clear(self, track=None):
        for key in TRACKS if track is None else [track]:
            self.grid[key] = [False] * STEPS

    def text(self, kit):
        head = "        " + "".join(
            f"{i + 1:<2}"[0] if (i + 1) % 4 == 1 else "." for i in range(STEPS)
        )
        lines = [f"  tempo {self.tempo:.0f}  swing {self.swing:.2f}  bass {kit.root}"]
        lines.append(head)
        for key, name in TRACKS.items():
            row = "".join("X" if on else "." for on in self.grid[key])
            lines.append(f"  {key} {name:<7}{row}")
        return "\n".join(lines)


def render(pattern, kit, bars=1):
    """Mix the whole pattern down to one float32 buffer."""
    sr = kit.sr
    step_s = 60.0 / pattern.tempo / 4.0
    bar_n = int(round(step_s * STEPS * sr))
    tail = max(len(s) for s in kit.sounds.values())
    buf = np.zeros(bar_n * bars + tail, np.float32)
    for bar in range(bars):
        for track in TRACKS:
            sound = kit.sounds[track]
            gain = kit.volumes[track]
            for step, on in enumerate(pattern.grid[track]):
                if not on:
                    continue
                offset = step_s * (pattern.swing * 0.5 if step % 2 else 0.0)
                start = int(round((bar * STEPS + step) * step_s * sr + offset * sr))
                buf[start : start + len(sound)] += sound * gain
    if bars == 1:
        # Looping bar: fold the ring-out back over the start so it joins up.
        head = buf[bar_n : bar_n + tail]
        buf = buf[:bar_n].copy()
        buf[: len(head)] += head[: len(buf)]
    else:
        buf = buf[: bar_n * bars + int(sr * 0.4)]
    return np.tanh(buf * 0.9).astype(np.float32)


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def handle(text, pattern, kit, state):
    """Run one typed command. Returns a message, or None to quit."""
    parts = text.split()
    if not parts:
        return ""
    cmd, args = parts[0].lower(), parts[1:]

    if cmd in ("q", "quit", "exit"):
        return None
    if cmd in TRACKS:
        steps = []
        for a in args:
            if not a.lstrip("-").isdigit():
                return f"'{a}' is not a step number."
            steps.append(int(a))
        pattern.toggle(cmd, steps)
        return pattern.text(kit)
    if cmd == "tempo":
        pattern.tempo = float(np.clip(float(args[0]), 40, 240))
        return f"tempo {pattern.tempo:.0f}"
    if cmd == "swing":
        pattern.swing = float(np.clip(float(args[0]), 0.0, 0.75))
        return f"swing {pattern.swing:.2f}"
    if cmd == "note":
        kit.set_root(args[0])
        return f"bass root {kit.root}"
    if cmd == "vol":
        track, amount = args[0], float(args[1])
        kit.volumes[track] = float(np.clip(amount, 0.0, 2.0))
        return f"{TRACKS[track]} volume {kit.volumes[track]:.2f}"
    if cmd == "preset":
        pattern.load_preset(args[0])
        return pattern.text(kit)
    if cmd == "clear":
        pattern.clear(args[0] if args else None)
        return pattern.text(kit)
    if cmd == "grid":
        return pattern.text(kit)
    if cmd in ("play", "stop"):
        state["playing"] = cmd == "play"
        return cmd
    if cmd == "export":
        path = args[0] if args else "beat.wav"
        bars = int(args[1]) if len(args) > 1 else 4
        write_wav(path, render(pattern, kit, bars), kit.sr)
        return f"wrote {path} ({bars} bars)"
    if cmd == "save":
        data = {
            "grid": pattern.grid,
            "tempo": pattern.tempo,
            "swing": pattern.swing,
            "root": kit.root,
            "volumes": kit.volumes,
        }
        with open(args[0], "w") as fh:
            json.dump(data, fh, indent=1)
        return f"saved {args[0]}"
    if cmd == "load":
        with open(args[0]) as fh:
            data = json.load(fh)
        for key in TRACKS:
            pattern.grid[key] = list(data["grid"].get(key, [False] * STEPS))
        pattern.tempo = data.get("tempo", pattern.tempo)
        pattern.swing = data.get("swing", pattern.swing)
        kit.volumes.update(data.get("volumes", {}))
        kit.set_root(data.get("root", kit.root))
        return pattern.text(kit)
    return f"Unknown command '{cmd}'. Type grid, preset four, tempo 100, or q."


def run_live(args, pattern, kit):
    import sounddevice as sd  # imported here so --export needs no sound card

    if args.list_devices:
        print(sd.query_devices())
        return

    state = {"playing": True}
    loop = {"buf": render(pattern, kit), "pos": 0}

    def callback(outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        buf = loop["buf"]
        if not state["playing"] or len(buf) == 0:
            outdata[:] = 0
            return
        pos = loop["pos"]
        idx = (pos + np.arange(frames)) % len(buf)
        outdata[:, 0] = buf[idx]
        loop["pos"] = (pos + frames) % len(buf)

    print(__doc__)
    print(pattern.text(kit))
    try:
        with sd.OutputStream(
            samplerate=kit.sr,
            blocksize=args.block,
            channels=1,
            dtype="float32",
            device=args.out_device,
            callback=callback,
        ):
            while True:
                try:
                    text = input("beat> ")
                except (EOFError, KeyboardInterrupt):
                    break
                before = (pattern.tempo, pattern.swing)
                msg = handle(text, pattern, kit, state)
                if msg is None:
                    break
                if msg:
                    print(msg)
                # Re-render the loop, keeping our place in the bar.
                fresh = render(pattern, kit)
                where = loop["pos"] / max(1, len(loop["buf"]))
                loop["buf"] = fresh
                if before != (pattern.tempo, pattern.swing):
                    loop["pos"] = int(where * len(fresh)) % max(1, len(fresh))
    except Exception as exc:
        raise SystemExit(f"Could not open the sound card: {exc}")


def main():
    p = argparse.ArgumentParser(description="Step-sequencer drum machine")
    p.add_argument("--preset", choices=list(PRESETS), default="boombap")
    p.add_argument("--tempo", type=float, default=96.0)
    p.add_argument("--swing", type=float, default=0.0)
    p.add_argument("--note", default="c2", help="bass root note, e.g. c2, f#1")
    p.add_argument("--export", metavar="WAV", help="render to a file and stop")
    p.add_argument("--bars", type=int, default=4, help="bars for --export")
    p.add_argument("--rate", type=int, default=44100)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--out-device", type=int)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    kit = Kit(args.rate, args.note)
    pattern = Pattern()
    pattern.load_preset(args.preset)
    pattern.tempo = args.tempo
    pattern.swing = args.swing

    if args.export:
        write_wav(args.export, render(pattern, kit, args.bars), kit.sr)
        print(f"Wrote {args.export} ({args.bars} bars at {pattern.tempo:.0f} bpm)")
        return
    run_live(args, pattern, kit)


if __name__ == "__main__":
    main()
