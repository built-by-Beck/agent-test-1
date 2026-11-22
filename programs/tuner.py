#!/usr/bin/env python3
"""
tuner.py - an instrument tuner and metronome that listens through the mic.

The tuner works out the pitch of whatever you play by autocorrelation, names
the note, and shows how many cents sharp or flat you are on a moving needle.
The metronome generates its own clicks with an accent on beat one.

Install:   pip install sounddevice numpy
Tune:      python tuner.py --instrument guitar
Click:     python tuner.py --mode metro --tempo 90 --sig 3/4
Both:      python tuner.py --mode both --tempo 100
Analyse:   python tuner.py --input riff.wav
Devices:   python tuner.py --list-devices

Tuner modes draw a live line and quit on Ctrl+C.  Metronome mode takes typed
commands: tempo 120, sig 6/8, start, stop, q.

In --mode both the mic hears the clicks, which can confuse the needle for an
instant. Use headphones if that bothers you.
"""

import argparse
import sys
import time
import wave

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
INSTRUMENTS = {
    "guitar": ["E2", "A2", "D3", "G3", "B3", "E4"],
    "dropd": ["D2", "A2", "D3", "G3", "B3", "E4"],
    "bass": ["E1", "A1", "D2", "G2"],
    "ukulele": ["G4", "C4", "E4", "A4"],
    "violin": ["G3", "D4", "A4", "E5"],
    "cello": ["C2", "G2", "D3", "A3"],
    "chromatic": [],
}
NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# --------------------------------------------------------------------------
# Notes and pitch
# --------------------------------------------------------------------------


def note_to_midi(name):
    """'E2', 'F#3', 'Bb1' -> MIDI number."""
    text = name.strip()
    head = text[0].upper()
    if head not in NOTE_BASE:
        raise ValueError(f"not a note name: {name}")
    semi = NOTE_BASE[head]
    i = 1
    while i < len(text) and text[i] in "#b":
        semi += 1 if text[i] == "#" else -1
        i += 1
    return semi + (int(text[i:]) + 1) * 12


def midi_to_freq(midi, a4=440.0):
    return a4 * 2.0 ** ((midi - 69) / 12.0)


def freq_to_note(freq, a4=440.0):
    """Returns (name, cents away from that name, exact midi value)."""
    midi = 69.0 + 12.0 * np.log2(freq / a4)
    nearest = int(round(midi))
    cents = (midi - nearest) * 100.0
    name = f"{NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}"
    return name, cents, midi


def detect_pitch(block, sr, fmin=28.0, fmax=1400.0):
    """Autocorrelation pitch detection. Returns (freq or None, confidence)."""
    x = block.astype(np.float32)
    x = x - x.mean()
    if float(np.sqrt((x * x).mean())) < 2e-3:
        return None, 0.0  # too quiet to bother

    n = len(x)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x * np.hanning(n).astype(np.float32), size)
    corr = np.fft.irfft(spec * np.conj(spec))[:n]
    if corr[0] <= 0:
        return None, 0.0
    corr = corr / corr[0]

    # A low note still correlates almost perfectly with itself a few samples
    # along, so the global maximum would just be the shortest lag. Start
    # looking only once the correlation has dipped below zero.
    negative = np.nonzero(corr < 0)[0]
    first_dip = int(negative[0]) if len(negative) else 2

    lo = max(2, first_dip, int(sr / fmax))
    hi = min(n - 2, int(sr / fmin))
    if hi - lo < 4:
        return None, 0.0
    peak = int(np.argmax(corr[lo:hi])) + lo
    conf = float(corr[peak])
    if conf < 0.30:
        return None, conf  # not periodic enough to call it a note

    # Autocorrelation loves octave-down errors: if half the lag is nearly as
    # strong, that shorter period is the real one.
    half = peak // 2
    if half >= lo and corr[half] > conf * 0.82:
        peak, conf = half, float(corr[half])

    a, b, c = corr[peak - 1], corr[peak], corr[peak + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    period = peak + float(np.clip(shift, -1, 1))
    return (sr / period if period > 0 else None), conf


def nearest_string(midi, strings):
    """Closest open string to a midi value, as (name, cents off it)."""
    if not strings:
        return None, 0.0
    targets = [(name, note_to_midi(name)) for name in strings]
    name, target = min(targets, key=lambda nm: abs(nm[1] - midi))
    return name, (midi - target) * 100.0


def needle(cents, width=21):
    """A text needle: centre is in tune, left is flat, right is sharp."""
    mid = width // 2
    pos = int(round(mid + np.clip(cents, -50, 50) / 50.0 * mid))
    bar = ["-"] * width
    bar[mid] = "|"
    bar[int(np.clip(pos, 0, width - 1))] = "#"
    return "".join(bar)


def verdict(cents):
    if abs(cents) < 5:
        return "in tune"
    return ("flat " if cents < 0 else "sharp ") + f"{abs(cents):4.0f}c"


# --------------------------------------------------------------------------
# Metronome
# --------------------------------------------------------------------------


def click(sr, freq, dur=0.035):
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    env = np.exp(-32.0 * t).astype(np.float32)
    return (np.sin(2 * np.pi * freq * t) * env * 0.45).astype(np.float32)


def metro_bar(sr, tempo, beats, unit):
    """One bar of clicks: accented first beat, plain ones after."""
    beat_s = 60.0 / tempo * (4.0 / unit)
    bar_n = max(1, int(round(beat_s * beats * sr)))
    buf = np.zeros(bar_n + sr // 4, np.float32)
    accent, plain = click(sr, 1320.0), click(sr, 880.0)
    for b in range(beats):
        tick = accent if b == 0 else plain
        start = int(round(b * beat_s * sr))
        buf[start : start + len(tick)] += tick
    head = buf[bar_n:]
    out = buf[:bar_n].copy()
    out[: len(head)] += head[: len(out)]
    return out


def parse_sig(text):
    try:
        beats, unit = (int(v) for v in text.split("/"))
        if beats < 1 or unit not in (2, 4, 8, 16):
            raise ValueError
        return beats, unit
    except ValueError:
        raise argparse.ArgumentTypeError("time signature looks like 4/4 or 6/8")


# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------


def analyse_file(args):
    with wave.open(args.input, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("Only 16-bit WAV files are supported for --input.")
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    audio = (data.reshape(-1, ch).mean(axis=1) / 32768.0).astype(np.float32)
    size = args.window
    print(f"{args.input}: {len(audio) / sr:.2f}s at {sr} Hz")
    print(f"{'time':>7}  {'freq':>8}  {'note':<5} {'cents':>6}  conf")
    for start in range(0, len(audio) - size, size // 2):
        freq, conf = detect_pitch(audio[start : start + size], sr)
        if freq is None:
            continue
        name, cents, _ = freq_to_note(freq, args.a4)
        print(
            f"{start / sr:7.2f}  {freq:8.2f}  {name:<5} {cents:+6.1f}  {conf:.2f}"
        )


def run_tuner(args, sd, strings):
    latest = {"block": None}

    def on_audio(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        latest["block"] = indata[:, 0].copy()

    held = {"name": "", "since": 0.0}
    stream = sd.InputStream(
        samplerate=args.rate,
        blocksize=args.window,
        channels=1,
        dtype="float32",
        device=args.in_device,
        callback=on_audio,
    )
    with stream:
        while True:
            block = latest["block"]
            if block is None:
                time.sleep(0.03)
                continue
            freq, conf = detect_pitch(block, args.rate)
            if freq is None:
                line = f"  {'--':>9}  listening... {' ' * 34}"
            else:
                name, cents, midi = freq_to_note(freq, args.a4)
                target, off = nearest_string(midi, strings)
                if target:
                    name, cents = target, off
                # Only call it stable once the same note has held for a moment.
                now = time.time()
                if name != held["name"]:
                    held["name"], held["since"] = name, now
                steady = "*" if now - held["since"] > 0.4 else " "
                line = (
                    f"  {freq:7.2f}Hz  {name:<4}{steady} "
                    f"[{needle(cents)}] {verdict(cents):<11} conf {conf:.2f}"
                )
            print(f"\r{line}", end="", flush=True)
            time.sleep(0.05)


def run_metro(args, sd):
    sr = args.rate
    beats, unit = args.sig
    state = {"on": True, "tempo": args.tempo, "sig": (beats, unit)}
    loop = {"buf": metro_bar(sr, args.tempo, beats, unit), "pos": 0}

    def on_out(outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        buf = loop["buf"]
        if not state["on"] or len(buf) == 0:
            outdata[:] = 0
            return
        idx = (loop["pos"] + np.arange(frames)) % len(buf)
        outdata[:, 0] = buf[idx]
        loop["pos"] = (loop["pos"] + frames) % len(buf)

    stream = sd.OutputStream(
        samplerate=sr,
        blocksize=args.block,
        channels=1,
        dtype="float32",
        device=args.out_device,
        callback=on_out,
    )
    with stream:
        print(f"Metronome: {state['tempo']:.0f} bpm, {beats}/{unit}.")
        print("Commands: tempo 120, sig 6/8, start, stop, q")
        while True:
            try:
                parts = input("metro> ").split()
            except (EOFError, KeyboardInterrupt):
                break
            if not parts:
                continue
            cmd, rest = parts[0].lower(), parts[1:]
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "tempo" and rest:
                state["tempo"] = float(np.clip(float(rest[0]), 30, 300))
            elif cmd == "sig" and rest:
                try:
                    state["sig"] = parse_sig(rest[0])
                except argparse.ArgumentTypeError as exc:
                    print(exc)
                    continue
            elif cmd in ("start", "stop"):
                state["on"] = cmd == "start"
                print(cmd)
                continue
            else:
                print("Try: tempo 120, sig 6/8, start, stop, q")
                continue
            loop["buf"] = metro_bar(sr, state["tempo"], *state["sig"])
            loop["pos"] = 0
            print(f"{state['tempo']:.0f} bpm, {state['sig'][0]}/{state['sig'][1]}")


def run_live(args):
    import sounddevice as sd  # imported here so --input needs no sound card

    if args.list_devices:
        print(sd.query_devices())
        return
    strings = INSTRUMENTS[args.instrument]
    if args.mode in ("tuner", "both") and strings:
        print(f"{args.instrument}: {' '.join(strings)}   (A4 = {args.a4:.1f} Hz)")

    try:
        if args.mode == "metro":
            run_metro(args, sd)
            return
        if args.mode == "both":
            beats, unit = args.sig
            sr = args.rate
            loop = {"buf": metro_bar(sr, args.tempo, beats, unit), "pos": 0}

            def on_out(outdata, frames, time_info, status):
                buf = loop["buf"]
                idx = (loop["pos"] + np.arange(frames)) % len(buf)
                outdata[:, 0] = buf[idx]
                loop["pos"] = (loop["pos"] + frames) % len(buf)

            with sd.OutputStream(
                samplerate=sr,
                blocksize=args.block,
                channels=1,
                dtype="float32",
                device=args.out_device,
                callback=on_out,
            ):
                print(f"Metronome at {args.tempo:.0f} bpm. Ctrl+C to stop.")
                run_tuner(args, sd, strings)
        else:
            print("Play a note. Ctrl+C to stop.")
            run_tuner(args, sd, strings)
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        raise SystemExit(f"Could not open the sound device: {exc}")


def main():
    p = argparse.ArgumentParser(description="Instrument tuner and metronome")
    p.add_argument("--mode", choices=["tuner", "metro", "both"], default="tuner")
    p.add_argument("--instrument", choices=list(INSTRUMENTS), default="chromatic")
    p.add_argument("--a4", type=float, default=440.0, help="reference pitch")
    p.add_argument("--tempo", type=float, default=100.0)
    p.add_argument("--sig", type=parse_sig, default="4/4", help="e.g. 4/4, 6/8")
    p.add_argument("--input", metavar="WAV", help="analyse a file instead")
    p.add_argument("--rate", type=int, default=44100)
    p.add_argument(
        "--window",
        type=int,
        default=8192,
        help="samples per reading; 8192 is needed to hear low bass notes",
    )
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--in-device", type=int)
    p.add_argument("--out-device", type=int)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()
    if isinstance(args.sig, str):
        args.sig = parse_sig(args.sig)

    if args.input:
        analyse_file(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
