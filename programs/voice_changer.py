#!/usr/bin/env python3
"""
voice_changer.py - live voice changer (mic -> effect -> speakers / WAV).

Install:   pip install sounddevice numpy
Live:      python voice_changer.py                 (WEAR HEADPHONES - see below)
Record:    python voice_changer.py --record me.wav
Convert:   python voice_changer.py --input in.wav --output out.wav --voice deep
Devices:   python voice_changer.py --list-devices

While it runs, type a voice number or name and press Enter to switch.
Type q and press Enter to quit (a --record file is saved on quit).

Headphones: with speakers, the mic hears the changed voice and feeds it back
in a loop. Use headphones, or add --no-monitor to record without playback.
"""

import argparse
import sys
import wave

import numpy as np

# name: (pitch ratio, ring-mod frequency Hz, ring-mod mix 0..1)
VOICES = {
    "normal": (1.00, 0, 0.0),
    "deep": (0.75, 0, 0.0),
    "anonymous": (0.65, 35, 0.3),
    "chipmunk": (1.60, 0, 0.0),
    "robot": (1.00, 55, 0.9),
    "alien": (1.30, 180, 0.6),
}
NAMES = list(VOICES)


class PitchShifter:
    """Streaming pitch shifter: two crossfading read heads on a delay line.

    Each head reads the recent past at `ratio` x speed. When a head runs out
    of buffer it jumps back, and the other head covers the jump, so there is
    no click. Adds roughly `window_s` of latency at most.
    """

    N = 1 << 16  # ring buffer length (samples)

    def __init__(self, sr, ratio=1.0, window_s=0.045):
        self.W = float(int(sr * window_s))
        self.buf = np.zeros(self.N, np.float32)
        self.wp = 0
        self.phase = 0.0
        self.ratio = ratio

    def process(self, x):
        n = len(x)
        t = np.arange(n)
        self.buf[(self.wp + t) % self.N] = x
        if abs(self.ratio - 1.0) < 1e-3:
            self.wp = (self.wp + n) % self.N
            return x.copy()
        rate = 1.0 - self.ratio  # how fast the delay grows, samples/sample
        ph = (self.phase + rate * t) % self.W
        self.phase = (self.phase + rate * n) % self.W
        out = np.zeros(n, np.float32)
        for d in (ph, (ph + self.W / 2) % self.W):
            gain = 0.5 - 0.5 * np.cos(2 * np.pi * d / self.W)
            pos = self.wp + t - d - 2.0
            i0 = np.floor(pos).astype(np.int64)
            frac = (pos - i0).astype(np.float32)
            s = self.buf[i0 % self.N] * (1 - frac) + self.buf[(i0 + 1) % self.N] * frac
            out += (gain * s).astype(np.float32)
        self.wp = (self.wp + n) % self.N
        return out


class Voice:
    """Full effect chain: pitch shift -> ring modulation -> soft limiter."""

    def __init__(self, sr, name="deep"):
        self.sr = sr
        self.shifter = PitchShifter(sr)
        self.ring_phase = 0.0
        self.set(name)

    def set(self, name):
        self.name = name
        ratio, self.ring_hz, self.ring_mix = VOICES[name]
        self.shifter.ratio = ratio

    def process(self, x):
        y = self.shifter.process(x.astype(np.float32))
        if self.ring_mix > 0:
            step = 2 * np.pi * self.ring_hz / self.sr
            ph = self.ring_phase + step * np.arange(len(y))
            self.ring_phase = (self.ring_phase + step * len(y)) % (2 * np.pi)
            y = y * ((1 - self.ring_mix) + self.ring_mix * np.sin(ph))
        return np.tanh(y * 1.2).astype(np.float32)  # gentle limiter


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit("Only 16-bit WAV files are supported for --input.")
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    data = data.reshape(-1, ch).mean(axis=1) / 32768.0
    return data.astype(np.float32), sr


def write_wav(path, audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def convert_file(args):
    audio, sr = read_wav(args.input)
    voice = Voice(sr, args.voice)
    out = [voice.process(audio[i : i + 1024]) for i in range(0, len(audio), 1024)]
    path = args.output or f"{args.voice}_{args.input}"
    write_wav(path, np.concatenate(out) if out else audio, sr)
    print(f"Wrote {path} ({len(audio) / sr:.1f}s, voice: {args.voice})")


def pick_voice(text):
    text = text.strip().lower()
    if text.isdigit() and 1 <= int(text) <= len(NAMES):
        return NAMES[int(text) - 1]
    matches = [n for n in NAMES if n.startswith(text)] if text else []
    return matches[0] if len(matches) == 1 else None


def run_live(args):
    import sounddevice as sd  # imported here so file conversion needs no audio device

    if args.list_devices:
        print(sd.query_devices())
        return

    sr = args.rate
    voice = Voice(sr, args.voice)
    recorded = []

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        y = voice.process(indata[:, 0])
        outdata[:, 0] = 0 if args.no_monitor else y
        if args.record:
            recorded.append(y.copy())

    menu = "  ".join(f"{i + 1}={n}" for i, n in enumerate(NAMES))
    print(f"Voices: {menu}   q=quit")
    if not args.no_monitor:
        print("Wear headphones to avoid feedback.")
    try:
        with sd.Stream(
            samplerate=sr,
            blocksize=args.block,
            channels=1,
            dtype="float32",
            device=(args.in_device, args.out_device),
            callback=callback,
        ):
            while True:
                text = input(f"[{voice.name}] > ")
                if text.strip().lower() in ("q", "quit", "exit"):
                    break
                name = pick_voice(text)
                if name:
                    voice.set(name)
                else:
                    print(f"Unknown voice. {menu}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if args.record and recorded:
            write_wav(args.record, np.concatenate(recorded), sr)
            print(f"\nSaved {args.record}")


def main():
    p = argparse.ArgumentParser(description="Live voice changer")
    p.add_argument("--voice", choices=NAMES, default="deep")
    p.add_argument("--record", metavar="WAV", help="save the changed voice")
    p.add_argument("--no-monitor", action="store_true", help="don't play back")
    p.add_argument("--input", metavar="WAV", help="convert a WAV file instead")
    p.add_argument("--output", metavar="WAV", help="output path for --input")
    p.add_argument("--rate", type=int, default=44100)
    p.add_argument("--block", type=int, default=512, help="lower = less delay")
    p.add_argument("--in-device", type=int, help="mic index (--list-devices)")
    p.add_argument("--out-device", type=int, help="speaker index")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()
    if args.input:
        convert_file(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
