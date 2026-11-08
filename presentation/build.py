#!/usr/bin/env python3
"""Build deck.pdf (and optionally a narrated talk.mp4) for one deck folder.

    python build.py talk-3min            # PDF only
    python build.py video-1min --video   # PDF + narrated video

Video mode reads <deck>/narration.json:
  voice, speed, anim_slide, overlays {slide: [{file,x,y,w,h}]}, slides [text...]
Clips named in "overlays" are looked up in ./media/ and composited over the slide
at the given pixel rectangle (1920x1080 stage). Missing clips leave the placeholder.

Video mode needs Kokoro (pip install kokoro soundfile; brew install espeak-ng),
ffmpeg and Google Chrome. Run inside the venv that has kokoro, with
PHONEMIZER_ESPEAK_LIBRARY and ESPEAK_DATA_PATH pointing at Homebrew espeak-ng.
"""
import argparse, base64, json, mimetypes, re, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
ANIM_FPS, ANIM_LOOP = 15, 4.0   # ANIM_LOOP must match LOOP in shared/deck.js
PAD = 0.7                       # silence after each slide's narration, seconds


def run(cmd, quiet=False):
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL if quiet else None,
                   stderr=subprocess.DEVNULL if quiet else None)


def chrome(args):
    run([CHROME, "--headless=new", "--disable-gpu"] + args, quiet=True)


def shot(url, out):
    chrome(["--hide-scrollbars", "--window-size=1920,1080", "--force-device-scale-factor=1",
            "--virtual-time-budget=3000", f"--screenshot={out}", url])


def duration(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(p)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def tts(texts, voice, speed, outdir):
    import numpy as np, soundfile as sf
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a")
    outdir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(texts, 1):
        audio = np.concatenate([a for _, _, a in pipe(t, voice=voice, speed=speed)])
        sf.write(outdir / f"{i:02d}.wav", audio, 24000)
        print(f"tts {i:02d}: {len(audio)/24000:.1f}s")


def build_standalone(deck):
    """Single-file HTML: CSS, JS, images and videos inlined, so it works in any viewer."""
    html = (deck / "index.html").read_text()
    css = (ROOT / "shared" / "deck.css").read_text()
    js = (ROOT / "shared" / "deck.js").read_text()
    html = html.replace('<link rel="stylesheet" href="../shared/deck.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="../shared/deck.js"></script>', f"<script>\n{js}\n</script>")

    def inline(m):
        f = (deck / m.group(1)).resolve()
        if not f.exists():
            return m.group(0)          # missing file keeps its placeholder
        mime = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        return f'data-src="data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}"'
    html = re.sub(r'data-src="(\.\./media/[^"]+)"', inline, html)
    out = deck / "standalone.html"
    out.write_text(html)
    print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")


def build_pdf(deck):
    out = deck / "deck.pdf"
    chrome(["--no-pdf-header-footer", "--virtual-time-budget=8000", f"--print-to-pdf={out}", (deck / "index.html").as_uri() + "?clean"])
    print("wrote", out)


def build_video(deck, skip_tts):
    cfg = json.loads((deck / "narration.json").read_text())
    texts = cfg["slides"]; n = len(texts)
    anim = cfg.get("anim_slide")
    b = deck / "build"; b.mkdir(exist_ok=True)
    if not skip_tts:
        tts(texts, cfg.get("voice", "af_heart"), cfg.get("speed", 1.0), b / "audio")
    base = (deck / "index.html").as_uri()

    (b / "img").mkdir(exist_ok=True)
    for i in range(1, n + 1):
        if i != anim:
            shot(f"{base}?clean#{i}", b / "img" / f"{i:02d}.png")
    fdir = b / "anim"
    if anim:
        shutil.rmtree(fdir, ignore_errors=True); fdir.mkdir()
        for k in range(int(ANIM_LOOP * ANIM_FPS)):
            shot(f"{base}?clean&t={k/ANIM_FPS:.4f}#{anim}", fdir / f"{k:04d}.png")
    print("screenshots done")

    segs = []
    for i in range(1, n + 1):
        wav = b / "audio" / f"{i:02d}.wav"
        dur = duration(wav) + PAD
        seg = b / f"seg{i:02d}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if i == anim:
            cmd += ["-stream_loop", "-1", "-framerate", str(ANIM_FPS), "-i", str(fdir / "%04d.png")]
        else:
            cmd += ["-loop", "1", "-framerate", "30", "-i", str(b / "img" / f"{i:02d}.png")]
        clips = [o for o in cfg.get("overlays", {}).get(str(i), []) if (ROOT / "media" / o["file"]).exists()]
        for o in clips:
            cmd += ["-stream_loop", "-1", "-i", str(ROOT / "media" / o["file"])]
        cmd += ["-i", str(wav)]
        a_idx = 1 + len(clips)
        chain, last = [], "[0:v]"
        for k, o in enumerate(clips, 1):
            chain.append(f"[{k}:v]scale={o['w']}:{o['h']}:force_original_aspect_ratio=decrease,"
                         f"pad={o['w']}:{o['h']}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[c{k}]")
            chain.append(f"{last}[c{k}]overlay={o['x']}:{o['y']}:shortest=0[o{k}]")
            last = f"[o{k}]"
        chain.append(f"{last}fps=30,format=yuv420p[v]")
        cmd += ["-filter_complex", ";".join(chain), "-map", "[v]", "-map", f"{a_idx}:a",
                "-af", f"apad=pad_dur={PAD}", "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",
                "-ar", "48000", "-t", f"{dur:.3f}", str(seg)]
        run(cmd)
        segs.append(seg)
        print(f"segment {i:02d}: {dur:.1f}s  clips={[o['file'] for o in clips]}")

    lst = b / "list.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    out = deck / "talk.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    print("wrote", out, f"{duration(out):.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    a = ap.parse_args()
    d = (ROOT / a.deck).resolve()
    build_pdf(d)
    build_standalone(d)
    if a.video:
        build_video(d, a.skip_tts)
