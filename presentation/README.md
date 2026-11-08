# Presentation

| Folder | What it is |
|---|---|
| `talk-3min/` | 8 slides for a ~3 min talk. `index.html` (open in browser), `deck.pdf`, `notes.md` (speaker script, Q&A answers, evidence checklist). |
| `video-1min/` | 6-slide narrated version, ~1 min. `talk.mp4` (Kokoro voice `af_heart`), `narration.json`, `deck.pdf`. |
| `media/` | **Drop your demo clips here.** Slots pick them up automatically (see below). |
| `shared/` | Common CSS / JS. |
| `build.py` | Regenerates PDFs and the video. |

## Adding your videos

Put clips in `media/` with these names. Slides play them automatically; missing files show a dashed placeholder.

- 3-min deck: `good_1.mp4 good_2.mp4 good_3.mp4` (slide 5), `bad_1.mp4 bad_2.mp4 bad_3.mp4` (slide 6)
- 1-min video: `good_1.mp4` and `bad_1.mp4` are composited into slide 4 (rectangles set in `video-1min/narration.json`)

Edit the "Describe the scenario here" captions in `talk-3min/index.html`.

## Filling in results

Slide 8 of the 3-min deck and slide 5 of the 1-min deck show the results. At the bottom of each `index.html`:
`window.RESULTS` (before/after percent), `window.CURVE` (steps and success rates for the chart), and `window.FAKE=true`.
**The numbers currently there are fake, for layout preview, and slides show a FAKE DATA badge.** Replace them with real numbers and delete the `window.FAKE` line.

## Rebuild

    python build.py talk-3min              # PDF
    python build.py video-1min --video     # PDF + narrated mp4 (needs Kokoro, espeak-ng, ffmpeg, Chrome)

The decks contain no experimental results; the grader and training were not implemented when these were made.
