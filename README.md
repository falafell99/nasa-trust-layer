# Trust Layer

**NASA Space Apps Challenge 2026, Budapest. Team Zakaz Konina.**
Challenge: *Create Health Monitoring Software for Astronauts on Space Missions.*

> A monitor raised an alarm. Can the crew trust it?
> Trust Layer is an offline layer that re-checks an alarm that is already raised, using ECG and PPG (pulse), and shows a verdict with a reason. The original alarm is never hidden. No link to Earth is needed.

## Try the demo (no install)

Open [`docs/demo.html`](docs/demo.html) in a browser. It works offline and replays three real recordings: a real alarm that is supported, a false alarm that is not confirmed, and a real alarm that is not confirmed (an honest miss). The recordings come from the PhysioNet VTaC validation set.

## What it does

1. An existing monitor raises an alarm (in our tests: ventricular tachycardia).
2. Trust Layer looks at the last 10 seconds of ECG and PPG and gives a score from 0 to 1 (LightGBM model, 520 KB).
3. Score at or above the threshold (0.7155): **red**, "Available signals support the monitor alert."
4. Below the threshold: **yellow**, "Verification inconclusive. The original alert remains active."

Yellow does not mean safe. In our test, 58 of the 396 yellow alarms were real, so yellow alarms stay on screen and the crew still checks them.

## Result (VTaC test set, 482 ICU alarms the model never saw)

| Method | Real among marked red | Share real |
|---|---|---|
| Keep every alarm | 137 / 482 | 28% |
| ECG-only rule | 84 / 269 | 31% |
| Our earlier model | 63 / 111 | 57% |
| **Trust Layer (V5)** | **79 / 86** | **92%** (95% interval 84-98%) |

Red holds 79 of the 137 real alarms (58%). The other 58 real alarms stay yellow and visible. Details, intervals and commit hashes: [`docs/RESULTS.md`](docs/RESULTS.md).

Speed, measured on a laptop (Apple M5 Pro, 1 thread, 100 alarms): median 30 ms per alarm, 381 MB peak memory, model file 520 KB. This is a laptop, not flight hardware.

## What we do not claim

- No astronaut data. Everything here is Earth ICU data used as a proxy.
- Not clinical, not validated for medical use.
- Main result is on ventricular tachycardia alarms only.
- The test set was opened twice (an earlier model first, then V5). The threshold was chosen on a separate validation set before the test was read. Patient overlap between splits cannot be ruled out.
- The continuous event detector is future work: our detector experiments were too weak to show.

## Run the full app (needs the datasets)

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# download the datasets listed in ATTRIBUTION.md into data/ (not in git)
./run_demo.sh        # http://127.0.0.1:8000, no internet needed
```

## Data and sources

All data are open PhysioNet datasets. See [`ATTRIBUTION.md`](ATTRIBUTION.md). Data files are not stored in this repository.

## Repository

- `src/trust/` pipeline, model loading, demo app
- `scripts/` training and evaluation scripts
- `models/` frozen V5 model
- `results/` result files (VTaC test, latency, learning curve)
- `docs/` design notes, evaluation protocol, decision log (some notes are in Russian)
- `progress/` team logs

## Team and tools

Rafael Ibayev (product, pitch), Javidan Bandaliyev (detector experiments, demo reliability), Daniyar Iskakov (submission, video, publishing). Built with AI coding assistants (Claude Code and Codex) that the team directed.
