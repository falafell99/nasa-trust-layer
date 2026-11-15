# Submission page text (NASA Space Apps 2026, Budapest)

Draft for the project page. Written 2026-11-15. Language: English (the page is in English). Placeholders `[__]` must be filled by a person before submission, see "Open items" at the bottom (not part of the page).

---

## Project name
Trust Layer

## Team
Zakaz Konina

## Team members
- [__ real name, as on the Space Apps registration] — [__ what they did in these 2 days]
- [__] — [__]
- [__] — [__]

## Challenge
Create Health Monitoring Software for Astronauts on Space Missions

## One-line summary (24 words)
An offline layer that sits between a heart monitor and the crew. It keeps the monitor's alert visible and adds a status and a reason.

## Description (about 140 words)
On a long mission the crew may have to judge heart-monitor alerts alone, and most alerts in intensive-care studies turn out not to be real. Trust Layer is an offline layer between the sensor and the crew. The monitor's own alert is always shown and never replaced. For each alert we add one status, red ("Available signals support the monitor alert") or yellow ("Verification inconclusive. The original alert remains active"), plus one line of reason. A small LightGBM model reads ECG and PPG features; it runs in about 30 ms per alert on a laptop, with no cloud and no LLM. The demo replays recorded Earth ICU alerts through a virtual sensor. This is not a diagnosis tool and not a medical device. It was not tested on astronauts, and it is not clinically validated.

## Results (our measurement, 2026-11-15)
Held-out VTaC test split: 482 alerts, 137 confirmed by annotators, 226 patients/records. Model frozen at commit `c4cd79d`; command `python scripts/v5.py test`; file `results/v5_vtac_test.json`.

| | Red group | Share confirmed inside red group (95% CI) | Confirmed alerts that are red (95% CI) | Red group as share of all alerts |
|---|---|---|---|---|
| Keep everything red | 482 | 0.284 [0.218, 0.348] | 1.000 | 1.000 |
| Trust Layer (V5) | 86 (79 confirmed, 7 not) | 0.919 [0.843, 0.977] | 0.577 [0.457, 0.687] | 0.178 |

Intervals: bootstrap by patient/record, B=2000. The other 58 of 137 confirmed alerts stay yellow, still visible, nothing is hidden.
Limits: this test split was opened once before for an earlier model, so it is not a clean one-shot test; the threshold was chosen on the validation split; VTaC holds only ventricular tachycardia alerts; Earth ICU data only.

What did not work: we also tried a continuous VT/VF detector on VFDB, CUDB and MIT-BIH. On its one-shot held-out run the logistic model found 0 of 28 episodes (a simple rule found 7 of 28 at 157.3 false alarms per hour; 10.35 h of background). So we do not claim event detection: Trust Layer only checks an alert the monitor has already raised. Source: `results/detector_holdout.json`, branch `feature/event-detector`.

## How open data made this possible
We did not use any NASA dataset. We checked the Resources tab of our challenge page on 2026-11-15: it lists NASA Human Research Program web pages (hazards, risks, ongoing research) and a Canadian Space Agency article on radiation monitoring, and no datasets. We used the NASA Human Research Program material only as background to frame the problem (distance from Earth, crew judging health on its own). That is context, not validation.

What made the work possible is open data from PhysioNet, all with open licences:
- **VTaC v1.1** (ICU ventricular tachycardia alerts, ECG + PPG, 5,037 labelled events): training and the held-out test above.
- **PhysioNet/CinC Challenge 2015** (750 ICU alerts, ECG + PPG): training and a second look.
- **BUT QDB** (wearable ECG with expert signal-quality labels): the signal-quality demo scene.

The demo bundles small excerpts of VTaC (CC BY-SA 4.0) and BUT QDB (CC BY 4.0) so it runs offline; sources, authors and licences are in `demo_data/ATTRIBUTION.md` (branch `feature/event-detector`, not yet on `main`).

No data from astronauts is openly available for this task, as far as we found. Every number on this page comes from Earth hospital data.

## Built with
Python, LightGBM, NeuroKit2, SciPy, FastAPI, one offline HTML page. Code: [__ repository link, or "available on request" if private].

## Links
- Slides (PDF): [__]
- Demo video: [__]
- Code: [__]

---

## Open items (not part of the page)

1. Team members and roles: need real names from the team (task C in `progress/handoffs/2026-11-15-team-tasks-v2.md`). Not invented here.
2. Numbers come from `results/v5_vtac_test.json` (V5, commit `c4cd79d`), not from `docs/11-technical-status-1145.md`: that file was written at 11:45 and still shows the V3 numbers, which V5 replaced. The "88.8% of 12,671 alerts" figure (Drew et al. 2014) is left out of the page because the description is capped at 150 words; it is in `docs/02-problem-evidence.md` if needed.
3. Latency (30.2 ms median, 45.9 ms p95 on an Apple M5 Pro laptop, `results/latency.json`) appears in the description as "about 30 ms on a laptop".
4. The challenge asks for a tool where astronauts assess their own health indicators; our project verifies a heart-monitor alert. Judges on Relevance may notice. The page leads with the "crew judges alone" angle on purpose.
5. Repository public or private is not decided (Q9 in `docs/09`). As of 13:50 it is `falafell99/nasa-trust-layer` and private (the clean-clone report says access had to be authorised). Judges cannot open a private link, so either make it public or write "available on request".
6. Branch `feature/event-detector` (Javidan) is not merged into `main`. It holds `demo_data/`, `requirements-demo.txt`, the failed detector experiment, `docs/qa.md` and `docs/clean-clone-report.md`. On `main` a clean clone shows no alert screen (needs `data/`), per its own report. Merge before the repo link goes on the page.
7. `docs/qa.md` on that branch quotes the V3 numbers from `docs/11-technical-status-1145.md` (11 of 14 red on CinC), which V5 replaced. Update before rehearsal.
