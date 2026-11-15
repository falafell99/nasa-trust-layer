# Team tasks v2 — Sun 2026-11-15, written 11:40 (supersedes 2026-11-15-team-tasks.md where they differ)

Deadline: submission 16:00 Budapest (aim 15:30). Pitch 17:00, 3 minutes. English everywhere on screen.

## Read first (every agent, every person)
1. progress/STATUS.md
2. this file
3. docs/10-decision-log.md (decisions D-07, D-08, D-17, D-20, D-21)

## Current state (11:40)
- Product: Trust Layer. Offline layer between sensor and crew. Original monitor alarm is never hidden. Output: status + one reason.
- Story (deck order): cover with real screen → problem (24 min one way to Mars + 89% of ICU alarms false) → solution → demo → result → proposed commercial model → team and next step.
- Deck (view and comment): https://claude.ai/artifact/N5J1H23YKFDpeRMxDPqtpr
- Hold-out result (n=150, CinC 2015, frozen commit 1dad545): Trust Layer red 11 of 14 real (79%); alarm-type-only rule 31 of 43 (72%). Real alarms marked red: 11 vs 31 of 53. Not better than the simple rule. Interval 54–100%.
- IN PROGRESS (Claude Code, until ~13:45): better model on VTaC + CinC. Final claim will come from the VTaC test split, opened ONCE at the end. Do not touch VTaC test files. Do not tune on the CinC hold-out (already spent).
- Do not claim: astronaut data, clinical use, reducing false alarms, real-time monitoring validated. Say: offline replay of recorded Earth ICU data.
- Moon delay is seconds. The long delay (up to ~24 min one way) is Mars / deep space.

## Rules for everyone
- Do not edit models/, results/, scripts/, src/ unless Rafael says so. Claude Code owns them right now.
- Docs and recordings go in new files. Pull before push. Small commits.
- Log what you did in progress/logs/<your-name>.md (3 lines).
- Never invent a number. If a number is missing, write [__] and ask Rafael.

## Tasks (all can start now, none block each other)

### A. Submission page text → docs/submission-page.md, by 14:00
- Project name, one-line summary (≤ 25 words), description (≤ 150 words), challenge name.
- "How NASA open data made this possible": open the challenge page, list the NASA / partner datasets and resources it names, say which ones we used and which we did not. If we used none, say so plainly and list what we used instead (PhysioNet CinC 2015, VTaC, BUT QDB).
- Low word count, plain words, no hype.

### B. Backup demo recordings → docs/assets/video/, by 14:30
- Run ./run_demo.sh offline (airplane mode on) at http://127.0.0.1:8000.
- Record two screen captures: 90 s and 30 s. English subtitles. No mouse jitter, 1080p.
- Scenes: original alarm visible → verdict "Looks real. Check the person now" → a noisy signal → "Signal quality low. Check the sensor." → "Show labels" once.
- The UI should not change with the new model. If Rafael says it changed, re-record.

### C. Names and roles → send to Rafael by 12:15
- Real names (as on the Space Apps registration), one role line each, for the cover and the team slide.
- Roles must be true. Example format: "Name — what they did in these 2 days".

### D. Pitch rehearsal and Q&A, from 13:00
- One person times Rafael with a stopwatch, 3:00 hard stop. Target: problem 0:00–0:25, product + demo 0:25–2:15, result 2:15–2:40, business + close 2:40–3:00.
- Ask these questions, Rafael answers out loud:
  1. Is there not a doctor on board?
  2. Your model is not better than the alarm-type rule. Why does this matter?
  3. This is Earth ICU data, not astronauts. Why should we trust it?
  4. Could the same patients be in train and test?
  5. Why offline, why not an LLM?
  6. Who pays?
- Write the answers Rafael liked into docs/qa.md.

### E. Publishing checklist → by 15:15
- Slides: export PDF from the deck link, upload to Google Drive, share "anyone with the link". Check it in a private window.
- Video: upload 30 s or 90 s to YouTube as unlisted. Check in a private window.
- GitHub repo: Rafael decides public or private. If public, check no keys, no raw datasets, no private notes in the repo.
- Put all links in docs/submission-page.md.
