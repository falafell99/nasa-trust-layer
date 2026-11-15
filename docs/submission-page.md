# Submission page text (NASA Space Apps 2026, Budapest)

Replace the three links at the bottom before sending.

## Project name
Trust Layer

## Team
Zakaz Konina: Rafael Ibayev, Javidan Bandaliyev, Daniyar Iskakov

## Challenge
Create Health Monitoring Software for Astronauts on Space Missions

## Short summary (about 100 words)
On a Mars mission, a crew cannot wait up to 24 minutes for ground support to tell them whether an alarm is real. Hospital studies show that most monitor alarms are false. Trust Layer is an offline layer that sits on top of an existing monitor. When the monitor raises an alarm, Trust Layer re-checks it using ECG and pulse (PPG) and shows a verdict and a reason. The original alarm is never hidden. On 482 ICU alarms the model had never seen, 92% of Trust Layer's red alarms were real, against 28% if every alarm is trusted. It runs in about 30 milliseconds on a laptop, with no cloud.

## What we built
- An alarm-checking model (LightGBM, 520 KB) that scores an alarm from the last 10 seconds of ECG and PPG.
- A verdict screen: red "Available signals support the monitor alert", or yellow "Verification inconclusive. The original alert remains active."
- An offline replay demo with three real recordings.

## Results
- VTaC test set: 482 ICU ventricular tachycardia alarms, 137 real.
- Trust Layer marked 86 alarms red, 79 of them real: 92% (95% interval 84-98%). Keeping every alarm gives 28%.
- Red holds 58% of the real alarms. The rest stay yellow and visible, so no alarm is hidden.
- Median 30 ms per alarm, 381 MB peak memory, on one laptop thread.

## Honest limits
- Earth ICU data, not astronauts. Not clinical, not validated for medical use.
- Ventricular tachycardia alarms only.
- The test set was opened twice (an earlier model first, then the final one). The threshold was set on separate validation data first. Patient overlap between splits cannot be ruled out.
- The continuous event detector is future work.

## How open data made this possible
NASA does not publish raw astronaut ECG with labelled alarms. The project exists because researchers have opened ICU data: PhysioNet VTaC (labelled ventricular tachycardia alarms with ECG and PPG), the PhysioNet/CinC 2015 alarm challenge, and BUT QDB (wearable ECG quality). We used them as an Earth proxy. All sources are listed in ATTRIBUTION.md.

## Use of AI tools
We built Trust Layer with AI coding assistants (Claude Code and Codex) that the team directed. The team chose the problem, designed the evaluation protocol, froze the model and threshold before reading the test set, and wrote down every result, including the ones that did not work.

## Next steps
1. Record real crew data with space labs and analog-mission crews.
2. Build a continuous detector that watches the signal all the time.
3. Test how a crew reacts to the verdict, then run it on an on-board device.

## Links (fill in)
- Slides: <PASTE LINK>
- Demo video: <PASTE LINK>
- Repository: <PASTE LINK>
