# Submission page text (NASA Space Apps 2026, Budapest)

Matches the form fields of the official Project Submission Guide. All text must be in English. Links must be public, with no login. Fill the three links at the bottom before clicking "Submit for Judging".

## Project Name
Trust Layer

## High-Level Project Summary
On a Mars mission the crew cannot wait up to 24 minutes for ground support to say whether an alarm is real, and hospital studies show that most monitor alarms are false. Trust Layer is an offline layer that sits on top of an existing monitor. When the monitor raises an alarm, Trust Layer re-checks it using ECG and pulse (PPG) and shows a verdict with a reason. The original alarm is never hidden. On 482 ICU alarms the model had never seen, 92% of Trust Layer's red alarms were real, against 28% if every alarm is trusted. It runs in about 30 milliseconds on a laptop, with no cloud.

## Link to Project Demo
Slides (7 slides, public link): <PASTE LINK>

## Link to Final Project
Code, results and an offline demo (docs/demo.html): <PASTE GITHUB LINK, repository must be public>

## Detailed Project Description

**What it does.** An existing monitor raises an alarm (we tested ventricular tachycardia). Trust Layer takes the last 10 seconds of ECG and PPG, gives a score from 0 to 1, and shows one of two verdicts. Red: "Available signals support the monitor alert." Yellow: "Verification inconclusive. The original alert remains active." Yellow does not mean safe: 58 of the 396 yellow alarms in our test were real, so yellow alarms stay on screen.

**How it works.** A small LightGBM model (520 KB) on ECG and PPG features. The threshold was set on a separate validation set (495 alarms) and the model was frozen before the test set was read.

**Result.** On the VTaC test set (482 ICU alarms, 137 real, 226 patients): Trust Layer marked 86 alarms red and 79 were real, 92% (95% interval 84-98%). Keeping every alarm gives 28%; an ECG-only rule gives 31%. Red holds 58% of the real alarms; the rest stay yellow and visible. Median 30 ms per alarm, 381 MB peak memory, one laptop thread.

**Benefit.** The crew sees which alarms the signals support, without waiting for Earth, and nothing is hidden.

**Limits.** Earth ICU data, not astronauts. Not clinical, not validated for medical use. Ventricular tachycardia alarms only. The test set was opened twice (an earlier model first, then the final one); the threshold was fixed on validation data first. Patient overlap between splits cannot be ruled out. The continuous event detector is future work.

**Next.** Record real crew data with space labs and analog-mission crews, build a continuous detector, and test how a crew reacts to the verdict.

**Tools.** Python, NumPy, SciPy, pandas, scikit-learn, LightGBM, PyTorch (earlier model), WFDB, NeuroKit2, FastAPI, plain HTML and canvas for the offline demo.

## Use of Super Intelligence (SI)
We used AI assistants (Claude and Codex) to help write and run code, to analyse results, to draft documentation and this text, and to build the slides. No images or videos in the project are AI-generated: all screenshots come from our own demo screen. The repository README and ATTRIBUTION.md state that the code and documentation were produced with AI assistance. Results were produced by running our own code on public datasets and are logged in results/.

## NASA Data
We did not use NASA datasets. We used NASA published information for the problem statement: the one-way communication delay between Earth and Mars (up to about 24 minutes) and NASA Exploration Medical Capability material about crew care without ground support.

## Space Agency Partner Data and Other References
- Canadian Space Agency (CSA), Bio-Monitor smart shirt worn on the ISS: https://www.asc-csa.gc.ca/eng/sciences/bio-monitor.asp
- PhysioNet VTaC, ventricular tachycardia alarms (main result): https://physionet.org/content/vtac/1.1/
- PhysioNet/CinC Challenge 2015, false alarms in the ICU: https://physionet.org/content/challenge-2015/1.0.0/
- PhysioNet BUT QDB, wearable ECG quality: https://physionet.org/content/butqdb/1.0.0/
- Drew et al., PLoS ONE 2014: 89% of 12,671 ICU alarms were false.
- Full list and software: ATTRIBUTION.md in the repository.
