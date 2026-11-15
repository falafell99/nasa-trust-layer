# Trust Layer: technical status for external review
Written Sun 2026-11-15 11:45 (Budapest). Hackathon: NASA Space Apps 2026, Budapest. Challenge: "Create Health Monitoring Software for Astronauts on Space Missions". Submission 16:00, pitch 17:00. Team of 3 + AI coding agents. Time left for modelling: about 2 hours (decision at 13:45).

Please review the technical part critically. Questions are at the end (section 10).

## 1. What we are building (one paragraph)
Trust Layer is an offline layer between a patient monitor and the crew. The monitor's own alarm is always shown and never replaced. For each alarm we add a status and one reason. Status is one of two: red ("Looks real. Check the person now") or yellow ("Event not excluded. Check"). If the signal quality is too low we say "Signal quality low. Check the sensor." The word "false" is never shown. No cloud, no LLM, small models only. It is NOT a diagnosis system and NOT a replacement for the alarm.

Reason for the design: on Mars the one-way signal delay is up to about 24 minutes, so the crew must judge alarms alone. In ICU studies most arrhythmia alarms are false (89% of 12,671 annotated alarms in Drew et al. 2014), so the crew cannot tell which alarm to run to.

## 2. Pipeline contract (code: src/trust/pipeline.py, model.py, qual.py)
- quality(window, accel=None) -> float in 0..1 (a score, not a probability)
- verify(segment{ecg1, ecg2, ppg, alarm_type, fs=250}) -> p_true
- decide(q, p_true, alarm_type) -> {state, reason_code}
- Flow: rule trigger (the monitor's alarm) -> quality gate -> event check (ECG leads + PPG) -> status + reason.
- Thresholds fixed in advance, not tuned on validation: Q_GOOD = 0.60, T_RED = 0.5.
- Gray status ("probably not confirmed") was built and then DISABLED: on validation it contained 1 false and 0 true, so it carried no information.

## 3. Data
| Set | What | Size | Use |
|---|---|---|---|
| PhysioNet/CinC 2015 | ICU alarms (asystole, extreme brady/tachy, VT, VF), ECG + PPG, 250 Hz | 750 records, 456 false (61%) | main evaluation; split by record 450 train / 150 val / 150 hold-out (seed 42) |
| VTaC | ICU ventricular tachycardia alarms, ECG + PPG | 5,037 events, 1,441 true / 3,596 false; benchmark split 4,060 / 495 / 482 | extra training data; the 482-event test split is NOT to be opened until the final run |
| BUT QDB | wearable single-lead ECG, 1000 Hz, annotated quality classes 1/2/3 | annotated records 100001, 105001, 111001 only | quality gate training/test |
| NSTDB | noise stress test records | | synthetic stress only |
Caveats: all Earth data (ICU patients and a few wearable subjects). No astronaut data exists publicly. No data from NASA partners used so far. CinC may contain several alarms from one patient; patient IDs are not given, so patient overlap between train and test cannot be ruled out.

## 4. Evaluation protocol
- Label: true alarm vs false alarm, as annotated in the datasets.
- Split by record, never by window. CinC 450/150/150, seed 42. Signals truncated to the first 300 s, resampled with resample_poly to 250 Hz, channels picked by name.
- Main metric: share of true alarms among alarms marked red (a precision-like number), together with the size of the red group, with bootstrap 95% intervals by record (B=2000, percentile). Also the CinC 2015 challenge score as a secondary number. "Rule of three" used for zero counts.
- The CinC hold-out (150) was opened exactly once, on a frozen commit (1dad545). It is spent and must not be used for tuning any more.

## 5. Models so far (frozen at 1dad545)
- V0 baselines: keep-all (everything red), alarm-type-only (red if the type's training share of true alarms is >= 0.5), ECG-only rule.
- V3: logistic regression on 22 interpretable features + 5 alarm-type flags. Red if p_true >= 0.5 and quality is good.
- Quality rule v2 (q_v2): logistic regression on 9 scale-free features (flat-line fraction, share of power under 0.5 Hz, in 5-15, 20-40, 40-100 Hz, kurtosis, beat count, share of plausible RR, CV of RR); q = 1 - P(class 2 or 3). Trained on BUT 100001 and 105001, tested once on 111001. Variant chosen by leave-one-record-out between the two training records.
  - Old q had a bug: a huge low-frequency artefact (raw signal clipping at +-32767) lowered the 20-40 Hz fraction, so the old rule called it "clean".
  - Test on 111001 (share of 10 s windows below 0.60): class 1 56.0% -> 0.5%, class 2 92.8% -> 45.8%, class 3 61.6% -> 73.3%. AUROC class 3 vs class 1: 0.544 -> 0.964. One test subject only.

## 6. Results on the CinC hold-out (n=150: 53 true, 97 false), frozen commit 1dad545
"Red" = predicted real. Source: results/holdout_final.json.
| Method | red true / red total | true share in red [95% CI] | red group as share of all alarms | CinC score |
|---|---|---|---|---|
| keep-all | 53 / 150 | 0.353 [0.280, 0.433] | 1.00 | 35.3 |
| alarm-type only | 31 / 43 | 0.721 [0.585, 0.854] | 0.287 | 48.7 |
| ECG-only rule (V0) | 30 / 57 | 0.526 [0.400, 0.655] | 0.38 | 41.3 |
| V3 | 11 / 14 | 0.786 [0.538, 1.000] | 0.093 | 33.0 |
Reading, honestly:
- V3's red group is purer (11/14) but small: only 11 of 53 true alarms are marked red, versus 31 of 53 for the alarm-type rule. The 42 other true alarms stay visible as yellow, so nothing is hidden.
- V3 is NOT better than the alarm-type rule overall. CinC score of V3 (33.0) is below keep-all (35.3).
- The quality gate does not transfer: q < 0.60 for 103 of 150 records (68.7%) on CinC. The gate was trained on wearable ECG at 1000 Hz and is not validated on ICU data. This is the main reason the red group is small.
- On validation, V3 had 11/17 = 0.647 in red; on hold-out 0.786 with 14 alarms. The difference is within the wide intervals.

## 7. Work in progress (Claude Code agent, started about 11:15)
Goal: raise the honest quality of the event check. Stage 1 done so far:
- New features (src/trust/feat2.py): neurokit2 QRS detection, signal-quality indices per lead, features following the CinC alarm definitions (asystole, brady, tachy, VT, VF), PPG peaks and ECG/PPG agreement. Same extractor for CinC and VTaC.
- scripts/cv_gbm.py: 5-fold StratifiedGroupKFold (groups = CinC record / VTaC patient) on train+validation only. Models: LightGBM and sklearn HistGradientBoosting. Compares with an out-of-fold alarm-type-only rule at matched red-group size.
- Status messages seen so far: CinC GBM 5-fold CV AUROC 0.927, small 1D-CNN 0.82 (scripts/cv_cnn.py, src/trust/cnn.py). These are cross-validation numbers on train+val, not final, and not yet compared with the alarm-type-only rule at matched size. A bug in the per-lead SQI index offsets was found and is being fixed.
- VTaC download: 3,706 of 5,037 waveform files at 11:44.
Planned next:
1. Fix SQI bug, rerun CV on CinC; then CV on CinC + VTaC train/val.
2. Choose model by CV only (GBM vs V3 vs small CNN), fix threshold and operating point in advance.
3. Freeze (commit hash). Open VTaC test (482 events) once. Show the CinC hold-out only as a "second look" without any tuning.
4. Rule: if the new model is not clearly better than V3 and the alarm-type rule on the VTaC test, we keep V3 and say so.
5. Measure latency, memory and model size (results/latency.json) for the slide.

## 8. What we will and will not claim
Will claim: offline layer; the alarm is never hidden; results on held-out Earth ICU records with intervals; a working demo.
Will NOT claim: astronaut data, clinical validity, a general reduction of false alarms, real-time monitoring validated, superiority over the alarm-type rule (unless the final VTaC test shows it).
Synthetic data: allowed only for training augmentation (labelled) and for a separate "robustness to simulated noise" line. Not used in the headline result.

## 9. Known weaknesses
- Patient overlap in CinC cannot be excluded; CV and hold-out are by record.
- VTaC contains only VT alarms, CinC contains five alarm types; domain shift between the two sets (hospital, equipment, labelling).
- Quality gate: trained on 3 wearable subjects, one test subject, does not transfer to ICU.
- Small red groups, wide intervals.
- Labels "true/false" come from the datasets' annotators; not our own.
- Alarm-type-only is a very strong baseline on CinC because the types differ a lot in base rate.

## 10. Questions for the reviewer
1. Is "true share in the red group + red group size, with bootstrap by record" a sound main metric, or should we report something else (precision at fixed coverage, recall at fixed precision, PR curve)?
2. Cross-validation AUROC 0.927 on CinC looks high. Which leakage risks should we check first (patient overlap, alarm type as a feature, truncation to 300 s, label from the same signal)?
3. How do we choose the operating point (threshold) on CV without peeking at the final test, when the red group must stay big enough to matter? Fixed coverage, fixed precision, or a cost ratio?
4. Is a small 1D-CNN worth keeping next to gradient boosting at 600-5,600 samples, or should we drop it?
5. Quality gate: how can we make it usable on ICU data without ICU quality labels? (Self-supervised, SQI from literature, synthetic artefact augmentation on NSTDB?)
6. Is it acceptable to train on CinC + VTaC and test on the VTaC test split for the headline number? What would you change to make the claim defensible to NASA judges?
7. What is the most convincing evaluation we can still add in under 90 minutes?
8. Anything in the design that is unsafe or misleading for the crew (wording of "Looks real. Check the person now", two statuses, yellow = "Event not excluded")?
