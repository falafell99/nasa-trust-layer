# Results (frozen)

Model V5: LightGBM on ECG and PPG features. Threshold T = 0.7155, the smallest threshold with precision of at least 0.85 on the VTaC validation set (495 alarms). Model frozen at commit `c4cd79d`; test results at commit `a485e64`; raw file `results/v5_vtac_test.json`.

## VTaC test set: 482 alarms, 137 real, 226 patients

Intervals are 95% bootstrap intervals by patient, B = 2000.

| Method | Marked red | Real among red | Share real (precision) | Real alarms caught (recall) | Coverage of all alarms |
|---|---|---|---|---|---|
| Keep every alarm | 482 | 137 | 0.284 [0.218, 0.348] | 1.00 | 1.00 |
| ECG-only rule | 269 | 84 | 0.312 [0.222, 0.402] | 0.613 | 0.558 |
| Earlier model V3, no quality gate | 111 | 63 | 0.568 [0.433, 0.685] | 0.460 | 0.230 |
| Earlier model V3, with quality gate | 24 | 11 | 0.458 | 0.080 | 0.050 |
| **Trust Layer V5** | **86** | **79** | **0.919 [0.843, 0.977]** | **0.577 [0.457, 0.687]** | **0.178 [0.122, 0.239]** |

Where the other alarms go: 396 yellow alarms, of which 58 real and 338 false. Yellow alarms stay visible.

## Earlier model V4 (secondary result)

LightGBM and a 1D-CNN averaged, frozen at `8754c53`, results at `a024eb0`. On VTaC test: 122 of 143 red were real (0.853, recall 0.89). On the CinC 2015 hold-out: 41 of 44 red were real (0.932), recall 41 of 53.

## Speed (laptop)

Apple M5 Pro, 1 thread, 100 alarms: median 30.2 ms per alarm (95th percentile 45.9 ms), peak memory 381 MB, model file 520 KB. File: `results/latency.json`.

## Disclosures

- The VTaC test set was opened twice: first by V4, then by V5. V5's threshold was fixed on validation data before its test run.
- Only ventricular tachycardia alarms. ICU data from Earth, not astronauts. Not clinically validated.
- Patient overlap between splits cannot be ruled out.
- The exact threshold is noisy (chosen on 495 validation alarms). More training data changes results within noise at this size.
- Continuous event detector: experiments were too weak to report as a result (leave-record-out episode detection near zero at usable false-alarm rates). It is future work.
