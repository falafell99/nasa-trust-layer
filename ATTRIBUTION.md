# Data and sources

No NASA or astronaut data is used. All data below are open datasets on PhysioNet. Check each dataset page for its licence and citation requirements before reuse. Data files are not stored in this repository.

## Datasets

| Dataset | Used for | Link |
|---|---|---|
| VTaC (ventricular tachycardia alarms, ICU) | Main result and the model (train, validation, test) | https://physionet.org/content/vtac/1.1/ |
| PhysioNet/CinC Challenge 2015 (false alarms in ICU) | Earlier model, second dataset check | https://physionet.org/content/challenge-2015/1.0.0/ |
| BUT QDB (wearable ECG quality) | Signal quality scene and early quality experiments | https://physionet.org/content/butqdb/1.0.0/ |
| VFDB, CUDB, MIT-BIH arrhythmia (PhysioNet) | Detector experiments only (not in the headline result) | https://physionet.org/ |

## Sources for the problem statement

- Drew et al., PLoS ONE 2014: 89% of 12,671 ICU alarms were false (one hospital study, not astronauts).
- NASA: one-way communication delay to Mars of up to about 24 minutes; the Moon has a few seconds.
- Canadian Space Agency, Bio-Monitor smart shirt worn on the ISS: https://www.asc-csa.gc.ca/eng/sciences/bio-monitor.asp
- NASA SBIR 2026 award sizes (Phase I up to $225,000, Phase II up to $1.275M), used only as a size benchmark for our price estimates.

## Software

Python, NumPy, SciPy, pandas, scikit-learn, LightGBM, PyTorch (earlier model), WFDB, NeuroKit2, FastAPI.
