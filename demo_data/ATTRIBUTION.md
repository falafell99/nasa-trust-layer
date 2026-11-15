# Demo data attribution

`demo_data/` contains only the excerpts required by the four offline demo scenes. The original recordings and annotations remain attributable to their respective PhysioNet datasets; bundling these excerpts does not imply endorsement of this project by the dataset authors or PhysioNet.

## VTaC v1.1 — scenes B1, B2, B3

- Source: [VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors, version 1.1](https://physionet.org/content/vtac/1.1/), published 2026-09-14; DOI [10.13026/jg2e-tf67](https://doi.org/10.13026/jg2e-tf67).
- Authors: Li-wei Lehman, Benjamin Moody, Lucas McCullum, Hasan Saeed, Harsh Deep, Diane Perry, Tristan Struja, Qiao Li, Gari Clifford, and Roger Mark. Cite the dataset version and its original NeurIPS 2023 paper when reusing the data.
- License: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). The original PhysioNet `LICENSE.txt` is included in this folder.
- The three examples are validation-split VT events: B1 `0060de/0060de_0560` (True), B2 `012473/012473_0351` (False), B3 `02d43c/02d43c_0267` (True). Only the first 300 seconds (the five minutes before the alarm) are bundled from each six-minute WFDB record; the header sample count is adjusted to the included 75,000 samples. The one-minute post-alarm portion is omitted.
- `benchmark_data_split.csv` and `event_labels.csv` are restricted to these three event rows. Labels are displayed only when the user reveals the VTaC annotation; they are not passed to the model.

## BUT QDB v1.0.0 — scene A

- Source: [Brno University of Technology ECG Quality Database (BUT QDB), version 1.0.0](https://physionet.org/content/butqdb/1.0.0/), DOI [10.13026/kah4-0w24](https://doi.org/10.13026/kah4-0w24).
- Authors: Andrea Nemcova, Radovan Smisek, Kamila Opravilová, Martin Vitek, Lukas Smital, and Lucie Maršánová. Cite the dataset version and original paper when reusing the data.
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The original PhysioNet `LICENSE.txt` is included in this folder.
- Scene A uses a 180-second excerpt of record `111001` from 67,740 to 67,920 seconds: one ECG channel at 1,000 Hz, three accelerometer axes at 100 Hz, and only consensus annotation rows intersecting that excerpt. The ECG/ACC samples are raw signed 16-bit values; the annotations retain the source record's sample coordinates.

The two dataset folders retain their separate licenses. The sample waveforms are included for offline demonstration and are not evidence of clinical or spaceflight validation.
