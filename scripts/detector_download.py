"""Download the three open PhysioNet ECG databases used by this branch."""

from pathlib import Path

import wfdb


ROOT = Path("data/detector")
DATABASES = ("vfdb", "cudb", "mitdb")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for database in DATABASES:
        destination = ROOT / database
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {database} into {destination}", flush=True)
        wfdb.dl_database(database, str(destination), keep_subdirs=False)


if __name__ == "__main__":
    main()
