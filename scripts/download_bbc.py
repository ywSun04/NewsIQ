"""Download the BBC News classification CSV into data/bbc-text.csv."""

import sys
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "bbc-text.csv"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already present: {DEST}")
        return 0
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
