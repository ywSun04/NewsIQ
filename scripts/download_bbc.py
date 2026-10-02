"""Download the BBC News classification CSV into data/bbc-text.csv.

If the file is already there, this script checks its SHA-256 and does not
download again. A mismatch exits without replacing the file.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/bbc-text.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "bbc-text.csv"
EXPECTED_SHA256 = "fdaee0f7451cd8db2709d00e992886fe1c387ee332b8c7b7c1554ed3d3e3382e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        got = sha256(DEST)
        if got != EXPECTED_SHA256:
            print(f"Hash mismatch: {DEST}", file=sys.stderr)
            print(f"expected {EXPECTED_SHA256}", file=sys.stderr)
            print(f"got      {got}", file=sys.stderr)
            print("The existing file was not replaced.", file=sys.stderr)
            return 1
        print(f"Already present and hash matches: {DEST}")
        return 0
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    got = sha256(DEST)
    if got != EXPECTED_SHA256:
        print("Downloaded file hash does not match the recorded digest.", file=sys.stderr)
        print(f"expected {EXPECTED_SHA256}", file=sys.stderr)
        print(f"got      {got}", file=sys.stderr)
        return 1
    print(f"Saved {DEST} ({DEST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
