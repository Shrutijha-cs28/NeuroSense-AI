"""Download the openly licensed UCI datasets used by the separate prototypes."""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "darwin": "https://archive.ics.uci.edu/static/public/732/data.csv",
    "parkinsons": "https://archive.ics.uci.edu/static/public/489/parkinson%2Bdataset%2Bwith%2Breplicated%2Bacoustic%2Bfeatures.zip",
}


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "NeuroSenseAI-research-prototype/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace existing downloaded CSV files")
    args = parser.parse_args()
    targets = {
        "darwin": ROOT / "data" / "external" / "uci_darwin" / "data.csv",
        "parkinsons": ROOT / "data" / "external" / "uci_parkinsons" / "data.csv",
    }
    for name, target in targets.items():
        if target.exists() and not args.force:
            print(f"Already present: {target}")
            continue
        payload = download(SOURCES[name])
        if name == "parkinsons":
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                member = "ReplicatedAcousticFeatures-ParkinsonDatabase.csv"
                if member not in archive.namelist():
                    raise RuntimeError(f"Expected {member} in official UCI archive")
                payload = archive.read(member)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        print(f"Downloaded {name}: {target} ({len(payload):,} bytes)")


if __name__ == "__main__":
    main()
