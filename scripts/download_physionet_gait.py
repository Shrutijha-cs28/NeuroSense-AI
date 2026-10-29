"""Download only the gait recordings needed from PhysioNet GaitPDB.

The source recordings are kept under data/raw (ignored by Git). The public S3
mirror is the mirror linked by PhysioNet's official dataset page.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


BUCKET_URL = "https://physionet-open.s3.amazonaws.com"
PREFIX = "gaitpdb/1.0.0/"
RECORDING = re.compile(r"(?:Ga|Ju|Si)(?:Co|Pt)\d+_\d+\.txt$")


def list_recordings() -> list[tuple[str, int]]:
    url = f"{BUCKET_URL}/?prefix={PREFIX}"
    request = urllib.request.Request(url, headers={"User-Agent": "NeuroSenseAI dataset-prep"})
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    return [
        (key, int(size))
        for item in root.findall("s3:Contents", ns)
        if (key := item.findtext("s3:Key", namespaces=ns))
        and (size := item.findtext("s3:Size", namespaces=ns))
        and RECORDING.fullmatch(key.rsplit("/", 1)[-1])
    ]


def download_one(item: tuple[str, int], destination: Path) -> str:
    key, expected_size = item
    target = destination / key.rsplit("/", 1)[-1]
    if target.is_file() and target.stat().st_size == expected_size:
        return target.name
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        request = urllib.request.Request(f"{BUCKET_URL}/{key}", headers={"User-Agent": "NeuroSenseAI dataset-prep"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temporary.stat().st_size != expected_size:
                raise IOError(f"Size mismatch for {target.name}: expected {expected_size} bytes")
            temporary.replace(target)
            return target.name
        except (OSError, TimeoutError):
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/physionet_gaitpdb/1.0.0"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = list_recordings()
    if not files:
        raise RuntimeError("No GaitPDB recordings found in the public PhysioNet mirror")
    total_bytes = sum(size for _, size in files)
    print(f"Downloading {len(files)} recordings ({total_bytes / 1_000_000:.1f} MB) to {args.output}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, item, args.output) for item in files]
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            if number % 25 == 0 or number == len(futures):
                print(f"Downloaded {number}/{len(futures)} recordings", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
