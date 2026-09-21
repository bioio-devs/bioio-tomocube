"""Read-performance benchmark for bioio-tomocube.

Reads every available scene from the sample TCF file and from the TIFF-export
fixture (when present) and writes timing + shape results to ``output.csv``.
When neither file is present the script writes an empty CSV so the CI upload
step still finds the file.
"""

import csv
import pathlib
import time

RESOURCES = pathlib.Path(__file__).parent.parent / "bioio_tomocube" / "tests" / "resources"
SAMPLES = [
    RESOURCES / "sample.TCF",
    RESOURCES
    / "tiff"
    / "251003.112316.6 Well Mito.018.Group1.A1.TP01_HT3D_0.00.TIFF",
]
OUTPUT = pathlib.Path("output.csv")
FIELDNAMES = ["file", "scene", "shape", "dtype", "elapsed_s"]

rows = []

for sample in SAMPLES:
    if not sample.exists():
        print(f"Sample file not found: {sample} — skipping")
        continue

    from bioio_tomocube import Reader

    rdr = Reader(sample)
    for scene in rdr.scenes:
        rdr.set_scene(scene)

        t0 = time.perf_counter()
        rdr.xarray_data  # forces in-memory load
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "file": sample.name,
                "scene": scene,
                "shape": str(rdr.shape),
                "dtype": str(rdr.dtype),
                "elapsed_s": f"{elapsed:.3f}",
            }
        )
        print(
            f"  {sample.name[:40]:40s} {scene:12s} shape={rdr.shape} "
            f"dtype={rdr.dtype} {elapsed:.3f}s"
        )

with OUTPUT.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUTPUT} ({len(rows)} row(s))")
