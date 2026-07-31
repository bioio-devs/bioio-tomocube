"""Read-performance benchmark for bioio-tomocube.

Reads every available scene from the sample TCF file (when present) and writes
timing + shape results to ``output.csv``.  When the sample file is absent the
script writes an empty CSV so the CI upload step still finds the file.
"""

import csv
import pathlib
import time

SAMPLE = (
    pathlib.Path(__file__).parent.parent
    / "bioio_tomocube"
    / "tests"
    / "resources"
    / "sample.TCF"
)
OUTPUT = pathlib.Path("output.csv")
FIELDNAMES = ["scene", "shape", "dtype", "elapsed_s"]

rows = []

if SAMPLE.exists():
    from bioio_tomocube import Reader

    rdr = Reader(SAMPLE)
    for scene in rdr.scenes:
        rdr.set_scene(scene)

        t0 = time.perf_counter()
        rdr.xarray_data  # forces in-memory load
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "scene": scene,
                "shape": str(rdr.shape),
                "dtype": str(rdr.dtype),
                "elapsed_s": f"{elapsed:.3f}",
            }
        )
        print(f"  {scene:12s}  shape={rdr.shape}  dtype={rdr.dtype}  {elapsed:.3f}s")

        # Reset cache so next scene triggers a fresh read
        rdr._xarray_data = None
        rdr._xarray_dask_data = None
else:
    print(f"Sample file not found: {SAMPLE} — writing empty output.csv")

with OUTPUT.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {OUTPUT} ({len(rows)} row(s))")
