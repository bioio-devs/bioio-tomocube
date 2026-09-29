"""Read-performance benchmark: every scene of every TCF fixture, timings to output.csv."""

import csv
import pathlib
import time

from bioio_tomocube import Reader

RESOURCES = pathlib.Path(__file__).parent.parent / "bioio_tomocube" / "tests" / "resources"
OUTPUT = pathlib.Path("output.csv")
FIELDNAMES = ["file", "scene", "shape", "dtype", "elapsed_s"]

rows = []
for sample in sorted(RESOURCES.glob("*.TCF")):
    rdr = Reader(sample)
    for scene in rdr.scenes:
        rdr.set_scene(scene)
        t0 = time.perf_counter()
        rdr.xarray_data
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
        print(f"  {sample.name[:44]:44s} {scene:10s} {rdr.shape} {elapsed:.3f}s")

with OUTPUT.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {OUTPUT} ({len(rows)} row(s))")
