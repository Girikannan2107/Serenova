from pathlib import Path
import json
import wfdb


BASE_DIR = Path(r"D:\Backend\serenova-backend")

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "ctu-chb"
    / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "results"
    / "ctu_uhb_ctg"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("CTU-UHB CTG ANNOTATION INSPECTION")
print("=" * 70)


records = sorted(RAW_DIR.glob("*.hea"))

print(f"Header files found: {len(records)}")

results = []

for header in records:

    record_name = header.stem

    record_path = str(header.with_suffix(""))

    item = {
        "record": record_name,
        "annotations": [],
        "annotation_errors": [],
    }

    try:
        record = wfdb.rdrecord(record_path)

        item["n_samples"] = int(record.sig_len)
        item["fs"] = float(record.fs)
        item["channels"] = list(record.sig_name)

    except Exception as exc:
        item["record_error"] = str(exc)
        results.append(item)
        continue

    # Look for WFDB annotation files associated with this record.
    annotation_files = sorted(
        RAW_DIR.glob(f"{record_name}.*")
    )

    for ann_file in annotation_files:

        suffix = ann_file.suffix.lower()

        # Skip normal signal/header files.
        if suffix in {".hea", ".dat"}:
            continue

        try:
            ann = wfdb.rdann(
                record_name,
                suffix.lstrip("."),
                pn_dir=None
            )

            symbols = sorted(
                set(
                    str(x)
                    for x in ann.symbol
                )
            )

            item["annotations"].append(
                {
                    "file": ann_file.name,
                    "extension": suffix,
                    "count": int(len(ann.sample)),
                    "symbols": symbols,
                    "first_sample": (
                        int(ann.sample[0])
                        if len(ann.sample) > 0
                        else None
                    ),
                    "last_sample": (
                        int(ann.sample[-1])
                        if len(ann.sample) > 0
                        else None
                    ),
                }
            )

        except Exception as exc:
            item["annotation_errors"].append(
                {
                    "file": ann_file.name,
                    "error": str(exc),
                }
            )

    results.append(item)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

records_with_annotations = [
    x for x in results
    if len(x["annotations"]) > 0
]

records_without_annotations = [
    x for x in results
    if len(x["annotations"]) == 0
]


print()
print(f"Records inspected:              {len(results)}")
print(
    f"Records with annotations:       "
    f"{len(records_with_annotations)}"
)
print(
    f"Records without annotations:    "
    f"{len(records_without_annotations)}"
)


print()
print("ANNOTATION TYPES")
print("-" * 70)

annotation_summary = {}

for item in results:
    for ann in item["annotations"]:

        ext = ann["extension"]

        if ext not in annotation_summary:
            annotation_summary[ext] = {
                "records": 0,
                "total_annotations": 0,
                "symbols": set(),
            }

        annotation_summary[ext]["records"] += 1
        annotation_summary[ext]["total_annotations"] += ann["count"]
        annotation_summary[ext]["symbols"].update(
            ann["symbols"]
        )


for ext, info in sorted(annotation_summary.items()):

    print(
        f"{ext}: "
        f"records={info['records']}, "
        f"annotations={info['total_annotations']}, "
        f"symbols={sorted(info['symbols'])}"
    )


# Convert sets for JSON.
for ext in annotation_summary:
    annotation_summary[ext]["symbols"] = sorted(
        annotation_summary[ext]["symbols"]
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output = {
    "dataset": "ctu_uhb_ctg",
    "raw_directory": str(RAW_DIR),
    "records_inspected": len(results),
    "records_with_annotations": len(
        records_with_annotations
    ),
    "records_without_annotations": len(
        records_without_annotations
    ),
    "annotation_summary": annotation_summary,
    "records": results,
}


output_path = OUTPUT_DIR / "annotation_inspection.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=2
    )


print()
print("=" * 70)
print(f"Saved: {output_path}")
print("=" * 70)