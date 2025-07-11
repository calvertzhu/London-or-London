import pandas as pd
from pathlib import Path
import shutil
from data_collection.config import DATA_DIR, PROJECT_ROOT

def balance_dataset(
    metadata_path=PROJECT_ROOT / "metadata/combined.csv",
    output_csv=PROJECT_ROOT / "metadata/balanced_subset.csv",
    output_dir=PROJECT_ROOT / "data_balanced",
    n_per_class=10,
    copy_images=True,
    run_id=None
):
    metadata_path = Path(metadata_path)
    output_csv = Path(output_csv)
    output_dir = Path(output_dir)

    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return

    df = pd.read_csv(metadata_path)

    if run_id:
        if "run_id" not in df.columns:
            print("No 'run_id' column in metadata. Skipping filter.")
        else:
            df = df[df["run_id"] == run_id]
            print(f"Filtering by run_id: {run_id} — {len(df)} rows remain")

    if "source" not in df.columns:
        df["source"] = "original"

    if df.empty:
        print("No data to balance after filtering.")
        return

    balanced_rows = []
    for (city, season, sharpness), group in df.groupby(["city", "season", "sharpness"]):
        originals = group[group["source"] != "augmented"]
        augments = group[group["source"] == "augmented"]

        needed = min(n_per_class, len(group))
        if len(originals) >= needed:
            sampled = originals.sample(n=needed, random_state=42)
        else:
            remaining = needed - len(originals)
            sampled = pd.concat([
                originals,
                augments.sample(n=min(remaining, len(augments)), random_state=42)
            ])

        print(f"Class {city}-{season}-{sharpness}: {len(sampled)} samples")
        balanced_rows.append(sampled)

    balanced_df = pd.concat(balanced_rows).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_csv, index=False)
    print(f"\nSaved balanced metadata to {output_csv}")

    if copy_images:
        missing = 0
        copied = 0
        for row in balanced_df.itertuples(index=False):
            src = DATA_DIR / row.city / row.season / row.sharpness / row.filename
            dst = output_dir / row.city / row.season / row.sharpness / row.filename
            dst.parent.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                print(f"Missing file: {src}")
                missing += 1
                continue

            try:
                shutil.copy(src, dst)
                copied += 1
            except Exception as e:
                print(f"Copy failed: {src} → {dst} — {e}")
                missing += 1

        print(f"\nCopied {copied} images to {output_dir}")
        if missing:
            print(f"{missing} images were skipped (missing or failed copy)")

    print("\nFinal sample counts per class:")
    print(balanced_df.groupby(["city", "season", "sharpness"]).size())

if __name__ == "__main__":
    balance_dataset()
